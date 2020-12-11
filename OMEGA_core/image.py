__author__ = "jz-rolling"

import numpy as np
import tifffile
import nd2reader as nd2
from . import helper
from .cluster import Cluster
from skimage import filters, morphology, registration
import pickle as pk

Version = '0.1.1'


class Image:
    """
    Read image data in the format of .nd2 hyperstack or .tif hyperstack (eg. by imageJ)

    """
    def __init__(self):

        """
        default perimeters for our microscope, changed accordingly when metadata is available

        pixel_microns
        ------- distance [um] per unit pixel

        shape
        ------- image shape

        filename 
        ------- if not specified, default filename is set as the file header of the input image file

        data
        ------- dictionary that stores different 2D imaging data by there corresponding channel names

        channels
        ------- image channel names

        mask_channel_name
        ------- channel name of the mask/bright field imaging data

        mask_binary
        ------- binary mask created by adaptive thresholding (local + global)

        mask_sobel
        ------- edge energy map using Sobel filter

        mask_clumpy_zone
        ------- energy map highlighting regions where cells aggregate notoriously

        mask_shape_indexed
        ------- 8-bit shape index map. Koenderink & Doorn, 1992

        mask_shape_indexed_smoothed
        ------- 8-bit shape index map smoothed using a median filter

        mask_labeled_clusters
        ------- labeled mini-clusters

        cluster_regionprops
        ------- regionprop dict of labeled mini-clusters

        clusters
        ------- filtered clusters that contain >= 1 cell like object(s)
        """
        self.pixel_microns = 0.065
        self.shape = [2048, 2048]

        self.filename = None
        self.data = {}
        self.channels = []
        self.mask_channel_name = '' # String key to retrieve mask/bright field imaging data
        self.mask_binary = None # Binary mask created using combinatory thresholding (local + global)
        self.mask_sobel = None # Edge energy map using Sobel filter
        self.mask_frangi = None # ridge detector
        self.mask_DoG = None # DoG filter
        self.mask_RoG = None # RoG filter
        self.mask_clumpy_zone = None
        self.mask_shape_indexed = None
        self.mask_shape_indexed_smoothed = None
        self.mask_labeled_clusters = None
        self.cluster_regionprops = {}
        self.clusters = {}
        self.boundary_stats = []

        self.image_background_median = {}
        self.image_foreground_median = {}
        self.image_background_std = {}
        self.image_foreground_std = {}
        self.config = None
        self.boundary_prediction_model = None
        self.boundary_normalizer = None
        self.cell_list = []


    def read_tiff_file(self, tiff, channels=None, filename=None, mask_channel_id=-1):

        """
        import image data from .tif file format, ideally preprocessed by imageJ
        :param tiff: input .tif hyperstack file
        :param channels: user specified channel names
        :param filename: user specified file header
        :param mask_channel_id: user specified key for retrieving bright field image

        """

        # load .tiff metafile if available
        tiff_data = tifffile.TiffFile(tiff)
        tiff_info = tiff_data.imagej_metadata["Info"]
        tiff_info = helper.ImageJinfo2dict(tiff_info)
        self.pixel_microns = float(tiff_info["dCalibration"])

        if 'channels' in tiff_data.imagej_metadata:
            _nChannels = int(tiff_data.imagej_metadata["channels"])
        else:
            _nChannels = 1

        if "Ranges" in tiff_data.imagej_metadata:
            tiff_bitdepth = tiff_data.imagej_metadata["Ranges"]
        else:
            bitdepth = 2**(int(tiff_info['BitsPerPixel']))-1
            tiff_bitdepth = []
            for i in range(_nChannels):
                tiff_bitdepth.append(0)
                tiff_bitdepth.append(bitdepth)

        if filename == None:
            self.filename = tiff.split("/")[-1]
        else:
            self.filename = filename

        if channels == None:
            channels = ['C{}'.format(x) for x in range(1,_nChannels+1)]
            self.channels = channels

        elif len(channels) != _nChannels:
            raise ValueError("The number of channels should match that of the input hyperstack.")

        self.mask_channel_name = self.channels[mask_channel_id]

        for i in range(_nChannels):
            fl_bit = tiff_bitdepth[2 * i + 1]
            self.data[channels[i]] = ((tiff_data.asarray(i) / fl_bit) * 65535).astype(np.uint16)

        # load image shape
        self.shape = self.data[self.mask_channel_name].shape

    def read_nd2_file(self, nd2file, filename=None, mask_channel_id=-1):

        """
        import image data directly from .nd2 file format
        :param nd2file: input .nd2 hyperstack image
        :param filename: user specified file header
        :param mask_channel_id: user specified key for retrieving bright field image

        """

        if filename == None:
            self.filename = nd2file.split('/')[-1]
        else:
            self.filename = filename

        # import data directly from .nd2 files
        if not nd2file.endswith(".nd2"):
            raise ValueError("Illegal input: {} is not a .nd2 file, try .read_tiff_file instead.".format(nd2file))
        nd2data = nd2.ND2Reader(nd2file)
        self.pixel_microns = nd2data.metadata["pixel_microns"]
        self.channels = nd2data.metadata["channels"]
        self.filename = filename
        self.mask_channel_name = self.channels[mask_channel_id]

        for channel in self.channels:
            self.data[channel] = np.asarray(nd2data[self.channels.index(channel)]).astype(np.uint16)

        # load image shape
        self.shape = self.data[self.mask_channel_name].shape

    def crop_edge(self, offset_correction=False, max_drift=2.5):

        """
        crop edges off the image
        :param offset_correction: correct for xy drift between phase contrast and fluorescent images if True
        :param max_drift.
        :return:
        """

        cropped = float(self.config['image']['crop_edge'])
        if self.data is None:
            raise ValueError("No images found!")

        if 0 <= cropped < 0.4:
            crop_width = int(cropped * self.shape[1])
            crop_height = int(cropped * self.shape[0])
            w1, w2 = crop_width, self.shape[1] - crop_width
            h1, h2 = crop_height, self.shape[0] - crop_height
            self.shape = (h2 - h1, w2 - w1)
        else:
            raise ValueError('Edge fraction should be no higher than 0.4 (40% from each side)!')

        if offset_correction:
            reference_image = self.data[self.mask_channel_name]
            reference_image = 100 + reference_image.max() - reference_image
            for channel, data in self.data.items():
                if channel != self.mask_channel_name:
                    shift, error, _diff = registration.phase_cross_correlation(reference_image, data,
                                                                               upsample_factor=10)
                    if max(np.abs(shift)) <= max_drift:
                        offset_image = helper.shift_image(data, shift)
                        self.data[channel] = offset_image[h1:h2, w1:w2]
                    else:
                        self.data[channel] = self.data[channel][h1:h2, w1:w2]
                else:
                    self.data[channel] = self.data[channel][h1:h2, w1:w2]

        else:
            for channel, data in self.data.items():
                self.data[channel] = self.data[channel][h1:h2, w1:w2]

    def enhance_brightfield(self, normalize=True, gamma=1.0):

        """
        attenuates signal aberrations in phase contrast image using FFT bandpass filters
        :param normalize: adjust exposure of brightfield image
        :param gamma: user specified gamma correction value, default = 1

        """
        maskimg = self.data[self.mask_channel_name].copy()
        mask_fft = helper.fft(maskimg, subtract_mean=True)
        fft_filters = helper.bandpass_filter(pixel_microns=self.pixel_microns,
                                             img_width=self.shape[1], img_height=self.shape[0],
                                             high_pass_width=float(self.config['image']['bandpass_high']),
                                             low_pass_width=float(self.config['image']['bandpass_low']))
        fft_reconstructed = helper.fft_reconstruction(mask_fft, fft_filters)

        if normalize:
            fft_reconstructed = helper.adjust_image(fft_reconstructed, adjust_gamma=True, gamma=gamma)

        self.data[self.mask_channel_name] = fft_reconstructed

        # Sobel edge operator
        sobel_filtered = filters.gaussian(filters.sobel(fft_reconstructed), sigma=1)
        self.mask_sobel = (sobel_filtered - sobel_filtered.min() + 1) / (sobel_filtered.max() + 1)

        # Frangi ridge operator
        self.mask_frangi = filters.gaussian(filters.frangi(fft_reconstructed,
                                                           black_ridges=True,
                                                           sigmas=(0.5, 2, 0.1)), sigma=1)
        _high_sigma_smoothed = filters.gaussian(fft_reconstructed, sigma=10)
        self.mask_RoG = fft_reconstructed/_high_sigma_smoothed
        self.mask_DoG = fft_reconstructed-_high_sigma_smoothed

        del mask_fft, fft_filters, fft_reconstructed, sobel_filtered

    def enhance_fluorescence(self,
                             normalize=False,
                             adjust_gamma=False,
                             gamma=1.0):
        """
        remove fluorescence background using the rolling ball method
        :param normalize: adjust exposure and data depth
        :param adjust_gamma: apply user specified gamma correction if True
        :param gamma: user specified gama correction value, default = 1
        :return:
        """

        for channel,data in self.data.items():
            if channel != self.mask_channel_name:
                bg_subtracted = helper.rolling_ball_bg_subtraction(data)
                if normalize:
                    bg_subtracted = helper.adjust_image(bg_subtracted,adjust_gamma=adjust_gamma,gamma=gamma)
                self.data[channel] = (filters.gaussian(bg_subtracted,sigma=0.5)*65535).astype(np.uint16)
                del bg_subtracted


    def locate_clusters(self):
        """
        primary segmentation, splits binary mask into patches (Cluster)
        """

        """
        initial segmentation
        mask_binary denotes the binary mask of cells;
        mask_shape_indexed denotes the shape_indexed surface curvature map of the phase contrast image;
        mask_labeled_clusters is a labeled image of which each unique integer (1->N) denotes a patch of pixels 
        that makes a Cluster.
        """

        """
        import configurations
        """
        clump_threshold = float(self.config['image']['clump_threshold'])
        clump_sigma = float(self.config['image']['clump_sigma'])
        shape_index_sigma = float(self.config['image']['shape_index_sigma'])
        area_pixel_low = float(self.config['image']['area_pixel_low'])

        """
        primary segmentation
        """
        self.mask_binary, \
        self.mask_shape_indexed, \
        self.mask_labeled_clusters, \
        regionprops = helper.init_segmentation(self.data[self.mask_channel_name],
                                               shape_indexing_sigma=shape_index_sigma,
                                               min_particle_size=area_pixel_low)

        """
        mask_clumpy_zone takes a DoG transformed phase contrast image and marks the region where 
        cells are likely forming clumps or overlapping. 
        """
        _flattened = self.mask_RoG[np.where(self.mask_binary>0)]
        self.mask_clumpy_zone = self.mask_RoG > \
                                (np.mean(_flattened)+clump_threshold*np.std(_flattened))
        self.mask_clumpy_zone = filters.gaussian(self.mask_clumpy_zone,
                                                 sigma=clump_sigma)

        """
        create a dictionary of regional information of the binary mask
        """
        for regionprop in regionprops:
            self.cluster_regionprops[regionprop.label] = regionprop
        self.mask_shape_indexed_smoothed = filters.median(self.mask_shape_indexed,
                                                          morphology.disk(2)).astype(np.uint8)

        """
        calculate global signal metrics
        """
        for channel, data in self.data.items():
            foreground = np.where(self.mask_binary > 0)
            background = np.where(self.mask_binary == 0)
            bg_data = data[background]
            fg_data = data[foreground]
            self.image_background_median[channel] = np.median(bg_data)
            self.image_foreground_median[channel] = np.median(fg_data)
            self.image_background_std[channel] = np.std(bg_data)
            self.image_foreground_std[channel] = np.std(fg_data)
        del bg_data, fg_data, foreground, background, regionprops

    def cluster_segmentation(self, predictor, normalizer,
                             boundary_annotation=False,
                             apply_median_filter=False):
        """

        :param predictor:
        :param normalizer:
        :param boundary_annotation:
        :param apply_median_filter:
        :return:
        """

        for idx, regionprop in self.cluster_regionprops.items():
            cluster = Cluster(self, regionprop)
            cluster.create_seeds(self, apply_median_filter=apply_median_filter)
            cluster.segmentation()
            cluster.compute_boundary_metrics(self)
            self.clusters[idx] = cluster

        # boundary prediction with trained model
        if not boundary_annotation:
            self.boundary_prediction_model = pk.load(open(predictor, 'rb'))
            self.boundary_normalizer = pk.load(open(normalizer, 'rb'))
            if len(self.boundary_stats) > 0:
                if len(self.boundary_stats) == 1:
                    self.boundary_stats = np.array(self.boundary_stats).reshape(1, -1)
                norm_boundary_stats = self.boundary_normalizer.transform(self.boundary_stats)
                boundary_prediction = self.boundary_prediction_model.predict(norm_boundary_stats)
                counter = 0
                for idx, cluster in self.clusters.items():
                    counter = cluster.boundary_classification(counter, boundary_prediction)
                    cluster.remove_false_boundaries()

    def cell_segmentation(self,
                          split_branches=True,
                          shapeindex_quality=True,
                          filter_by_shapeindex_quality=True):
        """
        evaluates the accuracy of Cluster segmentation, accepts or rejects Cell objects based
        on the computed multimetrics.
        :param split_branches: forces the subdivision of a non-rod-shaped particle
        :param shapeindex_quality: additional segmentation quality assessment using polar shape index measurements.
        :param filter_by_shapeindex_quality:
        :return:
        """
        for idx, cluster in self.clusters.items():
            if not cluster.discarded:
                cluster.filter_particles(self)
                if split_branches:
                    cluster.split_branches(self)
                cluster.measure_cells(polar_shapeindex=shapeindex_quality,
                                      reject_low_quality_cells=filter_by_shapeindex_quality)

    def get_cells(self,
                  discarded=False,
                  branched=False,
                  return_list=True):
        self.cell_list=[]
        for cluster_key, cluster in self.clusters.items():
            for cell_key, cell in cluster.cells.items():
                if cell.discarded:
                    if discarded:
                        self.cell_list.append(cell)
                elif cell.branched:
                    if branched:
                        self.cell_list.append(cell)
                else:
                    self.cell_list.append(cell)

        if return_list:
            return self.cell_list


def _test_stitched_data(tiff, shape=(5, 5), channels=None,
                        filename=None, mask_channel_id=0):
    """
    temporary handle for stitched image files.
    :param tiff: input .tif hyperstack file
    :param shape: N X M stitched images
    :param channels: user specified channel names
    :param filename: user specified file header
    :param mask_channel_id: user specified key for retrieving bright field image
    """


    """
    import data with tifffile
    """
    tiff_data = tifffile.TiffFile(tiff)
    tiff_info = tiff_data.imagej_metadata

    if filename == None:
        filename = tiff.split("/")[-1]
    else:
        filename = filename

    # load .tiff metafile if available
    pixel_microns = float(tiff_info["spacing"])
    _nChannels = int(tiff_info["channels"])

    if channels == None:
        channels = ['C{}'.format(x) for x in range(1, _nChannels + 1)]

    elif len(channels) != _nChannels:
        raise ValueError("The number of channels should match that of the input hyperstack.")

    mask_channel_name = channels[mask_channel_id]
    d_data = tiff_data.asarray()
    _c, h, w= d_data.shape

    if _c != len(channels):
        raise ValueError("The number of channels should match that of the input hyperstack.")

    if (h % shape[0] + w % shape[1]) != 0:
        raise ValueError("Error found when OMEGA attempted image splitting!")

    """
    user defined splitting parameters
    """
    unit_h = int(h/shape[0])
    unit_w = int(w/shape[0])

    image_count = 0
    pooled_images = []

    for i in range(0, h, unit_h):
        for j in range(0, w, unit_w):
            img = Image()
            img.filename = '{}_{}'.format(filename, image_count)
            img.pixel_microns = pixel_microns
            img.channels = channels
            img.mask_channel_name = mask_channel_name
            img.shape = [unit_h, unit_w]
            for c in range(_nChannels):
                channel = channels[c]
                img.data[channel] = d_data[c, i:i+unit_h, j:j+unit_w]
            pooled_images.append(img)
            image_count += 1

    return pooled_images
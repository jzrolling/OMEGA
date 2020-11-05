from .helper import *
from .data import Measurement
from . import io


class Cell:

    def __init__(self, image, cluster_label, cell_label, mask, bbox, regionprop=None):
        """
        As the name implies, the Cell object takes in particles from each Cluster, computes the corresponding
        contour line and topological skeleton (midline) as well as many other measurements. It also contains a
        set of filter functions which determines whether a cell is to be accepted as a single bacterium or not.

        :param image: master image object
        :param cluster_label: corresponding cluster id
        :param cell_label: cell id
        :param mask: binary mask
        :param bbox: refined bounding box
        :param regionprop: generated using mask if not provided
        """

        if regionprop != None:
            self.regionprop = regionprop
        else:
            self.regionprop = measure.regionprops(label_image=mask)[0]

        self.mask = ndi.median_filter(mask, 3)
        self.pixel_microns = image.pixel_microns
        self.bbox = bbox
        (x1, y1, x2, y2) = self.bbox
        self.mask_channel_name = image.mask_channel_name
        self.data = {}
        self.image_stat = {}
        for channel, data in image.data.items():
            self.data[channel] = data[x1:x2,y1:y2].copy()
            background_median = image.image_background_median[channel]
            foreground_median = image.image_foreground_median[channel]
            self.image_stat[channel] = [background_median, foreground_median]
        self.sobel = image.mask_sobel[x1:x2,y1:y2].copy()
        self.shape_index = image.mask_shape_indexed[x1:x2,y1:y2].copy()
        self.cluster_label = cluster_label
        self.cell_label = cell_label
        self.image_label = None
        self.contour = []
        self.config = image.config
        self.skeleton_coords = []
        self.skeleton = None
        self.branched = True
        self.discarded = False
        self.optimized_contour = []
        self._contour_optimization_converged = None
        self.largest_bend_angle = 0
        self.midlines = []
        self.width_lists = []
        self.roughness = 1
        self.lengths = []
        self.measurements = Measurement()
        self.vshaped = False
        self.fluorescent_puncta = {}
        self.segmentation_quality_by_shapeindex = 1

    def find_contour(self):

        """
        masked gray scale image is created by replacing background (mask==0) pixels with foreground maxima (mask==1)
        the masked image is then smoothed using a gaussian filter
        incipient contour is generated using a marching square algorithm implemented in skimage.measure

        skimage.measure.find_contours occasionally return tandem duplicated contour, coordinates.
        this is probably introduced while closing the contour
        postprocessing inspection is therefore necessary
        """

        masked_phase = self.mask * self.data[self.mask_channel_name]
        masked_phase[self.mask == 0] = int(masked_phase.max())
        masked_phase = filters.gaussian(masked_phase, sigma=1)
        lvl = np.percentile(masked_phase[self.mask > 0], 98)

        # only the outer contour enclosing the entire object is considered
        contour = measure.find_contours(masked_phase, level=lvl)[0]

        # find contour closing point
        x0, y0 = contour[0]
        closing_coords = np.where((contour[:, 0] == x0) & (contour[:, 1] == y0))[0]

        # in case the first two points are identical
        if len(closing_coords) == 1:
            contour = np.concatenate([contour,[contour[0]]])
        else:
            if closing_coords[1] == 1:
                p1, p2 = closing_coords[1], closing_coords[2]+1
                contour = contour[p1:p2]
            else:
                p1, p2 = closing_coords[0], closing_coords[1]+1
                contour = contour[p1:p2]

        # note that the recorded contour is by default closed
        contour_length = measure_length(contour, pixel_microns=1)
        self.contour = spline_approximation(contour, n=int(0.5*contour_length), smooth_factor=2)
        self.mask = draw.polygon2mask(self.mask.shape, self.contour)

    def extract_skeleton(self):

        """
        Extracts the pixelated topological skeleton of the binary mask
        note that the skimage embedded skeletonize function is sensitive
        to small extrusions near the edge. Although the binary mask was smoothed
        using a binary median filter, it is often not sufficient to suppress aberrant
        skeleton formation.

        Similar to MicrobeJ, OMEGA regards branched skeleton as a connected planar graph.
        The current implementation is suboptimal, as the graph is recorded using a
        default dictionary structure. For future versions, the Extract_skeleton function
        will be reconstructed with a NetworkX embedding.

        Cells with a severely bent skeleton is labeled as v-shaped. Notably, mycobacterial
        cell division is often succeeded by an abrupt force at the division site which
        immediately bends the divided cell and creates an angle of various degrees between the
        two sibling cells. Here we used a fixed cutoff (default value set to 50 degrees)
        to determine whether a particle is V-shaped, therefore a candidate for a microcluster
        containing two newly separated bacteria. The v-shaped particles are further segmented
        if the Cluster.split_branches function is being called.

        """
        if not self.discarded:
            self.skeleton_coords, self.skeleton = skeleton_analysis(self.mask)
            if len(self.skeleton_coords) == 1:
                xcoords = self.skeleton_coords[0][1]
                ycoords = self.skeleton_coords[0][2]

                # bend_angle function creates a moving window by the size of 2 * window.
                # skeleton shorter than 12 pixels (which should be discarded in most cases)
                # can not be properly handled.


                if len(xcoords) >= 12:
                    skel_coords = np.array([xcoords, ycoords]).T
                    self.largest_bend_angle = np.abs(bend_angle_open(skel_coords, window=5)).max()
                    if self.largest_bend_angle < float(self.config['cell']['largest_bend_angle']):
                        self.branched = False
                    else:
                        self.vshaped = True
            elif len(self.skeleton_coords) == 0:
                self.discarded = True

    def optimize_contour(self, optimize = True):
        """

        """
        self.optimized_contour, self._contour_optimization_converged = contour_optimization(self.contour,
                                                                                            self.sobel)
        if not optimize:
            self.optimized_contour = self.contour
            self._contour_optimization_converged = True

        self.measurements.particle_morphology(self)

    def generate_midline_no_branch(self, fast_mode=False):
        if not self.branched and not self.discarded:
            pole1, pole2 = self.skeleton_coords[0][0]
            xcoords = self.skeleton_coords[0][1]
            ycoords = self.skeleton_coords[0][2]
            if len(xcoords) <= 10:
                self.discarded = True
            else:
                if len(xcoords) <= 20:
                    interpolation_factor = 1
                else:
                    interpolation_factor = 2
                skel_coords = np.array([xcoords, ycoords]).T
                smooth_skel = spline_approximation(skel_coords, n=int(len(skel_coords)/interpolation_factor),
                                                   smooth_factor=8, closed=False)
                if not fast_mode:
                    smooth_skel, _converged = midline_approximation(smooth_skel, self.optimized_contour)
                    midline = extend_skeleton(smooth_skel, self.optimized_contour,
                                              find_pole1=pole1, find_pole2=pole2,
                                              interpolation_factor=interpolation_factor)
                else:
                    midline = extend_skeleton(smooth_skel, self.optimized_contour,
                                              find_pole1=pole1, find_pole2=pole2,
                                              interpolation_factor=1)

                width = direct_intersect_distance(midline, self.optimized_contour)
                length = line_length(midline)
                self.lengths.append(length*self.pixel_microns)
                self.midlines.append(midline)
                self.width_lists.append(width)

    def morphological_filter(self):
        morph_filter_keys = ['eccentricity', 'solidity', 'circularity',
                             'convexity', 'average_bending_energy',
                             'normalized_bending_energy', 'minimum_negative_concave',
                             'rough_Length', 'rough_sinuosity']
        is_cell = True
        for key in morph_filter_keys:
            val = self.measurements.morphology_measurements[key]
            low_threshold_config_key = key.lower()+'_low'
            high_threshold_config_key = key.lower()+'_high'
            low_threshold = float(self.config['cell'][low_threshold_config_key])
            high_threshold = float(self.config['cell'][high_threshold_config_key])
            is_cell *= in_range(val, low_threshold, high_threshold)
        if not is_cell:
            self.branched = True

    def generate_midline_branch_aware(self):
        print('Function currently unavailable')

    # future function

    def generate_measurements(self):
        # does not support branched cell measurements yet
        # will include graph based branch analysis shortly
        if not self.branched and not self.discarded:
            if self.midlines == [] or self.width_lists == []:
                print('No contour/midline coordinates found for cell {}-{}'.format(self.cluster_label, self.cell_label))
                self.discarded = True
            else:
                try:
                    self.measurements.cell_morphology(self)
                    midline = self.midlines[0]
                    width = self.width_lists[0]
                    mask = self.mask
                    length = self.lengths[0]
                    self.measurements.signal(self.shape_index, 'Shape_index',
                                             midline, width, mask, length)
                    for channel, img in self.data.items():
                        self.measurements.signal(img, channel, midline, width, mask, length)
                        if channel != self.mask_channel_name:
                            self.fluorescent_puncta[channel] = find_puncta(img, self.mask)
                except:
                    #print('odd found!')
                    self.discarded = True

    def compiled_cell_process(self):
        self.optimize_contour()
        self.morphological_filter()
        self.generate_midline_no_branch()
        self.generate_measurements()

    def _correct_xy_drift(self):
        if not self.branched and not self.discarded:
            # invert phase contrast data

            ref_img = self.data[self.mask_channel_name]
            ref_img = 100+ref_img.max()-ref_img

            for channel, img in self.data.items():
                if channel != self.mask_channel_name:
                    shift, error, _diff = feature.register_translation(ref_img, img, upsample_factor=10)
                    if np.max(np.abs(shift)) <= 2:
                        self.data[channel] = shift_image(img, shift)
                        if len(self.fluorescent_puncta[channel]) != 0:
                            puncta_info = self.fluorescent_puncta[channel]
                            puncta_info[:, 0] += shift[0]
                            puncta_info[:, 1] += shift[1]
                            self.fluorescent_puncta[channel] = puncta_info

    def _update_signal(self, channel):
        midline = self.midlines[0]
        width = self.width_lists[0]
        mask = self.mask
        length = self.lengths[0]
        img = self.data[channel]
        self.measurements.updata_signal(img, channel, midline, width, mask, length)







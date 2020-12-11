from .helper import *
from .measure import Measurement
from skimage import registration
from . import io


class Cell:

    def __init__(self,
                 image_obj,
                 cluster_label,
                 cell_label,
                 mask,
                 bbox,
                 regionprop=None):
        """
        As the name implies, the Cell object takes in particles from each Cluster, computes the corresponding
        contour line and topological skeleton (midline) as well as many other measurements. It also contains a
        set of filter functions which determines whether a cell is to be accepted as a single bacterium or not.

        :param image_obj: master Image object
        :param cluster_label: corresponding Cluster id
        :param cell_label: cell id
        :param mask: binary mask
        :param bbox: refined bounding box
        :param regionprop: skimage regionprops object, made from mask if not provided
        """

        if regionprop != None:
            self.regionprop = regionprop
        else:
            self.regionprop = measure.regionprops(label_image=mask)[0]

        self.mask = ndi.median_filter(mask, 3) # smooths the edge of the binary mask using a median kernel
        self.pixel_microns = image_obj.pixel_microns
        self.bbox = bbox
        self.mask_channel_name = image_obj.mask_channel_name

        """
        Inherit data from Image
        """
        self.data = {} # Cell image data
        self.image_stat = {} # inherit global stats from Image
        (x1, y1, x2, y2) = self.bbox
        for channel, data in image_obj.data.items():
            self.data[channel] = data[x1:x2, y1:y2].copy()
            background_median = image_obj.image_background_median[channel]
            foreground_median = image_obj.image_foreground_median[channel]
            self.image_stat[channel] = [background_median, foreground_median]
        self.sobel = image_obj.mask_sobel[x1:x2, y1:y2]
        self.frangi = image_obj.mask_frangi[x1:x2, y1:y2]
        self.shape_index = image_obj.mask_shape_indexed[x1:x2, y1:y2]
        self.config = image_obj.config

        self.cluster_label = cluster_label
        self.cell_label = cell_label
        self.image_label = None
        self.init_contour = [] # list of cellular contour(s)
        self.skeleton_coords = [] # list of morphological skeleton(s)
        self.skeleton = None # binary skeleton image
        self.optimized_contour = [] # subpixel contour(s)
        self._contour_optimization_converged = None # whether or not the contour optimization process converges
        self.fluorescent_puncta = {} # dictionary of fluorescent puncta profile(s)
        self.midlines = [] # list of subpixel midlines
        self.width_lists = [] # width measures along midline
        self.measurements = Measurement() # initiate cellular Measurement
        self.profile_mesh = None
        self.contour_l1 = None
        self.contour_l2 = None
        self.widths_interp = None
        self.midline_interp = None

        """
        basic morphological metrics
        """
        self.largest_bend_angle = 0
        self.roughness = 1
        self.lengths = []
        self.vshaped = False
        self.branched = True
        self.discarded = False
        self.segmentation_quality_by_shapeindex = 1

        """
        test_mode
        """
        self.test_mode = bool(int(self.config['cell']['test_mode']))

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
        contour_simplification_factor = float(self.config['cell']['contour_simplification_factor'])

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
        #contour_length = measure_length(contour, pixel_microns=1)
        self.init_contour = spline_approximation(contour,
                                                 n=int(contour_simplification_factor*len(contour)),
                                                 smooth_factor=5)
        self.mask = draw.polygon2mask(self.mask.shape, self.init_contour)

    def extract_skeleton(self):

        """
        Extracts the pixelated topological skeleton of the binary mask
        note that the skimage embedded skeletonize function is sensitive
        to small extrusions near the edge. Although the binary mask was smoothed
        using a binary median filter, it is often not sufficient to suppress aberrant
        skeleton formation.

        Similar to MicrobeJ, OMEGA treats branched skeleton as a connected planar graph.
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
                skel_length = np.count_nonzero(self.skeleton)
                if skel_length <= 3 or skel_length >= 1000:
                    self.discarded = True
                    if self.test_mode:
                        print('{}_{}_no skeleton found!'.format(self.cluster_label, self.cell_label))

    def optimize_contour(self):
        """
        infer object contour with subpixel resolution
        """
        optimize = bool(int(self.config['cell']['optimize_contour']))
        contour_enlarge_factor = float(self.config['cell']['contour_enlarge_factor'])
        expanded_contour = expand_contour(self.init_contour, scale=0.3)

        if not optimize:
            """
            use the initial contour rendered by skimage.measure.find_contours without further optimization    
            """
            self.optimized_contour = spline_approximation(self.init_contour,
                                                          n=int(2*len(self.init_contour)),
                                                          smooth_factor=2)
            self._contour_optimization_converged = True
        else:
            self.optimized_contour,\
            self._contour_optimization_converged = contour_optimization(expanded_contour,
                                                                        self.sobel)

        self.measurements.particle_morphology(self)

    def generate_midline_no_branch(self, fast_mode=False):
        """
        Our current midline algorithm doesn't work very well for branched objects
        future versions may include a graph embedded midline system that accounts for
        the various shapes
        :param fast_mode: extend the termini of the morphological skeleton to the poles and call it a day

        """
        if not self.branched and not self.discarded:
            pole1, pole2 = self.skeleton_coords[0][0]
            xcoords = self.skeleton_coords[0][1]
            ycoords = self.skeleton_coords[0][2]
            if len(xcoords) <= 10:
                if self.test_mode:
                    print('{}_{}_skeleton is too short!'.format(self.cluster_label,
                                                                self.cell_label))
                self.discarded = True
            else:
                if len(xcoords) <= 20:
                    interpolation_factor = 1
                else:
                    interpolation_factor = 2
                skel_coords = np.array([xcoords, ycoords]).T
                #
                smooth_skel = spline_approximation(skel_coords,
                                                   n=int(len(skel_coords)/interpolation_factor),
                                                   smooth_factor=5, closed=False)
                if not fast_mode:
                    smooth_skel, _converged = midline_approximation(smooth_skel,
                                                                    self.optimized_contour)
                    midline = extend_skeleton(smooth_skel, self.optimized_contour,
                                              find_pole1=pole1,
                                              find_pole2=pole2,
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
        """
        primary Cell object filter function using particle morphological descriptors. Empirically speaking,
        this function alone is sufficient to remove most of the falsely segmented objects, therefore no secondary
        vetting step is employed by the current edition of OMEGA, although an additional trained classifier could
        in theory further increase the segmentation accuracy.
        :return: None
        """
        morph_filter_keys = ['area', 'eccentricity',
                             'aspect_ratio',
                             'solidity',
                             'circularity',
                             'convexity',
                             'average_bending_energy',
                             'normalized_bending_energy',
                             'min_curvature',
                             'rough_Length',
                             'rough_sinuosity']
        like_cell = True
        for key in morph_filter_keys:
            val = self.measurements.morphology_measurements[key]
            low_threshold_config_key = key.lower()+'_low'
            high_threshold_config_key = key.lower()+'_high'
            low_threshold = float(self.config['cell'][low_threshold_config_key])
            high_threshold = float(self.config['cell'][high_threshold_config_key])
            like_cell *= in_range(val, low_threshold, high_threshold)

        if self.test_mode:
            print(self.vshaped, like_cell)
        if not like_cell:
            self.branched = True

    def generate_midline_branch_aware(self):
        # future function
        print('Function currently unavailable')

    def generate_measurements(self):
        """
        Signal and morphological profiling of Cell objects.
        :return: None
        """
        # does not support branched cell measurements yet
        # will include graph based branch analysis shortly

        if not self.branched and not self.discarded:
            if self.midlines == [] or self.width_lists == []:
                if self.test_mode:
                    print('No contour/midline coordinates found for cell {}-{}'.format(self.cluster_label,
                                                                                       self.cell_label))
                self.discarded = True
            else:
                correct_drift = bool(int(self.config['cell']['correct_cell_drift']))
                if correct_drift:
                    self._correct_xy_drift()
                try:
                    # signal profiling
                    unit_micron = float(self.config['cell']['profiling_unit_pixel'])
                    expanded_contour = expand_contour(self.optimized_contour, scale=0.5)
                    l1, l2, profiling_mesh,\
                    midline_interp = straighten_by_orthogonal_lines(expanded_contour,
                                                                    self.midlines[0],
                                                                    self.lengths[0],
                                                                    np.array(self.width_lists[0])*self.pixel_microns,
                                                                    unit_micron=unit_micron)

                    self.contour_l1 = l1
                    self.contour_l2 = l2
                    self.profile_mesh = profiling_mesh
                    self.midline_interp = midline_interp
                    self.widths_interp = np.sqrt(np.sum(np.square(l2-l1), axis=1))*self.pixel_microns
                    self.widths_interp = np.concatenate([[0], self.widths_interp, [0]])
                    self.measurements.signal(self, 'Shape_index', profiling_mesh)
                    for channel, img in self.data.items():
                        self.measurements.signal(self, channel, profiling_mesh)
                        if channel != self.mask_channel_name:
                            self.fluorescent_puncta[channel] = find_puncta(img, self.mask)

                    # morphological profiling
                    self.measurements.cell_morphology(self)
                except:
                    if self.test_mode:
                        print('{}_{}_contour optimization failed!'.format(self.cluster_label,
                                                                          self.cell_label))
                    self.discarded = True

    def compiled_cell_process(self):
        """
        Bundle of Cell functions
        :return:
        """
        self.optimize_contour()
        self.morphological_filter()
        self.generate_midline_no_branch()
        self.generate_measurements()
        # drop redundant data
        self.frangi = None
        self.sobel = None
        self.skeleton = None

    def _correct_xy_drift(self):
        """
        XY-drift correction at cellular level, may introduce numeric instability
        :return:
        """
        if not self.branched and not self.discarded:
            # invert phase contrast data

            ref_img = self.data[self.mask_channel_name]
            ref_img = 100+ref_img.max()-ref_img

            for channel, img in self.data.items():
                if channel != self.mask_channel_name:
                    shift, error, _diff = registration.phase_cross_correlation(ref_img, img,
                                                                               upsample_factor=10)
                    if np.max(np.abs(shift)) <= 3:
                        self.data[channel] = shift_image(img, shift)
                        if len(self.fluorescent_puncta) != 0:
                            puncta_info = self.fluorescent_puncta[channel]
                            puncta_info[:, 0] += shift[0]
                            puncta_info[:, 1] += shift[1]
                            self.fluorescent_puncta[channel] = puncta_info

    def _update_signal(self, channel):
        """
        post analysis signal update
        :param channel: data channel
        :return: None
        """
        midline = self.midlines[0]
        width = self.width_lists[0]
        mask = self.mask
        length = self.lengths[0]
        img = self.data[channel]
        self.measurements.update_signal(img, channel, midline, width, mask, length, self.profile_mesh)


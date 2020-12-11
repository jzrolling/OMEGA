from .helper import *
from .metrics import *


class Measurement:

    """
    The Measurement class generates multi-metrics signal profiles for each extracted Cell object.
    It contains the following
    """
    def __init__(self):
        """
        Attributes
        ==========

        signal_measurements
        ---------- image signal profiles

        morphology_measurements
        ---------- morphological profiles

        """
        #self.measure_midline = None
        #self.straighten = None
        #self.straghten_normalize_width = None
        #self.SNR = None
        self.signal_measurements = {}
        self.morphology_measurements = {}

    def signal(self, cell, channel, profiling_mesh):
        """
        signal profiling function

        :param cell: Cell object
        :param channel: channel name
        :param profiling_mesh: subpixel profiling mesh

        """
        if channel == 'Shape_index':
            data = cell.shape_index
        else:
            data = cell.data[channel]

        midline = cell.midlines[0]
        width = cell.width_lists[0]
        mask = cell.mask
        length = cell.lengths[0]

        if channel not in self.signal_measurements:
            channel_data = {}
            channel_data['axial'] = measure_along_strip(midline, data, width=5)
            channel_data['straighten'] = straighten_cell(data, midline, width)
            straighten_normalized = bilinear_interpolate_numpy(data, profiling_mesh[0], profiling_mesh[1]).T
            channel_data['straighten_normalized'] = straighten_normalized
            bg_removed = data[np.where(mask>0)]
            channel_data['median'] = np.median(bg_removed)
            channel_data['standard_deviation'] = np.std(bg_removed)
            channel_data['mean'] = np.mean(bg_removed)
            channel_data['max'] = np.max(bg_removed)
            channel_data['min'] = np.min(bg_removed)
            midline_kurtosis, midline_skewness = kurtosis_skewness(channel_data['axial'])
            channel_data['midline_skewness'] = midline_skewness
            channel_data['midline_kurtosis'] = midline_kurtosis
            cell_kurtosis, cell_skewness = kurtosis_skewness(data[np.where(mask>0)])
            channel_data['cell_kurtosis'] = cell_kurtosis
            channel_data['cell_skewness'] = cell_skewness
            channel_data['axial_symmetry'] = measure_symmetry(channel_data['axial'],weighted=True)
            channel_data['axial_mean'] = np.average(channel_data['straighten_normalized'],axis=0)
            channel_data['axial_std'] = np.std(channel_data['straighten_normalized'],axis=0)
            channel_data['lateral_mean'] = np.average(channel_data['straighten_normalized'],axis=1)
            channel_data['lateral_std'] = np.std(channel_data['straighten_normalized'], axis=1)
            FWHM, offset_score = FWHM_background_aware(channel_data['lateral_mean'])
            channel_data['lateral_FWHM'] = FWHM
            channel_data['lateral_center_offset'] = offset_score
            channel_data['normalized_lateral_FWHM'] = channel_data['lateral_FWHM']/len(channel_data['lateral_mean'])
            channel_data['lateral_symmetry'] = measure_symmetry(channel_data['lateral_mean'], weighted=True)
            channel_data['segmented_measurements'] = divede_cell_pos(channel_data['axial_mean'], length)
            self.signal_measurements[channel] = channel_data

    def update_signal(self, data,
                      channel,
                      midline,
                      width,
                      mask,
                      length,
                      profiling_mesh):
        """
        update existing data

        :param data: image data of interest
        :param channel: channel name
        :param midline: precomputed cell midline
        :param width: precomputed cell widths profile (unit: pixel)
        :param mask: precomputed cell binary mask
        :param length: cell length (unit: micron)
        :param profiling_mesh: precomputed orthogonal profiling mesh

        """
        channel_data = {}
        channel_data['axial'] = measure_along_strip(midline, data, width=5)
        channel_data['straighten'] = straighten_cell(data, midline, width)
        channel_data['straighten_normalized'] = bilinear_interpolate_numpy(data,
                                                                           profiling_mesh[0],
                                                                           profiling_mesh[1]).T
        bg_removed = data[np.where(mask > 0)]
        channel_data['median'] = np.median(bg_removed)
        channel_data['standard_deviation'] = np.std(bg_removed)
        channel_data['mean'] = np.mean(bg_removed)
        channel_data['max'] = np.max(bg_removed)
        channel_data['min'] = np.min(bg_removed)
        midline_kurtosis, midline_skewness = kurtosis_skewness(channel_data['axial'])
        channel_data['midline_skewness'] = midline_skewness
        channel_data['midline_kurtosis'] = midline_kurtosis
        cell_kurtosis, cell_skewness = kurtosis_skewness(data[np.where(mask > 0)])
        channel_data['cell_kurtosis'] = cell_kurtosis
        channel_data['cell_skewness'] = cell_skewness
        channel_data['axial_symmetry'] = measure_symmetry(channel_data['axial'], weighted=True)
        channel_data['axial_mean'] = np.average(channel_data['straighten_normalized'], axis=0)
        channel_data['axial_std'] = np.std(channel_data['straighten_normalized'], axis=0)
        channel_data['lateral_mean'] = np.average(channel_data['straighten_normalized'], axis=1)
        channel_data['lateral_std'] = np.std(channel_data['straighten_normalized'], axis=1)
        FWHM, offset_score = FWHM_background_aware(channel_data['lateral_mean'])
        channel_data['lateral_FWHM'] = FWHM
        channel_data['lateral_center_offset'] = offset_score
        channel_data['normalized_lateral_FWHM'] = channel_data['lateral_FWHM'] / len(channel_data['lateral_mean'])
        channel_data['lateral_symmetry'] = measure_symmetry(channel_data['lateral_mean'], weighted=True)
        channel_data['segmented_measurements'] = divede_cell_pos(channel_data['axial_mean'], length)
        self.signal_measurements[channel] = channel_data

    def particle_morphology(self, cell):
        """
        morphological properties of the binary mask of any extracted particle
        :param cell: corresponding Cell object
        """

        # basic morphological features
        self.morphology_measurements['area'] = cell.regionprop.area * (cell.pixel_microns**2)
        self.morphology_measurements['eccentricity'] = cell.regionprop.eccentricity
        self.morphology_measurements['aspect_ratio'] = cell.regionprop.minor_axis_length / \
                                                       cell.regionprop.major_axis_length
        self.morphology_measurements['solidity'] = cell.regionprop.solidity
        self.morphology_measurements['circularity'] = circularity(cell.regionprop)
        self.morphology_measurements['convexity'] = convexity(cell.regionprop)

        # note that here the rough contour rather than the optimized one is used
        self.morphology_measurements['average_bending_energy'] = bending_energy(cell.init_contour)
        self.morphology_measurements['normalized_bending_energy'] = normalized_contour_complexity(cell)

        angle_along_contour = bend_angle(cell.init_contour, window=5)
        # an additional 180 degrees are added to the min/max curvatures to suppress negativity
        self.morphology_measurements['min_curvature'] = (180+angle_along_contour.min())/180
        self.morphology_measurements['max_curvature'] = (180+angle_along_contour.max())/180
        self.morphology_measurements['std_curvature'] = np.abs(angle_along_contour).std()/180
        self.morphology_measurements['mean_curvature'] = np.abs(angle_along_contour).mean()/180

        #pixelated length estimation
        rough_length, major_axis_length = cell.skeleton.sum()*np.sqrt(2), cell.regionprop.major_axis_length
        self.morphology_measurements['rough_Length'] = max(rough_length, major_axis_length)*cell.pixel_microns
        self.morphology_measurements['rough_sinuosity'] = max(rough_length, major_axis_length)/major_axis_length
        if self.morphology_measurements['rough_sinuosity'] < 1:
            self.morphology_measurements['rough_sinuosity'] = 1
        self.morphology_measurements['branch_count'] = max(0, len(cell.skeleton_coords) - 2)

    def cell_morphology(self, cell):
        """
        morphological properties estimated with subpixel contour/midlines

        :param cell: Cell object

        """
        self.morphology_measurements['length'] = np.sum(cell.lengths)
        self.morphology_measurements['sinuosity'] = sinuosity(cell.midlines[0])

        # ignore widths measures from the polar 0.3 microns
        widths = np.array(cell.width_lists[0])
        dl = int(0.3*len(widths)/self.morphology_measurements['length'])
        filtered_widths = widths[dl:-dl] * cell.pixel_microns
        self.morphology_measurements['width_median'] = np.median(filtered_widths)
        self.morphology_measurements['width_std'] = np.std(filtered_widths)
        self.morphology_measurements['width_max'] = np.max(filtered_widths)
        self.morphology_measurements['width_min'] = np.max(filtered_widths)

        # find where the min and max width lies along the cell
        min_width_pos_fraction = np.argmin(filtered_widths)/len(filtered_widths)
        if min_width_pos_fraction >= 0.5:
            min_width_pos_fraction = 1-min_width_pos_fraction
        max_width_pos_fraction = np.argmax(filtered_widths)/len(filtered_widths)
        if max_width_pos_fraction >= 0.5:
            max_width_pos_fraction = 1-max_width_pos_fraction
        self.morphology_measurements['min_width_pos_fraction'] = min_width_pos_fraction
        self.morphology_measurements['max_width_pos_fraction'] = max_width_pos_fraction
        self.morphology_measurements['width_symmetry'] = measure_symmetry(filtered_widths, weighted=True)


    def data_to_dict(self, by='signal'):

        """
        export cellular metrics to data dictionaries

        :param by: selection of exported metric(s)
                   ----'signal' export only image signal metrics
                   ----'particle_morphology' export only basic morphological properties of a binary particle
                   ----'cell_morphology' export only cell-specific morphological properties
                   ----'morphology' export all morphological properties
                   ----'all' export all data
        :return: keys_dict
                 ------keys to retrieve data
                 output_dict
                 ------dictionaries corresponding to the selected metrics
        """

        signal_keys = ['median', 'mean', 'max', 'min', 'standard_deviation',
                       'midline_skewness', 'midline_kurtosis', 'cell_kurtosis', 'cell_skewness',
                       'axial_symmetry', 'lateral_FWHM', 'lateral_center_offset', 'normalized_lateral_FWHM',
                       'lateral_symmetry']

        particle_morphological_keys = ['eccentricity', 'solidity', 'circularity',
                                       'convexity', 'average_bending_energy',
                                       'min_curvature', 'max_curvature',
                                       'mean_curvature', 'std_curvature',
                                       'rough_Length', 'rough_sinuosity', 'branch_count']

        cell_morphological_keys = ['length', 'sinuosity',
                                   'width_median', 'width_std', 'width_max', 'width_min'
                                   'width_symmetry']

        output_dict = {}
        keys_dict = {}
        if by == 'signal':
            for channel in list(self.signal_measurements.keys()):
                output_dict[channel] = list(map(self.signal_measurements[channel].get, signal_keys))
                keys_dict[channel] = signal_keys
            return keys_dict, output_dict

        elif by == 'particle_morphology':
            output_dict[by] = list(map(self.morphology_measurements.get, particle_morphological_keys))
            keys_dict[by] = particle_morphological_keys
            return keys_dict, output_dict

        elif by == 'cell_morphology':
            output_dict[by] = list(map(self.morphology_measurements.get, cell_morphological_keys))
            keys_dict[by] = cell_morphological_keys
            return keys_dict, output_dict

        elif by == 'morphology':
            morphology_keys = particle_morphological_keys + cell_morphological_keys
            output_dict[by] = list(map(self.morphology_measurements.get, morphology_keys))
            keys_dict[by] = morphology_keys
            return keys_dict, output_dict

        elif by == 'all':
            for channel in list(self.signal_measurements.keys()):
                output_dict[channel] = list(map(self.signal_measurements[channel].get, signal_keys))
                keys_dict[channel] = signal_keys
            output_dict['morphology'] = list(map(self.morphology_measurements.get,
                                             particle_morphological_keys + cell_morphological_keys))
            keys_dict['morphology'] = particle_morphological_keys + cell_morphological_keys
            return keys_dict, output_dict

        else:
            text1 = "Illegal key name, try the following: "
            text2 = "'signal', 'particle_morphology', 'cell_morphology', 'morphology', 'all'."
            raise ValueError(text1+text2)


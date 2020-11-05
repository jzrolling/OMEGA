from .helper import *
from .metrics import *


class Measurement:

    def __init__(self):
        self.measure_midline = None
        self.straighten = None
        self.straghten_normalize_width = None
        self.SNR = None
        self.signal_measurements = {}
        self.morphology_measurements = {}

    def signal(self, data, channel, midline, width, mask, length):
        if channel not in self.signal_measurements:
            channel_data = {}
            channel_data['axial'] = measure_along_strip(midline, data, width=5)
            channel_data['straighten'] = straighten_cell(data, midline, width)
            channel_data['straighten_normalized'] = straighten_cell_normalize_width(data, midline, width)
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

    def updata_signal(self, data, channel, midline, width, mask, length):
        channel_data = {}
        channel_data['axial'] = measure_along_strip(midline, data, width=5)
        channel_data['straighten'] = straighten_cell(data, midline, width)
        channel_data['straighten_normalized'] = straighten_cell_normalize_width(data, midline, width)
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
        self.morphology_measurements['hu_moments'] = cell.regionprop.moments_hu
        self.morphology_measurements['eccentricity'] = cell.regionprop.eccentricity
        self.morphology_measurements['solidity'] = cell.regionprop.solidity
        self.morphology_measurements['circularity'] = circularity(cell.regionprop)
        self.morphology_measurements['convexity'] = convexity(cell.regionprop)
        self.morphology_measurements['average_bending_energy'] = average_bending_energy(cell.contour)
        self.morphology_measurements['normalized_bending_energy'] = normalized_contour_complexity(cell)
        self.morphology_measurements['minimum_negative_concave'] = bend_angle(cell.contour, window=2).min()
        rough_length, major_axis_length = cell.skeleton.sum()*np.sqrt(2), cell.regionprop.major_axis_length
        self.morphology_measurements['rough_Length'] = max(rough_length, major_axis_length)*cell.pixel_microns
        self.morphology_measurements['rough_sinuosity'] = max(rough_length, major_axis_length)/major_axis_length
        if self.morphology_measurements['rough_sinuosity'] < 1:
            self.morphology_measurements['rough_sinuosity'] = 1
        self.morphology_measurements['branch_count'] = max(0, len(cell.skeleton_coords) - 2)


    def cell_morphology(self, cell):
        self.morphology_measurements['length'] = np.sum(cell.lengths)
        self.morphology_measurements['sinuosity'] = sinuosity(cell.midlines[0])
        filtered_widths = width_omit_odds(cell.width_lists)
        self.morphology_measurements['width_median'] = np.median(filtered_widths) * cell.pixel_microns
        self.morphology_measurements['width_std'] = np.std(filtered_widths) * cell.pixel_microns
        self.morphology_measurements['width_symmetry'] = measure_symmetry(filtered_widths, weighted=True)



    def data_to_list(self, by = 'signal'):
        signal_keys = ['median', 'mean', 'max', 'min', 'standard_deviation',
                       'midline_skewness', 'midline_kurtosis', 'cell_kurtosis', 'cell_skewness',
                       'axial_symmetry', 'lateral_FWHM', 'lateral_center_offset', 'normalized_lateral_FWHM',
                       'lateral_symmetry']
        particle_morphological_keys = ['eccentricity', 'solidity', 'circularity',
                                       'convexity', 'average_bending_energy',
                                       'rough_Length', 'rough_sinuosity', 'branch_count']
        cell_morphological_keys = ['length', 'sinuosity', 'width_median', 'width_std',
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


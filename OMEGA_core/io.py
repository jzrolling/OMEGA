from .helper import *
from .metrics import *
from .image import Image, _test_stitched_data
from .config import *
from .plot import *
import timeit
import glob, os, tifffile
import pickle as pk
from scipy import stats
import multiprocessing as mp
from matplotlib import rc
from matplotlib.gridspec import GridSpec
import warnings


"""
===================================
functions for automated processes
===================================
"""


def process_single_input(imagefile,
                         header,
                         dest_folder,
                         mask_channel_id=-1,
                         configfile='default',
                         print_time=False,
                         offset_correction=True,
                         split_branches=True,
                         save_input=False,
                         save_segmentation_result=False,
                         save_cells=True,
                         save_discarded=False,
                         save_branched=False,
                         boundary_classifier='./OMEGA_core/configurations/boundary_MLP.pk',
                         boundary_normalizer='./OMEGA_core/configurations/normalizer.pk',
                         return_obj=True):
    """

    :param imagefile: input image file
    :param header: user specified identification label
    :param dest_folder: path where the specified output dataset will be exported to
    :param mask_channel_id: index of the reference image
    :param configfile: user specified configuration file dest
    :param print_time: print time cost when done if set to True
    :param offset_correction: correct xy drift if set to True
    :param split_branches: extensive segmentation if set to True
    :param save_input: save processed images if set to True
    :param save_segmentation_result: plot and save segmentation result if set to True
    :param save_cells: pickle a list of cells if set to True
    :param save_discarded: include discarded particles when pickling if set to True
    :param save_branched: include branched particles when pickling if set to True
    :param boundary_classifier: path to pretrained model for boundary classification
    :param boundary_normalizer: path to pretrained model for boundary data normalization
    :param return_obj: return the processed Image object when done if set to True
    :return return the processed Image object if return_obj is True

    """
    config = None
    t1 = timeit.default_timer()
    if configfile == 'default':
        config = Preset_configurations_default().config
    elif '.ini' in configfile:
        try:
            config = Preset_configurations_default(configfile=configfile).config
        except:
            print('{} config file not supported, proceed with default settings.'.format(configfile))
            config = Preset_configurations_default().config

    # activate an Image object, load configurations
    obj = Image()
    obj.config = config

    # load image data
    if imagefile.endswith('.nd2'):
        obj.read_nd2_file(imagefile, mask_channel_id=mask_channel_id)
    elif imagefile.endswith('.tif'):
        obj.read_tiff_file(imagefile, mask_channel_id=mask_channel_id)
    else:
        raise ValueError('.{} file not supported!'.format(imagefile.split('.')[-1]))

    # inspect and calibrate input data
    obj.crop_edge(offset_correction=offset_correction)
    obj.enhance_brightfield()
    obj.enhance_fluorescence()

    # segmentation
    obj.locate_clusters()
    obj.cluster_segmentation(predictor=boundary_classifier,
                             normalizer=boundary_normalizer)
    obj.cell_segmentation(split_branches=split_branches)

    # retrieve cells
    cell_list = obj.get_cells(branched=save_branched,
                              discarded=save_discarded,
                              return_list=True)
    t2 = timeit.default_timer()
    if print_time:
        print("Analysis complete, {} seconds used.".format(round(t2 - t1), 2))
    if save_input:
        save_processed_images(obj, header, dest_folder)
    if save_segmentation_result:
        plot_segmented_image(obj, savefig=True, dpi=100, lw=0.3, scale=0.4,
                             filename="{}{}_segmentation_result.tif".format(dest_folder, header))
    if save_cells:
        pickle_dump(cell_list, header+'_cells', dest_folder)
    if return_obj:
        return obj


def process_folder(source_folder,
                   dest_folder=None,
                   file_type='.nd2',
                   mask_channel_id=-1,
                   configfile='default',
                   print_time=False,
                   offset_correction=True,
                   split_branches=True,
                   save_input=False,
                   save_segmentation_result=True,
                   save_cells=True,
                   save_discarded=False,
                   save_branched=False,
                   boundary_classifier='./OMEGA_core/configurations/boundary_MLP.pk',
                   boundary_normalizer='./OMEGA_core/configurations/normalizer.pk',
                   n_cores=2):
    """

    :param source_folder: folder where input image files are located
    :param dest_folder: path where the specified output dataset will be exported to
    :param file_type: image file type, could be either .tiff or .nd2
    :param mask_channel_id: index of the reference image
    :param configfile: user specified configuration file dest
    :param print_time: print time cost when done if set to True
    :param offset_correction: correct xy drift if set to True
    :param split_branches: extensive segmentation if set to True
    :param save_input: save processed images if set to True
    :param save_segmentation_result: plot and save segmentation result if set to True
    :param save_cells: pickle a list of cells if set to True
    :param save_discarded: include discarded particles when pickling if set to True
    :param save_branched: include branched particles when pickling if set to True
    :param boundary_classifier: path to pretrained model for boundary classification
    :param boundary_normalizer: path to pretrained model for boundary data normalization
    :param n_cores: number of CPU cores called in parallel

    """
    warnings.filterwarnings("ignore")

    # setup output folder and locate image data
    if dest_folder is None:
        dest_folder = source_folder + 'output/'
    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)
    input_files = sorted(glob.glob(source_folder+'*'+file_type))
    if len(input_files) == 0:
        raise ValueError('No {} file(s) found in {}!'.format(file_type, source_folder))

    # multiprocessing
    for i in range(0, len(input_files), n_cores):
        processes = []
        for j in range(0, min(n_cores, len(input_files)-i)):
            file = input_files[i+j]
            header = file.split('/')[-1].split('.')[0]
            processes.append(mp.Process(target=process_single_input,
                                        args=(file, header, dest_folder,
                                              mask_channel_id,
                                              configfile,
                                              print_time,
                                              offset_correction,
                                              split_branches,
                                              save_input,
                                              save_segmentation_result,
                                              save_cells,
                                              save_discarded,
                                              save_branched,
                                              boundary_classifier,
                                              boundary_normalizer,
                                              False)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()


"""
========================================
functions for data import and export
========================================
"""


def pickle_dump(data, header, dest_folder):
    """
    pickle wrapper
    :param data: data of interest
    :param header: user specified file header/name
    :param dest_folder: destination folder
    :return: None
    """
    pk.dump(data, open("{}{}.pk".format(dest_folder, header), 'wb'))


def pickle_load_all_cells(folder):
    """
    load pickled cells from any folder
    :param folder: folder where *cells.pk files are stored
    :return: a list of retrieved Cell objects
    """
    pks = sorted(glob.glob(folder+'*cells.pk'))
    cells = []
    for file in pks:
        cells += pk.load(open(file, 'rb'))
    return cells


def save_processed_images(Image_obj, header, dest_folder):
    """
    save processed image data as 16-bit tif files
    :param obj: processed Image object
    :param header: user specified header/name
    :param dest_folder: destination folder
    :return: None
    """
    for channel, img in Image_obj.data.items():
        channel_name = channel.replace(' ', '_')
        tifffile.imsave("{}{}_{}_filtered.tif".format(dest_folder, header, channel_name),
                        img.astype(np.uint16), imagej=True)


def save_experiment(Image_obj, header, dest_folder):
    """
    save the entire Image object
    :param Image_obj: processed Image object
    :param header: user specified header/name
    :param dest_folder: destination folder
    :return: None
    """
    pickle_dump(Image_obj, header, dest_folder)


def get_measurements(cells,
                     header,
                     dest_folder):
    """
    save numerical measurements to excel sheets
    :param cells: list of cells
    :param header:
    :param dest_folder:
    :return:
    """
    branched_cells = []
    discarded_cells = []
    accepted_cells = []
    filtered_cell_dfs = {}
    outlier_id = []

    for i in range(len(cells)):
        cell = cells[i]
        if cell.discarded:
            discarded_cells.append(cell)
        elif cell.branched:
            branched_cells.append(cell)
        else:
            accepted_cells.append(cell)

    if discarded_cells is not []:
        discarded_df_dict = _to_dataframe(discarded_cells, by='particle_morphology')
        for key, df in discarded_df_dict.items():
            df.to_excel("{}{}_{}_{}.xls".format(dest_folder, header, 'discarded_cells', key))

    if branched_cells is not []:
        branched_df_dict = _to_dataframe(branched_cells, by='particle_morphology')
        for key, df in branched_df_dict.items():
            df.to_excel("{}{}_{}_{}.xls".format(dest_folder, header, 'branched_cells', key))

    if accepted_cells is not []:
        accepted_df_dict = _to_dataframe(accepted_cells, by='all')
        for key, df in accepted_df_dict.items():
            if key == 'morphology':
                filtered_df, outliers = _filter_outliers(df,
                                                         by=['length', 'width_median',
                                                             'width_std', 'circularity'])
            else:
                filtered_df, outliers = _filter_outliers(df,
                                                         by=['median'])
            filtered_df.to_excel("{}{}_{}_{}.xls".format(dest_folder, header, 'accepted_cells', key))
            filtered_cell_dfs[key] = filtered_df
            outlier_id.append(outliers)

    is_outlier = (np.sum(np.array(outlier_id), axis=0) > 0)*1
    return discarded_cells, branched_cells, accepted_cells, filtered_cell_dfs, is_outlier


def measure_by_folder(header,
                      dest_folder):
    """
    gather and export signal profiles
    :param dest_folder:
    :return:
    """

    cells = pickle_load_all_cells(dest_folder)
    discarded_cells, branched_cells, accepted_cells, \
    accepted_cell_dfs, outlier_id = get_measurements(cells, header, dest_folder)
    _miscellaneous = _export_data(accepted_cells,
                                  accepted_cell_dfs,
                                  outlier_id,
                                  header,
                                  dest_folder)

def _to_dataframe(cells,
                  by='all'):
    """
    convert cellular data to pandas dataframes
    :param cells: list of cells
    :param by: metrices of interest
    :return: dictionary of dataframes
    """
    all_keys = {}
    all_measurements = {}
    dataframe_dict = {}
    for cell in cells:
        image_label = cell.image_label
        cluster_label = cell.cluster_label
        cell_label = cell.cell_label
        key_dict, measurement_dict = cell.measurements.data_to_dict(by=by)
        for key, val in measurement_dict.items():
            keys = key_dict[key]
            if key not in all_keys:
                all_keys[key] = ['image_label', 'cluster_label', 'cell_label'] + keys
            if key not in all_measurements:
                all_measurements[key] = [[image_label, cluster_label, cell_label] + val]
            else:
                all_measurements[key].append([image_label, cluster_label, cell_label] + val)

    for key, matrix in all_measurements.items():
        columns = all_keys[key]
        dataframe_dict[key] = pd.DataFrame(matrix, columns=columns)
    return dataframe_dict


def _filter_outliers(df,
                     z_score_th=3,
                     iqr_th=2,
                     by=('length', 'width_median', 'width_std', 'circularity')):
    """
    combine outliers found by the z-score method of IQR method
    :param df: input multi-metrics dataframe
    :param z_score_th: z-score high bound
    :param iqr_th: IQR high bound
    :param by: metric keys of interest
    :return: updated dataframe, outlier identification list
    """

    # find outliers by zscore
    outlier_by_zscore = np.sum(stats.zscore(df[by], axis=0) >= z_score_th, axis=1) >= 1

    # find outliers by IQR
    q1,q3 = df[by].quantile(0.25), df[by].quantile(0.75)
    iqr = q3-q1
    outlier_by_iqr = np.sum(np.array((df[by] < q1-iqr_th*iqr) | (df[by] > q3+iqr_th*iqr)),
                            axis=1) >= 1
    df['is_outlier'] = ((outlier_by_iqr + outlier_by_zscore) > 0)*1
    return df, list(df['is_outlier'])


def _export_data(cell_list,
                 cell_dataframe_dict,
                 is_outlier,
                 header,
                 dest_folder,
                 save_data=True):

    cell_profiles = {}
    if 'morphology' not in cell_dataframe_dict:
        raise ValueError('Morphological dataframe not found!')

    morph_df = cell_dataframe_dict['morphology']
    if 'is_outlier' not in morph_df.columns:
        raise ValueError('Outlier information not found!')

    # record cell lengths and authenticity
    cell_profiles['is_outlier'] = is_outlier
    cell_profiles['lengths'] = morph_df['length'].values
    cell_profiles['cell_id'] = []
    counter = 0
    for key, df in cell_dataframe_dict.items():
        if key not in ['morphology', 'Shape_index']:

            axial_padded = []
            axial_normalized = []
            axial_straighten = []
            lateral_normalized = []
            channel_dict = {}
            for cell in cell_list:
                if counter == 0:
                    cell_profiles['cell_id'].append([cell.cluster_label, cell.cell_label])
                axial_data = cell.measurements.signal_measurements[key]['axial']
                axial_data_normalized = normalize_data1D(axial_data, re_orient=True, base=5)
                lateral_data = cell.measurements.signal_measurements[key]['lateral_mean']
                lateral_data_normalized = normalize_data1D(lateral_data, re_orient=False, base=5)
                axial_normalized.append(list(np.interp(np.linspace(0, 1, 100),
                                                       np.linspace(0, 1, len(axial_data_normalized)),
                                                       axial_data_normalized)))
                lateral_normalized.append(list(np.interp(np.linspace(0, 1, 100),
                                                         np.linspace(0, 1, len(lateral_data_normalized)),
                                                         lateral_data_normalized)))

                # pad_data, necessary for plotting demograph
                length = cell.measurements.morphology_measurements['length']
                axial_padded.append(list(pad_data(axial_data_normalized, length,
                                                  normalize=False, max_len=12)))

                # straightened data, necessary for making spherocylindrical projection plot
                straightened_data = cell.measurements.signal_measurements[key]['straighten_normalized']
                straightened_data = normalize_data_2D(straightened_data,
                                                      percentile_low_bound=0,
                                                      re_orient=True)
                axial_straighten.append(straightened_data)

            channel_dict['padded_axial_data'] = np.array(axial_padded)
            channel_dict['length_normalized_axial_data'] = np.array(axial_normalized)
            channel_dict['width_normalized_lateral_data'] = np.array(lateral_normalized)
            channel_dict['straightened_axial_data'] = axial_straighten
            cell_profiles[key] = channel_dict
            counter = 1
    if save_data:
        pk.dump(cell_profiles, open('{}{}_miscellaneous_data.pk'.format(dest_folder, header), 'wb'))
    return cell_profiles

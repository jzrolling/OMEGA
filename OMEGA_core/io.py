from .helper import *
from .metrics import *
from .image import Image, _test_stitched_data
from .config import *
import timeit
import glob, os, tifffile
import pickle as pk
from scipy import stats
import multiprocessing as mp
from matplotlib import rc
from matplotlib.gridspec import GridSpec


def plot_segmentated_cell(cell, channel=None, savefig=False,
                          filename=None, scale_bar=False, lw=1, return_fig = False,
                          d_range=[]):
    # step1, parameter initialization
    height, width = cell.mask.shape
    if min(height, width)<=2:
        fig = plt.figure(figsize=(int(width / 10), int(height / 10)))
    else:
        fig = plt.figure(figsize=(int(width / 20), int(height / 20)))
    one_micron_l = 1 / cell.pixel_microns

    # step2 determine cell orientation
    # find x,y coordinates of cell mask
    xcoor, ycoor = np.where(cell.mask > 0)
    # locate far-left, far-right indices
    ymin_id = np.argmin(ycoor)
    ymax_id = np.argmax(ycoor)
    x_left, x_right = xcoor[[ymin_id, ymax_id]]
    if x_left <= x_right:
        # left side tilting upwards
        scalebar_y = 0.1 * height
    else:
        # right side tilting upwards
        scalebar_y = 0.8 * height

    # draw scale bar
    start = width * 0.9 - one_micron_l
    end = width * 0.9
    if scale_bar:
        plt.hlines(y=scalebar_y, xmin=start, xmax=end, color='white', lw=3)

    if channel == None:
        plt.imshow(cell.data[cell.mask_channel_name], cmap='gist_gray')
    elif channel=='Shape_index':
        plt.imshow(cell.shape_index, cmap='viridis')
    else:
        if len(d_range)==0:
            plt.imshow(cell.data[channel], cmap='gist_gray')
        else:
            plt.imshow(cell.data[channel], cmap='gist_gray',
                       vmin=d_range[0], vmax=d_range[1])
    if cell.discarded or cell.branched:
        plt.plot(cell.contour.T[1], cell.contour.T[0], c='salmon', lw=lw)
    else:
        plt.plot(cell.optimized_contour.T[1], cell.optimized_contour.T[0], c='orange', lw=lw)
    if scale_bar:
        height, width = cell.mask.shape

    plt.axis('off')
    for midline in cell.midlines:
        plt.plot(midline.T[1], midline.T[0], c='royalblue',lw=lw)
    if savefig:
        fig.savefig(filename, bbox_inches='tight', dpi=160)
        plt.close(fig)
    elif return_fig:
        return fig


def plot_puncta(cell, channel, savefig=False, filename=None, lw=1):
    height, width = cell.mask.shape
    fig = plt.figure(figsize=(int(width / 20), int(height / 20)))
    if channel == cell.mask_channel_name:
        raise ValueError('Puncta function is not supported for non-fluorescent images!')
    else:
        plt.imshow(cell.data[channel], cmap='gist_gray')

    if cell.discarded or cell.branched:
        raise ValueError('Abnormal cell!')
    else:
        plt.plot(cell.optimized_contour.T[1], cell.optimized_contour.T[0], c='orange', lw=lw)
        puncta = cell.fluorescent_puncta[channel]
        if len(puncta) > 0:
            xcoor = puncta[:, 0]
            ycoor = puncta[:, 1]
            intensity = np.log2(puncta[:, 2])
            plt.scatter(ycoor, xcoor, s=intensity * 30, ec='royalblue', lw=2, fc='None')
            plt.scatter(ycoor, xcoor, s=5, color='red')
    plt.axis('off')
    if savefig:
        fig.savefig(filename, bbox_inches='tight', dpi=160)
        plt.close(fig)


def cell_plot_full(cell, savefig=False, filename=None):
    n_channels = len(cell.data)
    r = 4 * (1 + 2 * (n_channels - 1))
    c = 20
    fig = plt.figure(figsize=(c / 2, r / 2))
    gs = GridSpec(r, c)

    ref_channel = cell.mask_channel_name
    data = cell.measurements.signal_measurements
    width = cell.width_lists[0]
    length = cell.lengths[0]

    # plot ref data
    ax0 = fig.add_subplot(gs[:4, :16])
    ref_data = -data[ref_channel]['axial']
    ref_data = (ref_data - ref_data.min()) / (ref_data.max() - ref_data.min())
    ref_corr = np.linspace(0, length, len(ref_data))
    ax0.plot(ref_corr, ref_data, lw=3, ls='--', color='salmon', label='reversed phase contrast')
    ax0.set_ylim(0, 1.2)
    ax0.set_ylabel('normalized,\nreversed\nphase-contrast', fontname='Arial', fontsize=12)
    ax0.set_yticks([0.2, 0.6, 1.0])
    ax0.set_yticklabels([0.2, 0.6, 1.0], fontname='Arial', fontsize=12)
    ax0.get_xaxis().set_visible(False)
    plt.legend(loc=(0.12, 0.1), prop=rc('font', family='Arial'), fontsize=12)

    ax1 = ax0.twinx()
    ax1.plot(ref_corr, width * cell.pixel_microns, lw=3,
             color=(33 / 255, 179 / 255, 214 / 255, 1), label='cell width')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('cell width [µm]', fontname='Arial', fontsize=12, rotation=270, va='bottom')
    ax1.set_yticks([0.1, 0.5, 0.9])
    ax1.set_yticklabels([0.1, 0.5, 0.9], fontname='Arial', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    plt.legend(loc=(0.6, 0.1), prop=rc('font', family='Arial'), fontsize=12)
    # ax2 = fig.add_subplot(gs[:4,16:])
    # ax2.get_xaxis().set_visible(False)
    # ax2.get_yaxis().set_visible(False)
    counter = 0

    for channel, img in cell.data.items():
        if channel != ref_channel:
            ax_plot = fig.add_subplot(gs[4 + counter * 4:8 + counter * 4, :16])
            ax_img = fig.add_subplot(gs[8 + counter * 4:12 + counter * 4, :16])
            ax_lateral = fig.add_subplot(gs[8 + counter * 4:12 + counter * 4, 16:])

            if counter == 0:
                ax_text = fig.add_subplot(gs[4 + counter * 4:8 + counter * 4, 16:])
                ax_text.text(x=0.05, y=0.8, s='Cell_id: {}-{}'.format(cell.cluster_label, cell.cell_label),
                             fontname='Arial', fontsize=12)
                ax_text.text(x=0.05, y=0.6, s='length [µm]: {}'.format(round(length, 2)),
                             fontname='Arial', fontsize=12)
                ax_text.text(x=0.05, y=0.4, s='width [µm]: {}'.format(round(np.median(width * cell.pixel_microns), 2)),
                             fontname='Arial', fontsize=12)

            counter += 2

            ax_plot.plot(np.linspace(0, 1, len(data[channel]['axial_mean'])),
                         data[channel]['axial_mean'], lw=3, color=(33 / 255, 179 / 255, 214 / 255, 1))
            ax_plot.fill_between(x=np.linspace(0, 1, len(data[channel]['axial_mean'])),
                                 y1=data[channel]['axial_mean'] - data[channel]['axial_std'],
                                 y2=data[channel]['axial_mean'] + data[channel]['axial_std'],
                                 color=(33 / 255, 179 / 255, 214 / 255, 1), alpha=0.2)

            ax_plot.set_ylabel('{} intensity'.format(channel), fontname='Arial', fontsize=12)
            ax_plot.set_yticklabels(ax_plot.get_yticks(), fontname='Arial', fontsize=12)
            ax_plot.get_xaxis().set_visible(False)

            ax_img.imshow(data[channel]['straighten_normalized'], aspect='auto')
            ax_img.get_xticks()
            xticks = (ax_img.get_xticks())
            newxticks = [np.round(length * (tick / data[channel]['straighten_normalized'].shape[1]), 1) for tick in
                         xticks]
            ax_img.set_xticklabels(newxticks, fontname='Arial', fontsize=12)
            ax_img.set_yticks([])
            ax_img.set_ylabel('{}\nstraighten image\n'.format(channel), fontname='Arial', fontsize=12)
            ax_img.set_xlabel('cell length [µm]'.format(channel), fontname='Arial', fontsize=12)

            ax_text.get_xaxis().set_visible(False)
            ax_text.get_yaxis().set_visible(False)

            lateral_mean = data[channel]['lateral_mean']
            lateral_std = data[channel]['lateral_std']

            ax_lateral.plot(lateral_mean, np.linspace(0, 1, len(lateral_mean)), lw=3,
                            color=(33 / 255, 179 / 255, 214 / 255, 1))
            ax_lateral.fill_betweenx(y=np.linspace(0, 1, len(lateral_mean)), color=(33 / 255, 179 / 255, 214 / 255, 1),
                                     x1=lateral_mean - lateral_std, x2=lateral_mean + lateral_std, alpha=0.2)
            ax_lateral.get_yaxis().set_visible(False)

            lateral_ticks = ax_lateral.get_xticks()
            lateral_ticks = [lateral_ticks[0], lateral_ticks[-1]]
            ax_lateral.set_xticks(lateral_ticks)
            ax_lateral.set_xticklabels(lateral_ticks, fontname='Arial', fontsize=12)
            ax_lateral.set_xlabel('{} lateral axis'.format(channel), fontname='Arial', fontsize=12)

    if savefig:
        fig.savefig(filename, bbox_inches='tight', dpi=160)
        plt.close(fig)



def plot_segmented_cluster(image, cluster_key, channel='default',
                           savefig=False, filename=None,
                           c_discarded='black', c_branched='crimson',
                           c_accepted='orange', c_midline='royalblue',
                           line_style='-', lw=1, scale_bar=False, scale_bar_length=10):

    fig = plt.figure(figsize=(10,10))
    cluster = image.clusters[cluster_key]
    xc1, yc1, xc2, yc2 = cluster.bbox

    if channel == 'default':
        canvas = image.data[image.mask_channel_name][xc1:xc2, yc1:yc2]
    elif channel in image.data:
        canvas = image.data[channel][xc1:xc2, yc1:yc2]
    else:
        raise ValueError('No {} channel found!'.format(channel))

    plt.imshow(canvas, cmap='gist_gray')
    for cell_id, cell in cluster.cells.items():
        x1, y1, x2, y2 = cell.bbox
        x1 -= xc1
        x2 -= xc1

        y1 -= yc1
        y2 -= yc1
        if cell.discarded:
            contour = cell.contour
            plt.plot(contour[:, 1] + y1, contour[:, 0] + x1, lw=lw, c=c_discarded, ls=line_style)
        elif cell.branched:
            contour = cell.contour
            plt.plot(contour[:, 1] + y1, contour[:, 0] + x1, lw=lw, c=c_branched, ls=line_style)
        else:
            contour = cell.optimized_contour
            midline = cell.midlines[0]
            plt.plot(contour[:, 1] + y1, contour[:, 0] + x1, lw=lw, c=c_accepted, ls=line_style)
            plt.plot(midline[:, 1] + y1, midline[:, 0] + x1, lw=lw, c=c_midline, ls=line_style)
    plt.axis('off')

    # plot scale bar:
    if scale_bar:
        height, width = cluster.mask.shape
        micron_l = scale_bar_length / image.pixel_microns
        scalebar_y = 0.9 * height
        start = width * 0.9 - micron_l
        end = width * 0.9
        if scale_bar:
            plt.hlines(y=scalebar_y, xmin=start, xmax=end, color='white', lw=6)

    if savefig:
        plt.savefig(filename, bbox_inches='tight', dpi=240)
        plt.close(fig)


def plot_segmented_image(image, channel='default',
                         savefig = False, filename = None,
                         c_discarded='black', c_branched='crimson',
                         c_accepted='orange', c_midline='royalblue',
                         lw=0.5):
    fig = plt.figure(figsize=(10,10))
    if channel == 'default':
        canvas = image.data[image.mask_channel_name]
    elif channel in image.data:
        canvas = image.data[channel]
    else:
        raise ValueError('No {} channel found!'.format(channel))
    plt.imshow(canvas, cmap='gist_gray')
    for cluster_id, cluster in image.clusters.items():
        if cluster.cells != {}:
            for cell_id, cell in cluster.cells.items():
                if cell.discarded:
                    contour = cell.contour
                    x1, y1, x2, y2 = cell.bbox
                    plt.plot(contour[:, 1]+y1, contour[:, 0]+x1, lw=lw, c=c_discarded)
                elif cell.branched:
                    contour = cell.contour
                    x1, y1, x2, y2 = cell.bbox
                    plt.plot(contour[:, 1] + y1, contour[:, 0] + x1, lw=lw, c=c_branched)
                else:
                    contour = cell.optimized_contour
                    midline = cell.midlines[0]
                    x1, y1, x2, y2 = cell.bbox
                    plt.plot(contour[:, 1] + y1, contour[:, 0] + x1, lw=lw, c=c_accepted)
                    plt.plot(midline[:, 1] + y1, midline[:, 0] + x1, lw=lw, c=c_midline)
    plt.axis('off')
    if savefig:
        fig.savefig(filename, bbox_inches='tight', dpi=240)
        plt.close(fig)


def plot_segmented_binary(image, savefig=False, filename=None):

    canvas = np.zeros(image.mask_binary.shape)
    for cluster_id, cluster in image.clusters.items():
        if cluster.cells != {}:
            for cell_id, cell in cluster.cells.items():
                if not cell.discarded and not cell.branched:
                    x1, y1, x2, y2 = cell.bbox
                    canvas[x1:x2, y1:y2][cell.mask>0] = 255
    canvas = canvas.astype(np.uint8)
    if savefig:
        tifffile.imsave(filename, canvas, imagej=True)
    else:
        tifffile.imshow(canvas)


def auto_process_no_return(imagefile,
                           header,
                           output_folder,
                           configfile = 'default',
                           print_time = False,
                           save_fig=False,
                           save_segmentation_result=True,
                           save_cells=True,
                           save_discarded=False,
                           return_obj = False):
    cell_list = []
    t1 = timeit.default_timer()
    if configfile == 'default':
        config = Preset_configurations_default().config
    elif '.ini' in configfile:
        try:
            config = Preset_configurations_default(configfile=configfile).config
        except:
            print('{} config file not supported, proceed with default settings.'.format(configfile))
            config = Preset_configurations_default().config
    obj = Image()
    obj.config = config
    if imagefile.endswith('.nd2'):
        obj.read_nd2_file(imagefile)
    elif imagefile.endswith('.tif'):
        obj.read_tiff_file(imagefile)
    else:
        raise ValueError('.{} file not supported!'.format(imagefile.split('.')[-1]))
    obj.crop_edge()
    obj.enhance_brightfield()
    obj.enhance_fluorescence()
    obj.locate_clusters()
    obj.cluster_segmentation()
    obj.cell_segmentation()
    for _key, cluster in obj.clusters.items():
        for _label, cell in cluster.cells.items():
            cell.image_label = header
            if not save_discarded:
                if not cell.discarded:
                    cell_list.append(cell)
            else:
                cell_list.append(cell)
    t2 = timeit.default_timer()
    if print_time:
        print("Analysis complete, {} seconds used.".format(round(t2 - t1), 2))
    if save_fig:
        save_processed_images(obj,header,output_folder)
    if save_segmentation_result:
        plot_segmented_image(obj, savefig=True,
                             filename="{}{}_segmentation_result.tif".format(output_folder, header))
    if save_cells:
        pickle_dump(cell_list, header+'_cells', output_folder)
    if return_obj:
        return obj


def save_processed_images(obj, header, output_folder, plot_segmentation_result = True):
    for channel, img in obj.data.items():
        channel_name = channel.replace(' ', '_')
        tifffile.imsave("{}{}_{}_filtered.tif".format(output_folder, header, channel_name),
                        img.astype(np.uint16), imagej=True)
    if plot_segmentation_result:
        plot_segmented_image(obj, savefig=True,
                             filename="{}{}_segmentation_result.tif".format(output_folder, header))


def process_folder(input_folder,
                   output_folder=None,
                   file_type='.nd2',
                   configfile='default',
                   save_img=False,
                   save_segmentation_result=True,
                   n_cores=2):
    if output_folder == None:
        output_folder = input_folder + 'output/'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    input_files = sorted(glob.glob(input_folder+'*'+file_type))
    if len(input_files) == 0:
        raise ValueError('No {} file(s) found in {}!'.format(file_type, input_folder))

    for i in range(0,len(input_files),n_cores):
        processes = []
        for j in range(0,min(n_cores,len(input_files)-i)):
            file = input_files[i+j]
            header = file.split('/')[-1].split('.')[0]
            processes.append(mp.Process(target=auto_process_no_return, args=(file, header, output_folder,
                                                                             configfile, False,
                                                                             save_img, save_segmentation_result,
                                                                             True, False,False)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()


def folder_measurements(output_folder, save_all_cells = False, save_accepted_cells=True):
    pk_files = sorted(glob.glob(output_folder + '*_cells.pk'))
    allcells = []
    if len(pk_files) != 0:
        for pks in pk_files:
            allcells += pk.load(open(pks, 'rb'))
    discarded_cells, \
    branched_cells, \
    accepted_cells, \
    accepted_cell_dfs, \
    outlier_id = measurements_to_dataframe(allcells, 'all_cells', output_folder)
    _miscellaneous = _export_data(accepted_cells, accepted_cell_dfs, outlier_id, 'all_cells', output_folder)
    if save_all_cells:
        pickle_dump(allcells, 'all_cells', output_folder)
    if save_accepted_cells:
        pickle_dump(accepted_cells, 'accepted_cells', output_folder)


def _export_data(accepted_cell_list, accepted_cell_dfs, outlier_id, header, output_folder,
                 correct_drift = True):

    everybit = {}
    if 'morphology' not in accepted_cell_dfs:
        print(accepted_cell_dfs.keys())
    morph_df = accepted_cell_dfs['morphology']
    sort_by_length_filtered = np.array(morph_df.iloc[np.where(outlier_id==1)].sort_values(by='length').index)
    everybit['sorted_filtered_id'] = sort_by_length_filtered
    for key, df in accepted_cell_dfs.items():
        if key not in ['morphology', 'Shape_index']:
            axial_padded = []
            axial_normalized = []
            lateral_normalized = []
            channel_dict = {}
            for cell in accepted_cell_list:
                if (key != cell.mask_channel_name) and (key in cell.data):

                    abs_skewness = abs(cell.measurements.signal_measurements[key]['cell_skewness'])
                    lateral_center_offset = cell.measurements.signal_measurements[key]['lateral_center_offset']
                    mean_intensity = cell.measurements.signal_measurements[key]['mean']
                    fg_intensity = cell.image_stat[key][1]
                    if correct_drift:
                        if (lateral_center_offset == 1) and (abs_skewness <= 2):
                            if (mean_intensity/fg_intensity) >= 3:
                                cell._correct_xy_drift()
                                cell._update_signal(key)

                axial_data = cell.measurements.signal_measurements[key]['axial']
                axial_data_normalized = normalize_data1D(axial_data, re_orient=True)
                lateral_data = cell.measurements.signal_measurements[key]['lateral_mean']
                lateral_data_normalized = normalize_data1D(lateral_data, re_orient=False)
                length = cell.measurements.morphology_measurements['length']
                axial_padded.append(list(pad_data(axial_data_normalized, length, normalize=False, max_len=12)))
                axial_normalized.append(list(np.interp(np.linspace(0, 1, 100),
                                                  np.linspace(0, 1, len(axial_data_normalized)),
                                                  axial_data_normalized)))
                lateral_normalized.append(list(np.interp(np.linspace(0, 1, 100),
                                                    np.linspace(0, 1, len(lateral_data_normalized)),
                                                    lateral_data_normalized)))

            channel_dict['padded_axial_data'] = np.array(axial_padded)
            channel_dict['length_normalized_axial_data'] = np.array(axial_normalized)
            channel_dict['width_normalized_lateral_data'] = np.array(lateral_normalized)
            everybit[key] = channel_dict
    pk.dump(everybit, open('{}{}_miscellaneous_data.pk'.format(output_folder, header), 'wb'))
    return everybit


def pickle_dump(data, header, output_folder):
    pk.dump(data, open("{}{}.pk".format(output_folder, header), 'wb'))


def pickle_load_all_cells(folder):
    pks = glob.glob(folder+'*cells.pk')
    cells = []
    for file in pks:
        cells += pk.load(open(file, 'rb'))
    return cells


def measurements_to_dataframe(cells, header, output_folder):
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


    if discarded_cells != []:
        discarded_df_dict = _to_dataframe(discarded_cells, by='particle_morphology')
        for key, df in discarded_df_dict.items():
            df.to_excel("{}{}_{}_{}.xls".format(output_folder, header, 'discarded_cells', key))

    if branched_cells != []:
        branched_df_dict = _to_dataframe(branched_cells, by='particle_morphology')
        for key, df in branched_df_dict.items():
            df.to_excel("{}{}_{}_{}.xls".format(output_folder, header, 'branched_cells', key))

    if accepted_cells != []:
        accepted_df_dict = _to_dataframe(accepted_cells, by='all')
        for key, df in accepted_df_dict.items():
            if key == 'morphology':
                filtered_df, outliers = filter_outliers(df, by=['length', 'width_median', 'width_std', 'circularity'])
            else:
                filtered_df, outliers = filter_outliers(df, by=['median'])
            filtered_df.to_excel("{}{}_{}_{}.xls".format(output_folder, header, 'accepted_cells', key))
            filtered_cell_dfs[key] = filtered_df
            outlier_id.append(outliers)

    outlier_id = (np.sum(np.array(outlier_id), axis=0)==0)*1
    return discarded_cells, branched_cells, accepted_cells, filtered_cell_dfs, outlier_id


def _to_dataframe(cells, by='all'):
    all_keys = {}
    all_measurements = {}
    dataframe_dict = {}
    for cell in cells:
        image_label = cell.image_label
        cluster_label = cell.cluster_label
        cell_label = cell.cell_label
        key_dict, measurement_dict = cell.measurements.data_to_list(by = by)
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


def filter_outliers(df, z_score_th = 3, IQR_th=2,
                    by=['length', 'width_median', 'width_std', 'circularity']):
    in_zscore_range = np.sum(stats.zscore(df[by], axis=0)>=z_score_th, axis=1)>=1
    q1,q3 = df[by].quantile(0.25), df[by].quantile(0.75)
    iqr = q3-q1
    in_IQR_range = np.sum(np.array((df[by]<q1-IQR_th*iqr)|(df[by]>q3+IQR_th*iqr)),
                          axis=1) >= 1
    df['is_outlier'] = ((in_IQR_range+in_zscore_range)>0)*1
    return df, list(df['is_outlier'])


def demograph(data_dict_file, channel,
              max_length=8, pixel_width=512,
              filename = None, width=3, height=4,
              save_img=False, dpi=240):
    data_dict = pk.load(open(data_dict_file,'rb'))
    cell_id = data_dict['sorted_filtered_id']
    data = data_dict[channel]['padded_axial_data'][cell_id]

    fig = plt.figure(figsize=(width,height))
    ax = plt.subplot(111)
    ax.imshow(data,cmap='viridis',aspect='auto')
    pos = [int(pixel_width*0.25)*x for x in range(5)]
    label = [int(max_length*0.25)*x for x in range(5)]
    plt.xticks(pos,label)
    plt.xlabel('length [um]')
    ax.yaxis.set_visible(False)
    if save_img:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close


def create_canvas(width=101, height=600):
    canvas = np.zeros((width, height))
    r = int(width / 2)
    rr1, cc1 = draw.ellipse(r, int(r * 1.6), r, int((r) * 1.6))
    rr2, cc2 = draw.ellipse(r, int(height - 1.6 * r), r, int((r) * 1.6))
    rr3, cc3 = draw.rectangle(start=(1, int(r * 1.6)), end=(width - 2, height - int(r * 1.6)),
                                        shape=canvas.shape)
    canvas[rr3, cc3] = 1
    canvas[rr1, cc1] = 1
    canvas[rr2, cc2] = 1
    l = len(np.nonzero(np.sum(canvas, axis=0))[0])
    counter = 0
    canvas = canvas.T
    return canvas


def create_mesh(canvas):
    xt, yt = np.nonzero(canvas)
    l, m = np.sum(canvas, axis=0), np.sum(canvas, axis=1)
    norm_yt = np.zeros(xt.shape)
    norm_xt = (xt - xt.min()) / (l.max() - 1)
    count = 0
    for i in range(len(xt)):
        r = m.max() / m[xt[i]]
        norm_yt[i] = count * r / (m.max() - 1)
        if i != len(xt) - 1:
            if xt[i + 1] == xt[i]:
                count += 1
            else:
                count = 0
        else:
            count += 1
    return (xt, yt, norm_xt, norm_yt)


def project_image(xt, yt, norm_xt, norm_yt, canvas, data):
    paint = np.zeros(canvas.shape)
    xid = norm_xt.copy()
    yid = norm_yt.copy()
    xid *= data.shape[0] - 1
    yid *= data.shape[1] - 1
    interpolated = bilinear_interpolate_numpy(data, xid, yid)
    paint[xt, yt] = interpolated
    return (paint)


def group_by_length(length, groups=[3, 4, 5, 6]):
    if length < groups[0]:
        return 1
    elif length < groups[1]:
        return 2
    elif length < groups[2]:
        return 3
    elif length < groups[3]:
        return 4
    else:
        return 5


def initiate_projection():
    width = 75
    heights = [250, 350, 450, 550, 650]
    pad = np.zeros((50, width))
    gap = np.zeros((750, 10))
    padded = [gap]
    xt_list, yt_list, nxt_list, nyt_list = [], [], [], []
    for i in range(len(heights)):
        a = create_canvas(width=width, height=heights[i])
        half_pad = np.tile(pad, (len(heights) - i, 1))
        m_pad = np.concatenate([half_pad, a, half_pad], axis=0)
        m_pad = np.concatenate([m_pad, gap], axis=1)
        xt, yt, norm_xt, norm_yt = create_mesh(m_pad)
        xt_list.append(xt)
        yt_list.append(yt)
        nxt_list.append(norm_xt)
        nyt_list.append(norm_yt)
        padded.append(m_pad)
    contours = measure.find_contours(np.concatenate(padded, axis=1), level=0)
    optimized_outline = []
    for contour in contours:
        optimized_outline.append(spline_approximation(contour, n=2 * len(contour)))
    return padded, xt_list, yt_list, nxt_list, nyt_list, optimized_outline


def divide_by_fraction(data_length, by=5):
    out = []
    for i in range(by - 1):
        out.append(i * int(data_length / by))
    out.append(data_length)
    return out


def normalize_data_2D(data, percentile=1, re_align=True):
    half_l = int(0.5 * data.shape[1])
    if data[:, :half_l].mean() < data[:, -half_l:].mean():
        if re_align:
            data = np.flip(data, axis=1)

    th_low = np.percentile(data, percentile)

    normalized_data = (data - th_low) / (data.max() - th_low)
    normalized_data[normalized_data > 1] = 1
    normalized_data[normalized_data < 0] = 0
    return normalized_data


def project_by_percentile(length_sorted_index, cells, channel,
                          canvas=None, re_align=True, min_cells=10,
                          savefig=False, filename=None, hide_title=True):

    #
    if canvas == None:
        padded, xt_list, yt_list, nxt_list, nyt_list, optimized_outline = initiate_projection()
    else:
        padded, xt_list, yt_list, nxt_list, nyt_list, optimized_outline = canvas

    well_paints = padded.copy()
    for m in well_paints:
        m *= 0
    group_count = [0, 0, 0, 0, 0]

    grouped_index_fraction = divide_by_fraction(len(length_sorted_index), by=6)
    for i in range(5):
        p1 = grouped_index_fraction[i]
        p2 = grouped_index_fraction[i + 1]
        index_list = length_sorted_index[p1:p2]

        for idx in index_list:
            cell = cells[idx]

            group_count[i] += 1
            group = i + 1
            canvas = padded[group].copy()

            img = cell.data[channel]
            midline = cell.midlines[0]
            width = cell.width_lists[0]
            data = straighten_cell_normalize_width(img, midline, width)

            normalized_data = normalize_data_2D(data, re_align=re_align)

            well_paints[group] += project_image(xt_list[group - 1], \
                                                yt_list[group - 1], \
                                                nxt_list[group - 1], \
                                                nyt_list[group - 1], \
                                                canvas, normalized_data.T)

    for j in range(len(well_paints)):
        graph = well_paints[j]
        if graph.max() > 0:
            min_val = graph[graph > 0].min()
            normalized = (graph - min_val) / (graph.max() - min_val)
            normalized[graph == 0] = 0
            if (j > 0) and (group_count[j - 1] < min_cells):
                normalized *= 0
            well_paints[j] = normalized

    painttttt = np.concatenate(well_paints, axis=1)
    fig = plt.figure(figsize=(4.5, 7.5))
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.set_xticks([])
    ax.get_yaxis().set_visible(False)
    plt.imshow(painttttt, cmap="viridis", aspect='auto')
    if not hide_title:
        plt.title('spherocylindrical\nprojection', fontname='Arial', fontsize=24)
    for outline in optimized_outline:
        ax.plot(outline.T[1], outline.T[0], c="salmon", alpha=0.5)
    if savefig:
        fig.savefig(filename, bbox_inches='tight', dpi=160, transparent=True)
        plt.close(fig)


def _plate_triplot(folder, channel, canvas, z_cutoff=4, intensity_cutoff=50):
    """
    :param folder:
    :return:
    """

    # import data
    if not os.path.isfile(folder + 'accepted_cells.pk'):
        raise ValueError('Files not found, is {} an empty folder?'.format(folder))
    cells = pk.load(open(folder + 'accepted_cells.pk', 'rb'))
    data_dict = pk.load(open(folder + 'all_cells_miscellaneous_data.pk', 'rb'))
    morph_df = pd.read_excel(folder + 'all_cells_accepted_cells_morphology.xls')
    filtered_morph_df = morph_df[(morph_df['Quality_by_shape_index'] == 1) & (morph_df['is_outlier'] == 0)].copy()
    length_sorted_index = filtered_morph_df.sort_values(by='length').index

    # fig1: demograph
    demograph = data_dict[channel]['padded_axial_data'][length_sorted_index]
    # enhance contrast
    demograph[demograph > 0] = demograph[demograph > 0] + 0.1

    fig1 = plt.figure(figsize=(7.3, 4.5))
    ax = plt.subplot(111)
    ax.imshow(demograph.T, aspect='auto')
    ax.get_xaxis().set_visible(False)
    ticks = [0, 127, 255, 383, 511]
    tick_labels = [6, 3, 0, 3, 6]
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontname='Arial', fontsize=16)
    ax.set_ylabel('distance from center [µm]', fontname='Arial', fontsize=16)
    #plt.title('normalized demograph', fontname='Arial', fontsize=16)
    fig1.savefig(folder + 'demgraph.png', bbox_inches='tight', dpi=160, transparent=True)
    plt.close()

    # fig2: spherocylindrical projection
    sc_projection = folder + 'spherocylindrical_projection.png'
    project_by_percentile(length_sorted_index, cells, channel, canvas, savefig=True, filename=sc_projection)

    # puncta density plot:

    padded, xt_list, yt_list, nxt_list, nyt_list, optimized_outline = canvas
    coords = []
    for outline in optimized_outline:
        x0 = outline.T[1].mean()
        y0 = outline.T[0].mean()
        l = outline.T[0].max() - outline.T[0].min()
        w = outline.T[1].max() - outline.T[1].min()
        coords.append([x0, y0, l, w])

    fig3 = plt.figure(figsize=(4.5, 7.5))
    ax = fig3.add_subplot(111)
    # ax.get_xaxis().set_visible(False)
    ax.set_xticks([])
    # ax.set_xlabel('cell',fontname='Arial', fontsize=16)
    ax.get_yaxis().set_visible(False)
    # plt.imshow(painttttt, cmap="viridis")
    bg = np.concatenate(padded, axis=1)
    bg *= 0
    ax.imshow(bg, cmap='Greys', aspect='auto')
    plt.xlim(0, bg.shape[1]-1)

    for outline in optimized_outline:
        ax.plot(outline.T[1], outline.T[0], c="salmon", alpha=1)

    grouped_index_fraction = divide_by_fraction(len(length_sorted_index), by=6)
    grouped_data = [[], [], [], [], []]

    for i in range(5):
        p1 = grouped_index_fraction[i]
        p2 = grouped_index_fraction[i + 1]
        index_list = length_sorted_index[p1:p2]
        x0, y0, l, w = coords[4-i]

        for idx in index_list:
            reorient = False
            cell = cells[idx]
            puncta = cell.fluorescent_puncta[channel]
            cell_data = cell.data[channel][cell.mask > 0]
            axial_data=cell.measurements.signal_measurements[channel]['axial']
            half_l = int(0.5*len(axial_data))
            if axial_data[:half_l].mean() < axial_data[half_l:].mean():
                reorient = True

            if len(puncta) > 0:
                puncta_intensity = puncta[:, 2]
                normalized_puncta = puncta_projection(cell, channel)[:, -3:]
                ycoords = (normalized_puncta[:, 0] + normalized_puncta[:, 1]) - 0.5
                if reorient:
                    ycoords = -ycoords
                ycoords = ycoords * l + y0

                xcoords = (normalized_puncta[:, 2]) * w + x0

                z_scores = (puncta_intensity - cell_data.mean()) / cell_data.std()

                puncta_intensity = np.reshape(puncta_intensity, -1, 1)
                ycoords = np.reshape(ycoords, -1, 1)
                xcoords = np.reshape(xcoords, -1, 1)

                compiled_data = np.array([xcoords, ycoords, puncta_intensity, z_scores]).T
                compiled_data = compiled_data[np.where(np.abs(normalized_puncta[:, 2]<=0.51))]
                if len(compiled_data) > 0:
                    grouped_data[i].append(compiled_data)

        grouped_data[i] = np.concatenate(grouped_data[i], axis=0)
        accepted = np.where((grouped_data[i][:, 0] > 0) & (grouped_data[i][:, 2] > intensity_cutoff)
                            & (grouped_data[i][:, 3] > z_cutoff))
        grouped_data[i] = grouped_data[i][accepted]


        xy = grouped_data[i][:, :2]
        """
        dist_matrix = np.sqrt(np.sum((xy[:, np.newaxis] - xy) ** 2, axis=2))
        dist_matrix[dist_matrix == 0] = np.inf
        dist_matrix = 1 / dist_matrix
        weighted_dist = np.sum(dist_matrix, axis=1)
        weighted_dist /= weighted_dist.max()
        # weighted_dist = (weighted_dist-weighted_dist.min())/(weighted_dist.max()-weighted_dist.min())
        colors = cm.get_cmap('viridis')(weighted_dist)
        """


        #intensities = grouped_data[i][:, 2] / grouped_data[i][:, 2].max()

    merged_data=np.concatenate(grouped_data, axis=0)
    if len(merged_data) > 0:
        z = merged_data[:, 3]
        z = (z-z.min())/(z.max()-z.min())
        colors = cm.get_cmap('viridis')(z)
        ax.scatter(merged_data[:, 0], merged_data[:, 1], color=colors, alpha=0.2, s=80)

    #plt.title('fluorescent puncta\n density plot', fontname='Arial', fontsize=24)
    fig3.savefig(folder + 'puncta_density_plot', bbox_inches='tight',
                 dpi=160, transparent=True)
    plt.close()


def _auto_process_no_return_morpheus(obj,
                                     header,
                                     output_folder,
                                     configfile='OMEGA/configuration_MORPHEUS.ini',
                                     print_time=False,
                                     save_fig=False,
                                     save_segmentation_result=True,
                                     save_cells=True,
                                     save_discarded=False,
                                     return_obj = False):
    cell_list = []
    t1 = timeit.default_timer()
    if configfile == 'default':
        config = Preset_configurations_default().config
    elif '.ini' in configfile:
        try:
            config = Preset_configurations_default(configfile=configfile).config
        except:
            print('{} config file not supported, proceed with default settings.'.format(configfile))
            config = Preset_configurations_default().config

    obj.config = config
    obj.crop_edge()
    obj.enhance_brightfield()
    obj.enhance_fluorescence()
    obj.locate_clusters()
    obj.cluster_segmentation()
    obj.cell_segmentation(shapeindex_quality=False,
                          filter_by_shapeindex_quality=False)
    for _key, cluster in obj.clusters.items():
        for _label, cell in cluster.cells.items():
            cell.image_label = header
            if not save_discarded:
                if not cell.discarded:
                    cell_list.append(cell)
            else:
                cell_list.append(cell)
    t2 = timeit.default_timer()
    if print_time:
        print("Analysis complete, {} seconds used.".format(round(t2 - t1), 2))
    if save_fig:
        save_processed_images(obj, header, output_folder)
    if save_segmentation_result:
        plot_segmented_image(obj, savefig=True,
                             filename="{}{}_segmentation_result.tif".format(output_folder, header))
    if save_cells:
        pickle_dump(cell_list, header+'_cells', output_folder)
    if return_obj:
        return obj


def _OMEGA2Morpheus(filename, output_folder,
                    mask_channel_id = 0,
                    n_cores=4,
                    configfile='OMEGA/configuration_MORPHEUS.ini',
                    save_img=True, save_segmentation_result=True):

    header = str(filename.split('/')[-1].split('.')[0])

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    morpheus = _test_stitched_data(filename, mask_channel_id=mask_channel_id)
    n=0
    for i in range(0, len(morpheus), n_cores):
        processes = []
        for j in range(0,min(n_cores,len(morpheus)-i)):
            obj = morpheus[i+j]
            subheader = '{}_{}'.format(header, n)
            processes.append(mp.Process(target=_auto_process_no_return_morpheus, args=(obj, subheader, output_folder,
                                                                                       configfile, False,
                                                                                       save_img,
                                                                                       save_segmentation_result,
                                                                                       True, False, False)))
            n+=1
        for p in processes:
            p.start()
        for p in processes:
            p.join()

from .helper import *
from scipy.stats import kurtosis, skew, linregress
from scipy import signal


def center_of_mass(data, xcor):
    center = np.sum((data) * xcor) / np.sum((data))
    return np.round(center, 3)


def normalize_array(data, min_val, max_val, base=0):
    return (data-min_val+base)/(max_val-min_val+base)


def weighted_segmentation(data, quantile_filter=0.05, weighted_center=True):
    if quantile_filter:
        min_threshold = np.quantile(data, quantile_filter)
    else:
        min_threshold = 0
    filtered_data = data.copy()
    normalized_data = normalize_array(filtered_data, min_threshold, filtered_data.max())
    normalized_data[filtered_data <= min_threshold] = 0.0
    xcor = np.linspace(0, 1, len(normalized_data))
    if weighted_center:
        com_0 = center_of_mass(normalized_data, xcor)
    else:
        com_0 = 0.5
    adjusted_center_dist_1 = int(com_0*len(normalized_data))
    adjusted_center_dist_2 = len(normalized_data) - adjusted_center_dist_1
    first_half = normalized_data[:adjusted_center_dist_1]
    xcor_first = xcor[:adjusted_center_dist_1]
    second_half = normalized_data[-adjusted_center_dist_2:]
    xcor_second = xcor[-adjusted_center_dist_2:]
    com_1 = center_of_mass(first_half, xcor_first)
    com_2 = center_of_mass(second_half, xcor_second)
    com_dist = com_2-com_1
    return [com_1, com_0, com_2], round(com_dist, 3)


def segmented_mean(data, quantile_filter=0.05, weighted_center=True):
    interpolated = np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, len(data)), data)
    smoothed = moving_window_average(interpolated)
    [c1, c2, c3], c_dist = weighted_segmentation(smoothed,\
                                                 quantile_filter=quantile_filter,\
                                                 weighted_center=weighted_center)
    n_points = len(smoothed)
    c1, c2, c3 = int(c1*n_points), int(c2*n_points), int(c3*n_points)
    l1, l2, r1, r2 = smoothed[:c1].mean(), smoothed[c1:c2].mean(), smoothed[c2:c3].mean(), smoothed[c3:].mean()
    h1, h2 = smoothed[:c2].mean(), smoothed[c2:].mean()
    pole, center = np.concatenate([smoothed[:c1], smoothed[c3:]]).mean(), smoothed[c1:c3].mean()
    return l1, l2, r1, r2, h1, h2, pole, center


def sigmoid(x):
    return 1/(1+np.exp(-x))


def measure_symmetry(data, weighted):
    if weighted:
        weighted_center = center_of_mass(data, np.linspace(0, 1, len(data)))
    else:
        weighted_center = 0.5
    avg_1 = data[:int(round(weighted_center * len(data)))].sum()
    avg_2 = data[int(round(weighted_center * len(data))):].sum()
    ratio = max(avg_1, avg_2)/min(avg_1, avg_2)
    return 2-2*sigmoid(ratio-1)

"""
Deprecated
def measure_centrifugality(pole, center):
    return round(np.log2(pole/center), 3)


def measure_membrane_idx(data, quantile_filter=0.2, weighted_center=True, window=2):
    interpolated = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(data)), data)
    smoothed = helper.moving_window_average(interpolated)
    [c1, c2, c3], c_dist = weighted_segmentation(smoothed,\
                                                 quantile_filter=quantile_filter,\
                                                 weighted_center=weighted_center)
    n_points = len(smoothed)
    c1, c2, c3 = int(c1*n_points), int(c2*n_points), int(c3*n_points)
    v1 = smoothed[max(0, c1-window):min(len(smoothed), c1+window)].mean()
    v2 = smoothed[max(0, c2-window):min(len(smoothed), c2+window)].mean()
    v3 = smoothed[max(0, c3-window):min(len(smoothed), c3+window)].mean()
    ratio = round(np.average([v1, v3])/v2, 3)
    return ratio, c_dist
"""

def divide_cell(l, tip=0.25):
    left_tip_fraction = tip/l
    left_subpolar_fraction = tip/l+min(0.1, 1.5/l)
    left_periseptum_fraction = 0.4
    septum_fraction = 0.6
    right_perimseptum_fraction = 1-tip/l-min(0.1, 1.5/l)
    right_subpolar_fraction = 1-tip/l
    right_tip_fraction = 1
    return [left_tip_fraction, left_subpolar_fraction,\
            left_periseptum_fraction, septum_fraction,\
            right_perimseptum_fraction,
            right_subpolar_fraction, right_tip_fraction]


def divede_cell_pos(data, l, normalize=True):
    if normalize:
        data = normalize_data1D(data, re_orient=True)
    fractions = divide_cell(l)
    n_points = len(data)
    coords = np.round(np.array(fractions)*n_points).astype(int)
    output_data = []
    last_pos = 0
    for pos in coords:
        output_data.append(np.mean(data[last_pos:pos]))
        last_pos = pos
    return output_data


def _divede_cell_pos_raw(data, l):
    half_l = int(0.5 * len(data))
    if data[:half_l].mean() < data[-half_l:].mean():
        data = np.flip(data)
    fractions = divide_cell(l)
    n_points = len(data)
    coords = np.round(np.array(fractions)*n_points).astype(int)
    output_data = []
    last_pos = 0
    for pos in coords:
        output_data.append(np.mean(data[last_pos:pos]))
        last_pos = pos
    return output_data


def kurtosis_skewness(data):
    return kurtosis(data), skew(data)


def scaled_linregress(data_1D):
    data_1D = signal.medfilt(data_1D,kernel_size=5)
    slope, intercept, r_value, p_value, std_err = linregress(np.linspace(0,1,len(data_1D)),data_1D/data_1D.max())
    return slope, intercept, r_value, p_value, std_err


def FWHM_background_aware(data, percentile = False):
    confidence_score = 2
    FWHM = 0
    L = len(data)
    if percentile:
        max_val = np.percentile(data, 95)
        min_val = np.percentile(data, 5)
    else:
        max_val = data.max()
        min_val = data.min()
    HM = min_val + 0.5*(max_val-min_val)
    v1 = data[:-1]
    v2 = data[1:]
    p1 = v1-HM
    p2 = v2-HM
    p3 = p1*p2
    intersections = np.where(p3<0)[0]
    intersection_xcoords = []
    for i in intersections:
        intersection_xcoords.append(linear_intersection(i,v1[i],i+1,v2[i],HM))
    if len(intersection_xcoords) == 1:
        idx = np.where(data == data.max())[0]
        if min(L-idx,idx) >= 5:
            FWHM = abs(intersection_xcoords[0]-int(idx))*2
            confidence_score = 1
    if len(intersections) == 2:
        FWHM = intersections[1]-intersections[0]
        confidence_score = 0
    elif len(intersections) >= 3:
        FWHM = intersections[-1]-intersections[0]
        confidence_score = 1
    return FWHM, confidence_score

def linear_intersection(x0,y0,x1,y1,y):
    if y1 == y0:
        return 0.5*(x1+x0)
    else:
        return round(((x1-x0)*(y-y1)/(y1-y0))+x1,3)

# morphological metrics
def sinuosity(midline):
    if (midline[0]-midline[-1]).sum() == 0:
        raise ValueError('Midline coordinates appear to be closed!')
    end_to_end_dist = distance(midline[0],midline[-1])
    length = line_length(midline)
    ret = round(length/end_to_end_dist, 3)
    if ret < 1:
        ret = 1.0
    return ret


def average_bending_energy(contour, window=3):
    angles = bend_angle(contour[:-1], window=window)*np.pi/180
    ks = (np.sin(angles)**2).sum()/len(contour)
    return round(ks, 4)


def bending_energy(data, window=3):
    """
    Bending energy func by Bowie and Young, 1977
    BE = (1/P) * (sum(K(p)^2 * dp))
    P: contour length
    dp: unit contour arc length
    K(p): the rate of change in tangent direction θ of the contour,
          as a function of the arc length p (dp)
          K(p) = d_theta/dp
    :param data:
    :param window:
    :return:
    """
    p1 = np.concatenate((data[-window:], data[:-window])).T
    p2 = data.copy().T
    p3 = np.concatenate((data[window:], data[0:window])).T
    p1p2 = p1[0]*1+p1[1]*1j - (p2[0]*1+p2[1]*1j)
    p1p3 = p1[0]*1+p1[1]*1j - (p3[0]*1+p3[1]*1j)

    d_theta = np.angle(p1p3/p1p2)
    d_p = np.sqrt(np.sum(np.square(p2-p1).T, axis=1))
    P = d_p.sum()
    K_p = d_theta/d_p

    BE = np.sum(K_p**2/d_p)/P

    return BE


def circularity(regionprop):
    perimeter, area = regionprop.perimeter, regionprop.area
    return round(4*np.pi*area/(perimeter**2), 3)


def convexity(regionprop):
    image_perimeter = regionprop.perimeter
    convex_image_perimeter = measure.regionprops(regionprop.convex_image*1)[0].perimeter
    return round(convex_image_perimeter/image_perimeter, 3)


def width_omit_odds(width_list):
    new_list = []
    for w in width_list:
        new_list += list(w[2:-2])
    return np.array(new_list)


def angle_between_vector(v1, v2):
    d1 = np.sqrt(np.sum(v1 ** 2))
    d2 = np.sqrt(np.sum(v2 ** 2))
    return np.arccos(np.dot(v1, v2) / (d1 * d2))


def puncta_projection(cell, channel):
    puncta = cell.fluorescent_puncta[channel]
    p_puncta = puncta[:, :2]
    midline = cell.midlines[0]
    width = cell.width_lists[0]
    L = measure_length(midline)
    output = []
    for pp in p_puncta:
        v_p = pp[np.newaxis, :] - midline
        v_p_length = np.sqrt(np.sum(v_p ** 2, axis=1))

        nearest_loc = np.argmin(v_p_length)
        if nearest_loc == len(midline) - 1:
            v_m = midline[-1] - midline[-2]
        else:
            v_m = midline[nearest_loc + 1] - midline[nearest_loc]

        dl = v_p_length[nearest_loc]
        v_x_proj = np.array([1, 0])

        ang1 = angle_between_vector(v_m, v_x_proj)
        ang2 = angle_between_vector(v_p[nearest_loc], v_x_proj)
        ang12 = ang1 - ang2
        dx = np.cos(ang12) * dl
        dy = np.sin(ang12) * dl

        if nearest_loc == 0:
            l = 0
        else:
            l = measure_length(midline[:nearest_loc + 1])

        w = width[nearest_loc]

        if w == 0:
            norm_dy = dy / (2*np.max(width))
            norm_dx = dx / L
        else:
            norm_dx, norm_dy = dx / L, dy / w
        norm_l = l / L
        output.append([l, dl, dx, dy, norm_l, norm_dx, norm_dy])
    output = np.array(output)
    return output


def minimal_intrapuncta_distance(cell, channel):
    puncta = cell.fluorescent_puncta[channel]
    x_coords = puncta[:, 0]
    y_coords = puncta[:, 1]
    x_dist = x_coords[np.newaxis, :] - x_coords[:, np.newaxis]
    y_dist = y_coords[np.newaxis, :] - y_coords[:, np.newaxis]
    dist_matrix = np.sqrt(x_dist ** 2 + y_dist ** 2)
    dist_matrix[dist_matrix == 0] = np.inf
    return np.min(dist_matrix, axis=0) * cell.pixel_microns


def fractional_positions(cell, channel):
    counts = []
    fractions = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if len(cell.fluorescent_puncta[channel]) > 0:
        projected = puncta_projection(cell, channel)
        x_pos = projected[:, 4] + projected[:, 5]
        pos_count = np.sum((x_pos[:, np.newaxis] - fractions) >= 0, axis=1) - 1
        pos_count[pos_count == -1] = 0
        pos_count[pos_count == 5] = 4
        for i in range(5):
            counts.append(np.count_nonzero(pos_count == i))
    else:
        counts = [0] * 5
    return (counts)


def puncta_midline_dist(cell, channel):
    if len(cell.fluorescent_puncta[channel]) > 0:
        midline = cell.midlines[0]
        puncta = cell.fluorescent_puncta[channel]
        xy_coords = puncta[:, :2]
        dist_matrix = np.sqrt(np.sum((xy_coords[:, np.newaxis] - midline) ** 2, axis=2))
        dist_min = np.min(dist_matrix, axis=1).mean()
    else:
        dist_min = 0
    return dist_min


def puncta_contour_dist(cell, channel):
    if len(cell.fluorescent_puncta[channel]) > 0:
        contour = cell.optimized_contour
        puncta = cell.fluorescent_puncta[channel]
        xy_coords = puncta[:, :2]
        dist_matrix = np.sqrt(np.sum((xy_coords[:, np.newaxis] - contour) ** 2, axis=2))
        dist_min = np.min(dist_matrix, axis=1).mean()
    else:
        dist_min = 0
    return dist_min


def symmetric_point(data1d):
    xcor = np.linspace(0, 1, 100)
    data = np.interp(xcor,
                     np.linspace(0, 1, len(data1d)),
                     data1d)
    cumsum = np.cumsum(data)

    half_sum = 0.5 * data.sum()
    i1, i2 = np.where(cumsum > half_sum)[0][0:2]
    v1, v2 = cumsum[i1], cumsum[i2]
    x1, x2 = xcor[i1], xcor[i2]

    x = ((half_sum - v1) / (v2 - v1)) * (x2 - x1) + x1

    if x >= 0.5:
        x = 1 - x
    return round(x, 3)


def normalized_contour_complexity(cell, window=1):
    contour = cell.init_contour
    angles = np.abs(bend_angle(contour, window=window))
    angles[angles <= 15] = 0
    complexity_cell = 0.5*angles.sum()/len(contour)
    area = cell.regionprop.area
    L = cell.regionprop.major_axis_length
    skeleton_L = len(np.nonzero(cell.skeleton)[0])
    L = max(L, skeleton_L)
    standard_complexity = standard_rod_complexity(area, L, window, 2*len(contour))
    return complexity_cell/standard_complexity
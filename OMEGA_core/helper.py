__author__ = "jz-rolling"

# OMEGA helper
# 03/17/2020

import numpy as np
import pandas as pd
import cmath
from math import pi
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from scipy import ndimage as ndi
from itertools import combinations
from scipy import fftpack
from skimage import morphology, feature, measure, filters, segmentation, util, exposure, draw, transform
from scipy.interpolate import splprep, splev, RectBivariateSpline
import warnings
from numba import jit
from scipy.optimize import curve_fit


def ImageJinfo2dict(ImageJinfo):
    """
    Extract metadata from imageJ processsed .tif files
    :param ImageJinfo: imagej converted metadata
    :return: dictionary of metadata
    """
    if "\n" in ImageJinfo:
        ImageJinfo = ImageJinfo.split("\n")
    ImageJdict = {}
    for info in ImageJinfo:
        info = info.split(" = ")
        if len(info) == 2:
            key = info[0]
            val = info[1]
            if key.startswith(" "):
                key = key[1:]
            if val.startswith(" "):
                val = val[1:]
            ImageJdict[key] = val
    return ImageJdict


def shift_image(img, shift):
    """
    correct xy drift between phase contrast image and fluorescent image(s)
    :param img: input image
    :param shift: subpixel xy drift
    :return: drift corrected image
    """
    offset_image = ndi.fourier_shift(np.fft.fftn(img), shift)
    offset_image = np.fft.ifftn(offset_image)
    offset_image = np.round(offset_image.real)
    offset_image[offset_image <= 0] = 10

    return offset_image.astype(np.uint16)


def _plot_spectrum(im_fft):
    """
    spectrum plot, deprecated
    :param im_fft:
    :return:
    """
    plt.figure()
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5), cmap="Greys")
    plt.colorbar()


def fft(img, subtract_mean=True):
    """
    fast Fourier transform module
    :param img: input image
    :param subtract_mean:
    :return: FFT transformed image
    """
    warnings.filterwarnings("ignore")
    if subtract_mean:
        img = img - np.mean(img)
    return (fftpack.fftshift(fftpack.fft2(img)))


def fft_reconstruction(fft_img, filters):
    """
    reconstruct image after FFT band pass filtering.
    :param fft_img: input FFT transformed image
    :param filters: low/high frequency bandpass filters
    :return: bandpass filtered, restored phase contrast image
    """

    warnings.filterwarnings("ignore")
    if len(filters) > 0:
        for filter in filters:
            try:
                fft_img *= filter
            except:
                raise ValueError("Illegal input filter found, shape doesn't match?")
    return (fftpack.ifft2(fftpack.ifftshift(fft_img)).real)



def bandpass_filter(pixel_microns, img_width=2048, img_height=2048, high_pass_width=0.2, low_pass_width=20):

    """

    :param pixel_microns: pixel unit length
    :param img_width: width of image by pixel
    :param img_height: height of image by pixel
    :param high_pass_width: 1/f where f is the lower bound of high frequency signal
    :param low_pass_width: 1/f where f is the upper bound of low frequency signal
    :return: high/low bandpass filters

    """
    u_max = round(1 / pixel_microns, 3) / 2
    v_max = round(1 / pixel_microns, 3) / 2
    u_axis_vec = np.linspace(-u_max / 2, u_max / 2, img_width)
    v_axis_vec = np.linspace(-v_max / 2, v_max / 2, img_height)
    u_mat, v_mat = np.meshgrid(u_axis_vec, v_axis_vec)
    centered_mesh = np.sqrt(u_mat ** 2 + v_mat ** 2)
    if high_pass_width == 0:
        high_pass_filter = np.ones((img_width, img_height)).astype(np.int)
    else:
        high_pass_filter = np.e ** (-(centered_mesh * high_pass_width) ** 2)
    if low_pass_width == 0:
        low_pass_filter = np.ones((2048, 2048)).astype(np.int)
    else:
        low_pass_filter = 1 - np.e ** (-(centered_mesh * low_pass_width) ** 2)
    return (high_pass_filter, low_pass_filter)



def is_integer(x):
    try:
        isinstance(x, (int))
        return True
    except:
        return False



def adjust_image(img, dtype=16, adjust_gamma=True, gamma=1):
    """
    adjust image data depth and gamma value
    :param img: input image
    :param dtype: bit depth, 8, 12 or 16
    :param adjust_gamma: whether or not correct gamma
    :param gamma: gamma value
    :return: adjusted image
    """
    if is_integer(dtype) & (dtype > 2):
        n_range = (0, 2 ** dtype - 1)
    else:
        print("Illegal input found where an integer no less than 2 was expected.")
    outimg = exposure.rescale_intensity(img, out_range=n_range)
    if adjust_gamma:
        outimg = exposure.adjust_gamma(outimg, gamma=gamma)
    return outimg


# compute shape-index of given image and convert float output to 8bit
def shape_index_conversion(img, shape_indexing_sigma=2):
    """
    Jan J Koenderink and Andrea J van Doorn proposed a simple solution for numeric representation of local
    curvature, here we use an 8-bit converted shape index measure as the basis to infer different parts of a cell body
    :param img: input image
    :param shape_indexing_sigma: sigma for computing Hessian eigenvalues
    :return: shape index converted image (8-bit)
    """
    shape_indexed = feature.shape_index(img, sigma=shape_indexing_sigma)
    shape_indexed = np.nan_to_num(shape_indexed, copy=True)
    shape_indexed = (exposure.equalize_hist(shape_indexed) * 255).astype(np.uint8)
    return shape_indexed


def init_segmentation(img, shape_indexing_sigma=2, min_particle_size=40):
    """
    primary image segmentation that splits disconnected pixel patches into Clusters

    :param img: input image, usually phase contrast data
    :param shape_indexing_sigma:
    :param min_particle_size:
    :return:
    """
    #
    shape_indexed = shape_index_conversion(img, shape_indexing_sigma=shape_indexing_sigma)
    #
    ph_gau = filters.gaussian(img, sigma=1)
    ph_binary_glob = (ph_gau < filters.threshold_isodata(ph_gau))*1
    ph_binary_local = (ph_gau < filters.threshold_local(ph_gau, method="mean", block_size=15))
    ph_binary_local[morphology.dilation(ph_binary_glob, morphology.disk(5)) == 0] = 0
    #ph_binary_local[ph_binary_glob == 0] = 0
    ph_binary_local.dtype = bool
    ph_binary_local = morphology.remove_small_objects(ph_binary_local, min_size=min_particle_size)
    ph_binary_local = morphology.remove_small_holes(ph_binary_local, area_threshold=80)*1
    ph_binary_local = (filters.median(ph_binary_local,morphology.disk(2))>0).astype(int)
    ph_binary_local = morphology.opening(ph_binary_local, morphology.disk(2))*1

    microcolony_labels = measure.label(ph_binary_local, connectivity=1)
    region_info = measure.regionprops(microcolony_labels)
    #return ph_binary_local, shape_indexed, shape_indexed_binary, microcolony_labels, region_info
    return ph_binary_local, shape_indexed, microcolony_labels, region_info


class rolling_ball:
    def __init__(self, radius=50):
        self.width = 0
        if radius <= 10:
            self.shrink_factor = 1
            self.arc_trim_per = 24
        elif radius <= 20:
            self.shrink_factor = 2
            self.arc_trim_per = 24
        elif radius <= 100:
            self.shrink_factor = 4
            self.arc_trim_per = 32
        else:
            self.shrink_factor = 8
            self.arc_trim_per = 40
        self.radius = radius / self.shrink_factor
        self.build()

    def build(self):
        x_trim = int(self.arc_trim_per * self.radius / 100)
        half_width = int(self.radius - x_trim)
        self.width = int(2 * half_width + 1)
        r_squre = np.ones((self.width, self.width)) * (self.radius ** 2)
        squared = np.square(np.linspace(0, self.width - 1, self.width) - half_width)
        x_val = np.tile(squared, (self.width, 1))
        y_val = x_val.T
        self.ball = np.sqrt(r_squre - x_val - y_val)


def rolling_ball_bg_subtraction(data, radius=40):
    output_data = data.copy()
    smoothed = filters.gaussian(data, sigma=1) * 65535
    ball = rolling_ball(radius=radius)
    shrunk_img = shrink_img_local_min(smoothed, shrink_factor=ball.shrink_factor)
    bg = rolling_ball_bg(shrunk_img, ball.ball)
    bg_rescaled = transform.resize(bg, data.shape, anti_aliasing=True)
    output_data = output_data - bg_rescaled
    output_data[output_data <= 0] = 0
    return (output_data.astype(np.uint16))


@jit(nopython=True, cache=True)
def shrink_img_local_min(image, shrink_factor=4):
    s = shrink_factor
    r, c = image.shape[0], image.shape[1]
    r_s, c_s = int(r / s), int(c / s)
    shrunk_img = np.ones((r_s, c_s))
    for x in range(r_s):
        for y in range(c_s):
            shrunk_img[x, y] = image[x * s:x * s + s, y * s:y * s + s].min()
    return shrunk_img


@jit(nopython=True, cache=True)
def rolling_ball_bg(image, ball):
    width = ball.shape[0]
    radius = int(width / 2)
    r, c = image.shape[0], image.shape[1]
    bg = np.ones((r, c)).astype(np.float32)
    # ignore edges to begin with
    for x in range(r):
        for y in range(c):
            x1, x2, y1, y2 = max(x - radius, 0), min(x + radius + 1, r), max(y - radius, 0), min(y + radius + 1, c)
            cube = image[x1:x2, y1:y2]
            cropped_ball = ball[radius - (x - x1):radius + (x2 - x), radius - (y - y1):radius + (y2 - y)]
            bg_cropped = bg[x1:x2, y1:y2]
            bg_mask = ((cube - cropped_ball).min()) + cropped_ball
            bg[x1:x2, y1:y2] = bg_cropped * (bg_cropped >= bg_mask) + bg_mask * (bg_cropped < bg_mask)
    return (bg)


def optimize_bbox(img_shape, bbox, edge_width = 8):
    (rows,columns) = img_shape
    (x1,y1,x2,y2) = bbox
    return(max(0,x1-edge_width),max(0,y1-edge_width),\
           min(rows-1,x2+edge_width),min(columns-1,y2+edge_width))


def touching_edge(img_shape,optimized_bbox):
    (rows, columns) = img_shape
    (x1, y1, x2, y2) = optimized_bbox
    if min(x1, y1, rows-x2-1,columns-y2-1) <= 0:
        return True
    else:
        return False


def orientation_by_eig(regionprop):
    tensor = regionprop.inertia_tensor
    u20,u11,u02 = tensor[0,0],-tensor[0,1],tensor[1,1]
    if u20 == u02:
        if u11 == 0:
            theta = 0
        elif u11 > 0:
            theta = 0.25*np.pi
        elif u11 < 0:
            theta = -0.25*np.pi
    else:
        if u11 == 0:
            if u20<u02:
                theta = -0.5*np.pi
            elif u20>u02:
                theta = 0
        elif u20 > u02:
            theta = 0.5*np.arctan(2*u11/(u20-u02))
        elif u20 < u02:
            if u11 > 0:
                theta = 0.5*np.arctan(2*u11/(u20-u02)) + 0.5*np.pi
            elif u11 < 0:
                theta = 0.5*np.arctan(2*u11/(u20-u02)) - 0.5*np.pi
    return theta


def distance(v1, v2):
    #Euclidean distance of two points
    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2))


def in_range(val,min_val,max_val):
    if (val < min_val) or (val > max_val):
        return(False)
    else:
        return(True)


class merge_joint:

    def __init__(self, data, old_idxlist):
        self.data = [set(elem) for elem in data]
        self.taken = [False] * len(self.data)
        self.newdata = []
        self.old_idxlist = old_idxlist
        self.new_idxlist = []
        self.merge_sets()

    def dfs(self, node, idx):
        self.taken[idx] = True
        newset = node
        self.new_idxlist[-1].append(self.old_idxlist[idx])
        for i, item in enumerate(self.data):
            if not self.taken[i] and not newset.isdisjoint(item):
                newset.update(self.dfs(item, i))
        return newset

    def merge_sets(self):
        for idx, node in enumerate(self.data):
            if not self.taken[idx]:
                self.new_idxlist.append([])
                self.newdata.append(list(self.dfs(node, idx)))


def locate_nodes(skeleton):
    endpoints = []
    branch_points = []
    skeleton_path = np.where(skeleton>0)
    skeleton_length = len(skeleton_path[0])
    if skeleton_length > 5:
        for i in range(skeleton_length):
            x = skeleton_path[0][i]
            y = skeleton_path[1][i]
            if cube_nonzero(skeleton, x, y) == 1:
                endpoints.append([x, y])
            if cube_nonzero(skeleton,x, y) == 2:
                _n,  _neighbors = cube_nonzero_with_return(skeleton, x, y)
                _x0, _y0 = _neighbors[0]
                _x1, _y1 = _neighbors[1]
                dist=abs(_x0-_x1) +abs(_y0-_y1)
                if dist == 1:
                    skeleton[x, y] = 0
                    if skeleton[_x0, _y0] == 3:
                        skeleton[_x0, _y0] = 2
                    if skeleton[_x1, _y1] == 3:
                        skeleton[_x1, _y1] = 2
                    if [_x0, _y0] in branch_points:
                        branch_points.remove([_x0, _y0])
                    if [_x1, _y1] in branch_points:
                        branch_points.remove([_x1, _y1])
            if cube_nonzero(skeleton, x, y) > 2:
                branch_points.append([x, y])
                skeleton[x, y] = 3
        return endpoints,branch_points,skeleton
    else:
        return endpoints,branch_points,skeleton


@jit(nopython=True, cache=True)
def neighbor_search(input_map, endpoint, max_iterations=500):
    output_x = []
    output_y = []
    end_reached = False
    x, y = endpoint[0], endpoint[1]
    counter=0
    while not end_reached:
        if counter >= max_iterations:
            break
        n_neighbor, neighbor_list = cubecount_with_return(input_map, x, y, 2)
        if n_neighbor == 0:
            input_map[x, y] = 1
            output_x.append(x)
            output_y.append(y)
            end_reached = True
        elif n_neighbor == 1:
            input_map[x, y] = 1
            output_x.append(x)
            output_y.append(y)
            x, y = neighbor_list[-1]
        elif n_neighbor >= 2:
            input_map[x, y] = 1
            output_x.append(x)
            output_y.append(y)
            end_reached = True
        counter += 1
    return output_x, output_y


@jit(nopython=True, cache=True)
def cubesum(mask, px, py, val):
    n = 0
    for i in [0, 1, 2, 3, 5, 6, 7, 8]:
        dx = int(i / 3)
        dy = i - dx * 3
        x1, y1 = px + dx - 1, py + dy - 1
        if mask[x1, y1] == val:
            n += 1
    return n


@jit(nopython=True, cache=True)
def cube_nonzero(mask, px, py):
    n = 0
    for i in [0, 1, 2, 3, 5, 6, 7, 8]:
        dx = int(i / 3)
        dy = i - dx * 3
        x1, y1 = px + dx - 1, py + dy - 1
        if mask[x1, y1] > 0:
            n += 1
    return n


@jit(nopython=True, cache=True)
def cube_nonzero_with_return(mask, px, py):
    retlist = []
    n = 0
    for i in [0, 1, 2, 3, 5, 6, 7, 8]:
        dx = int(i / 3)
        dy = i - dx * 3
        x1, y1 = px + dx - 1, py + dy - 1
        if mask[x1, y1] > 0:
            retlist.append([x1, y1])
            n += 1
    return n, retlist


@jit(nopython=True, cache=True)
def cubecount_with_return(mask, px, py, val):
    retlist = []
    n = 0
    for i in [0, 1, 2, 3, 5, 6, 7, 8]:
        dx = int(i / 3)
        dy = i - dx * 3
        x1, y1 = px + dx - 1, py + dy - 1
        if mask[x1, y1] == val:
            retlist.append([x1, y1])
            n += 1
    return n, retlist


def skeleton_analysis(mask, pruning=False, min_branch_length=5, max_iterations=30):
    skeleton = morphology.skeletonize(mask)*2
    endpoints, branch_points, skeleton = locate_nodes(skeleton)
    anchor_points = endpoints+branch_points
    skeleton_branches = []
    n = 0
    if (len(endpoints) >= 2) and (len(branch_points) <= 10):
        while True:
            if len(anchor_points) == 0:
                break
            if n>=max_iterations:
                break
            is_real_pole = [0,0]
            startpoint = anchor_points[0]
            if startpoint in endpoints:
                is_real_pole[0] = 1
            xcoords,ycoords = neighbor_search(skeleton,startpoint)
            anchor_points.remove(startpoint)
            if len(xcoords)>=1:
                lx,ly = xcoords[-1],ycoords[-1]
                node_count, node_coords = cubecount_with_return(skeleton, lx, ly, 3)
                if (node_count == 0) and ([lx,ly] in endpoints):
                    is_real_pole[1] = 1
                    anchor_points.remove([lx, ly])
                elif (node_count == 1):
                    lx, ly = node_coords[0]
                    if [lx, ly] in anchor_points:
                        anchor_points.remove([lx, ly])
                    xcoords.append(lx)
                    ycoords.append(ly)
                else:
                    for xy in node_coords:
                        if distance([lx,ly],xy) == 1:
                            lx,ly = xy
                            break
                    if [lx, ly] in anchor_points:
                        anchor_points.remove([lx, ly])
                    xcoords.append(lx)
                    ycoords.append(ly)
                skeleton_branches.append([is_real_pole, xcoords, ycoords])
            n+=1
        branch_copy = skeleton_branches.copy()
        if pruning:
            for branch in branch_copy:
                if len(branch) <= min_branch_length:
                    skeleton_branches.remove(branch)
        if n < max_iterations:
            return skeleton_branches, skeleton
        else:
            return [], skeleton
    else:
        return [], skeleton


def interpolate_2Dmesh(data_array, smooth=1):
    x_mesh = np.linspace(0, data_array.shape[0] - 1, data_array.shape[0]).astype(int)
    y_mesh = np.linspace(0, data_array.shape[1] - 1, data_array.shape[1]).astype(int)
    return RectBivariateSpline(x_mesh, y_mesh, data_array, s=smooth)


def quadratic_maxima_approximation(q1, q2, q3):
    if q2 <= max(q1, q3):
        raise ValueError('Mid point must be local maxima!')
    else:
        return (0.5 * (q1 - q3)) / (q1 - 2 * q2 + q3)


def spline_approximation(init_contour, n=200, smooth_factor=1, closed = True):
    if closed:
        tck, u = splprep(init_contour.T, u=None, s=smooth_factor, per=1)
    else:
        tck, u = splprep(init_contour.T, u=None, s=smooth_factor)
    u_new = np.linspace(u.min(), u.max(), n)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.array([x_new, y_new]).T

"""
unstable function, deprecated
def contour_optimization(contour, sobel):
    newcontour = contour.copy()
    dxy = unit_perpendicular_vector(newcontour)
    fg = interpolate_2Dmesh(sobel)
    converged_list = np.ones(len(newcontour))
    for i in range(len(newcontour)):
        x, y = newcontour[i]
        dx, dy = dxy[i]
        p1, p2, p3 = fg(x - dx, y - dy), fg(x, y), fg(x + dx, y + dy)
        if p2 > max(p1, p3):
            r = quadratic_maxima_approximation(p1, p2, p3)
            newcontour[i] = [x + r * dx, y + r * dy]
            converged_list[i] = 0
        elif p1 > p3:
            pm = fg(x - dx * 0.5, y - dy * 0.5)
            if pm > max(p1, p2):
                r = quadratic_maxima_approximation(p1, pm, p2)
                newcontour[i] = [x + r * dx * 0.5, y + r * dy * 0.5]
                converged_list[i] = 0
        elif p1 < p3:
            pm = fg(x + dx * 0.5, y + dy * 0.5)
            if pm > max(p2, p3):
                r = quadratic_maxima_approximation(p2, pm, p3)
                newcontour[i] = [x + r * dx * 0.5, y + r * dy * 0.5]
                converged_list[i] = 0
    newcontour = spline_approximation(newcontour, n=len(contour), smooth_factor=3)
    return newcontour, converged_list
"""


def contour_optimization(input_contour,
                         input_sobel,
                         step=0.5,
                         edge_cutoff=0.2):
    sobel = input_sobel.copy()
    contour = input_contour.copy()
    converged = np.ones(len(contour))

    for i in range(5):

        # stop if converged
        if converged.max() == 0:
            break

        unstable_points = np.where(converged == 1)[0]
        updated_contour = contour[unstable_points].copy()
        unit_perp = (unit_perpendicular_vector(contour) * step)[unstable_points]

        inward = updated_contour + unit_perp
        outward = updated_contour - unit_perp

        v0 = bilinear_interpolate_numpy(sobel, updated_contour[:, 0], updated_contour[:, 1])
        v_in = bilinear_interpolate_numpy(sobel, inward[:, 0], inward[:, 1])
        v_out = bilinear_interpolate_numpy(sobel, outward[:, 0], outward[:, 1])

        vet = np.array([v_in, v0, v_out]).T
        vet_max_id = np.argmax(vet, axis=1)

        # suppress edge pixels with low sobel intensities
        suppressed_edge_id = np.where(vet[np.arange(len(vet)), vet_max_id] < edge_cutoff)

        quadratic_a = (0.5 * (v_in - v_out))
        quadratic_b = (v_in - 2 * v0 + v_out)
        quadratic_b[quadratic_b == 0] = np.inf
        dv = quadratic_a / quadratic_b
        dv[suppressed_edge_id] = 0
        updated_contour[vet_max_id == 1] -= dv[vet_max_id == 1, np.newaxis] * unit_perp[vet_max_id == 1]
        vet_max_id[suppressed_edge_id] = 1

        # move inward
        inward_id = np.where(vet_max_id == 0)
        updated_contour[inward_id] = (updated_contour + unit_perp * 0.5)[inward_id]

        # move outward
        outward_id = np.where(vet_max_id == 2)
        updated_contour[outward_id] = (updated_contour - unit_perp * 0.5)[outward_id]

        # update convergence list
        converged[unstable_points[vet_max_id == 1]] = 0

        # update contour
        contour[unstable_points] = updated_contour

    # close contour

    # remove extrema
    dist = np.sum(np.square(input_contour - contour), axis=1)
    contour[converged == 1] = input_contour[converged == 1]
    contour[dist >= 1.5] = input_contour[dist >= 1.5]
    contour[-1] = contour[0]

    # even number segments
    return spline_approximation(contour,
                                n=2*len(contour),
                                smooth_factor=5), converged


def unit_perpendicular_vector(data, closed=True):

    p1 = data[1:]
    p2 = data[:-1]
    dxy = p1 - p2
    ang = np.arctan2(dxy.T[1], dxy.T[0]) + 0.5 * np.pi
    dx, dy = np.cos(ang), np.sin(ang)
    unit_dxy = np.array([dx, dy]).T
    if not closed:
        unit_dxy = np.concatenate([[unit_dxy[0]], unit_dxy])
    else:
        unit_dxy = np.concatenate([unit_dxy,[unit_dxy[-1]]])
    return unit_dxy


def midline_approximation(skeleton, smoothed_contour,
                          tolerance=0.1, max_iteration=10):
    midline = skeleton.copy()
    n = 0
    converged = False
    while n < max_iteration:
        updated_midline = direct_intersect_points(midline, smoothed_contour)
        dxy = updated_midline - midline
        midline = spline_approximation(updated_midline, n=len(updated_midline),
                                       smooth_factor=1, closed=False)
        if dxy.max() <= tolerance:
            converged = True
            break
        n += 1
    return midline.astype(np.float), converged


# inefficient width calculation, deprecated
def measure_width(extended_skeleton, smoothed_contour):
    length = line_length(extended_skeleton)
    extended_skeleton = spline_approximation(extended_skeleton,
                                             int(round(length)),
                                             smooth_factor=0, closed=False)
    d_perp = unit_perpendicular_vector(extended_skeleton, closed=False)
    width_list = []
    for i in range(1, len(extended_skeleton)-1):
        xy = line_contour_intersection(extended_skeleton[i],
                                       d_perp[i] + extended_skeleton[i],
                                       smoothed_contour)
        coords = np.average(xy, axis=0)
        if (len(xy) == 2) and (np.isnan(coords).sum() == 0):
            width_list.append(distance(xy[0], xy[1]))
        else:
            raise ValueError('Error encountered while computing line intersection points!')
    return extended_skeleton, np.array([0]+width_list+[0]), length


def direct_intersect_distance(skeleton, contour):
    v1, v2 = contour[:-1], contour[1:]
    skel_x, skel_y = skeleton[1:-1].T
    intersect_x, intersect_y = intersect_matrix(skeleton[1:-1], contour)
    dx_v1 = intersect_x - v1.T[0][:, np.newaxis]
    dx_v2 = intersect_x - v2.T[0][:, np.newaxis]
    dy_v1 = intersect_y - v1.T[1][:, np.newaxis]
    dy_v2 = intersect_y - v2.T[1][:, np.newaxis]
    dx = dx_v1 * dx_v2
    dy = dy_v1 * dy_v2
    dist_x = skel_x[np.newaxis, :] - intersect_x
    dist_y = skel_y[np.newaxis, :] - intersect_y

    non_boundry_points = np.where(np.logical_and(dy > 0, dx > 0))
    dist_matrix = np.sqrt(dist_x ** 2 + dist_y ** 2)
    dist_matrix[non_boundry_points] = np.inf
    nearest_id_x = np.argsort(dist_matrix, axis=0)[:2]
    nearest_id_y = np.linspace(0, dist_matrix.shape[1] - 1, dist_matrix.shape[1]).astype(int)
    dists = dist_matrix[nearest_id_x[0], nearest_id_y] + dist_matrix[nearest_id_x[1], nearest_id_y]
    return np.concatenate([[0], dists, [0]])


def suppress_extreme_edge_points(edge, tolerance=1):
    dif = edge[1:] - edge[:-1]
    dist = np.sqrt(np.sum(dif ** 2, axis=1))
    extremities = np.where(dist > tolerance)[0]
    new_edge = edge.copy()
    if len(extremities) > 0:
        for i in extremities:
            new_edge[i + 1] = 2 * edge[i] - edge[i - 1]
    return new_edge


def straighten_by_orthogonal_lines(contour, midline, length, width, unit_micron=0.05):
    # estimate profile mesh size
    median_width = np.median(width)
    N_length = int(round(length / unit_micron))
    N_width = int(round(median_width / unit_micron))

    # interpolate midline
    midline = spline_approximation(midline, N_length, smooth_factor=0, closed=False)

    # divide contour
    #if contour[-1] == contour[-2]:
        #contour = contour[:-1]
    half_contour_1, half_contour_2 = divide_contour_by_midline(midline, contour[:-1])

    # infer orthogonal vectors
    ortho_unit_vectors = unit_perpendicular_vector(midline)

    # generate orthogonal profile lines for all midline points except for the polar ones
    l1 = orthogonal_intersection_point(midline, half_contour_1,
                                       precomputed_orthogonal_vector=ortho_unit_vectors)

    l2 = orthogonal_intersection_point(midline, half_contour_2,
                                       precomputed_orthogonal_vector=ortho_unit_vectors)

    # suppress extreme events
    # opt_l1 = suppress_extreme_edge_points(l1)
    # opt_l2 = suppress_extreme_edge_points(l2)
    # opt_l1 = l1
    # opt_l2 = l2
    dl = (l2 - l1) / N_width
    mult_mat = np.tile(np.arange(N_width + 1), (len(l1), 1))
    mat_x = l1[:, 0][:, np.newaxis] + mult_mat * dl[:, 0][:, np.newaxis]
    mat_y = l1[:, 1][:, np.newaxis] + mult_mat * dl[:, 1][:, np.newaxis]
    profile_mesh = np.array([mat_x, mat_y])
    return l1, l2, profile_mesh, midline


def divide_contour_by_midline(midline, contour):
    dist1 = distance_matrix(contour, midline[0]).flatten()
    dist2 = distance_matrix(contour, midline[-1]).flatten()

    id1, id2 = np.argsort(dist1)[:2]
    id3, id4 = np.argsort(dist2)[:2]

    contour_cp = contour.copy()
    if max(id1, id2) < max(id3, id4):
        term_p1 = max(id1, id2)
        if abs(id3 - id4) == 1:
            term_p2 = max(id3, id4) + 1
        elif abs(id3 - id4) > 1:
            term_p2 = max(id3, id4) + 2
        contour_cp = np.insert(contour_cp, term_p1, midline[0], axis=0)
        contour_cp = np.insert(contour_cp, term_p2, midline[-1], axis=0)

    else:
        term_p1 = max(id3, id4)
        if abs(id1 - id2) == 1:
            term_p2 = max(id1, id2) + 1
        elif abs(id1 - id2) > 1:
            term_p2 = max(id1, id2) + 2
        contour_cp = np.insert(contour_cp, term_p1, midline[-1], axis=0)
        contour_cp = np.insert(contour_cp, term_p2, midline[0], axis=0)

    if term_p1 == term_p2:
        raise ValueError('Two endpoints are identical!')
    else:
        pos1, pos2 = sorted([term_p1, term_p2])
        #print(id1, id2, id3, id4, pos1, pos2, len(contour_cp))
        half_contour_1 = contour_cp[pos1:min(pos2 + 1, len(contour_cp) - 1)]
        half_contour_2 = np.concatenate([contour_cp[pos2:], contour_cp[:pos1 + 1]])
    return half_contour_1, half_contour_2


def orthogonal_intersection_point(midline,
                                  outerline,
                                  precomputed_orthogonal_vector=None,
                                  min_dist=0.001):
    v1, v2 = outerline[:-1], outerline[1:]
    skel_x, skel_y = midline.T
    if precomputed_orthogonal_vector is None:
        intersect_x, intersect_y = intersect_matrix(midline, outerline)
    else:
        intersect_x, intersect_y = intersect_matrix(midline, outerline,
                                                    orthogonal_vectors=precomputed_orthogonal_vector)
    dx_v1 = intersect_x - v1.T[0][:, np.newaxis]
    dx_v2 = intersect_x - v2.T[0][:, np.newaxis]
    dy_v1 = intersect_y - v1.T[1][:, np.newaxis]
    dy_v2 = intersect_y - v2.T[1][:, np.newaxis]
    dx = dx_v1 * dx_v2
    dy = dy_v1 * dy_v2

    dist_x = skel_x[np.newaxis, :] - intersect_x
    dist_y = skel_y[np.newaxis, :] - intersect_y

    non_bounadry_points = np.where(np.logical_and(dy >= 0, dx >= 0))
    dist_matrix = np.sqrt(dist_x ** 2 + dist_y ** 2)
    dist_matrix[non_bounadry_points] = np.inf
    dist_matrix[dist_matrix <= min_dist] = np.inf
    nearest_id_x = np.argsort(dist_matrix, axis=0)[:1]
    nearest_id_y = np.linspace(0, dist_matrix.shape[1] - 1, dist_matrix.shape[1]).astype(int)
    pos_list = np.array([intersect_x[nearest_id_x[0], nearest_id_y],
                         intersect_y[nearest_id_x[0], nearest_id_y]]).T
    return pos_list


def measure_length(data, pixel_microns=1):
    v1,v2 = data[:-1], data[1:]
    length = np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2, axis=1)).sum()*pixel_microns
    return(length)


def spline_approximation(init_contour, n=200, smooth_factor=1, closed=True):
    if closed:
        tck, u = splprep(init_contour.T, u=None, s=smooth_factor, per=1)
    else:
        tck, u = splprep(init_contour.T, u=None, s=smooth_factor)
    u_new = np.linspace(u.min(), u.max(), n)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.array([x_new, y_new]).T


def find_poles(smoothed_skeleton,
               smoothed_contour,
               find_pole1=True,
               find_pole2=True):
    # find endpoints and their nearest neighbors on a midline
    length = len(smoothed_skeleton)
    extended_pole1 = [smoothed_skeleton[0]]
    extended_pole2 = [smoothed_skeleton[-1]]
    i = 0
    j = 0
    if find_pole1:
        for i in range(5):
            p1 = smoothed_skeleton[i]
            p2 = smoothed_skeleton[i + 1]
            # find the two intersection points between
            # the vectorized contour and line through pole1
            intersection_points_pole1 = line_contour_intersection(p1, p2, smoothed_contour)
            # find the interesection point with the same direction as the outward pole vector
            dxy_1 = p1 - p2
            p1_tile = np.tile(p1, (len(intersection_points_pole1), 1))
            p1dot = (intersection_points_pole1 - p1_tile) * dxy_1
            index_1 = np.where((p1dot[:, 0] > 0) & (p1dot[:, 1] > 0))[0]
            if len(index_1) > 0:
                extended_pole1 = intersection_points_pole1[index_1]
                break
    else:
        i = 1

    if find_pole2:
        for j in range(5):
            p3 = smoothed_skeleton[-1 - j]
            p4 = smoothed_skeleton[-2 - j]
            # find the two intersection points between
            # the vectorized contour and line through pole1
            intersection_points_pole2 = line_contour_intersection(p3, p4, smoothed_contour)
            # find the interesection point with the same direction as the outward pole vector
            dxy_2 = p3 - p4
            p3_tile = np.tile(p3, (len(intersection_points_pole2), 1))
            p3dot = (intersection_points_pole2 - p3_tile) * dxy_2
            index_2 = np.where((p3dot[:, 0] > 0) & (p3dot[:, 1] > 0))[0]
            if len(index_2) > 0:
                extended_pole2 = intersection_points_pole2[index_2]
                break
    else:
        j = 1
    trimmed_midline = smoothed_skeleton[i:length - j]
    return extended_pole1, extended_pole2, trimmed_midline


def extend_skeleton(smoothed_skeleton, smoothed_contour,
                    find_pole1=True, find_pole2=True, interpolation_factor=1):
    # initiate approximated tip points
    new_pole1, new_pole2, smoothed_skeleton = find_poles(smoothed_skeleton,
                                                         smoothed_contour,
                                                         find_pole1=find_pole1,
                                                         find_pole2=find_pole2)
    extended_skeleton = np.concatenate([new_pole1,
                                        smoothed_skeleton,
                                        new_pole2])
    return spline_approximation(extended_skeleton,
                                n=int(interpolation_factor * len(smoothed_skeleton)),
                                smooth_factor=1, closed=False)


# old method, inefficient looping, deprecated
def find_midpoints(smoothed_skeleton, smoothed_contour, max_dxy=1):
    d_perp = unit_perpendicular_vector(smoothed_skeleton, closed=False)
    updated_skeleton = smoothed_skeleton.copy()
    for i in range(len(smoothed_skeleton)):
        xy = line_contour_intersection(smoothed_skeleton[i],
                                       d_perp[i] + smoothed_skeleton[i],
                                       smoothed_contour)
        coords = np.average(xy, axis=0)
        if (len(coords) == 2) and (np.isnan(coords).sum() == 0):
            if distance(smoothed_skeleton[i], coords) <= 2:
                updated_skeleton[i] = coords
    return updated_skeleton


def intersect_matrix(line, contour,
                     orthogonal_vectors=None):
    if orthogonal_vectors is None:
        dxy = unit_perpendicular_vector(line, closed=False)
    else:
        dxy = orthogonal_vectors
    v1, v2 = contour[:-1], contour[1:]
    x1, y1 = v1.T
    x2, y2 = v2.T
    x3, y3 = line.T
    perp_xy = line + dxy
    x4, y4 = perp_xy.T
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    A1B2 = A1[:, np.newaxis] * B2
    A1C2 = A1[:, np.newaxis] * C2
    B1A2 = B1[:, np.newaxis] * A2
    B1C2 = B1[:, np.newaxis] * C2
    C1A2 = C1[:, np.newaxis] * A2
    C1B2 = C1[:, np.newaxis] * B2

    intersect_x = (B1C2 - C1B2) / (B1A2 - A1B2)
    intersect_y = (A1C2 - C1A2) / (A1B2 - B1A2)
    return intersect_x, intersect_y


def distance_matrix(data1, data2):
    x1, y1 = data1.T
    x2, y2 = data2.T
    dx = x1[:, np.newaxis] - x2
    dy = y1[:, np.newaxis] - y2
    dxy = np.sqrt(dx ** 2 + dy ** 2)
    return dxy


def between_contour_points(p, contour):
    dist = distance_matrix(contour, p)
    return np.argmin(dist)


def divide_contour(p1, p2, contour):
    pos1 = between_contour_points(p1, contour)
    pos2 = between_contour_points(p2, contour)
    if pos1 == pos2:
        raise ValueError('Two endpoints are identical!')
    else:
        pos1, pos2 = sorted([pos1, pos2])
        half_contour_1 = contour[pos1:min(pos2 + 1, len(contour) - 1)]
        half_contour_2 = np.concatenate([contour[pos2:-1], contour[:pos1]])
    return half_contour_1, half_contour_2


def direct_intersect_points(skeleton, contour):
    v1, v2 = contour[:-1], contour[1:]
    skel_x, skel_y = skeleton.T
    intersect_x, intersect_y = intersect_matrix(skeleton, contour)
    dx_v1 = intersect_x - v1.T[0][:, np.newaxis]
    dx_v2 = intersect_x - v2.T[0][:, np.newaxis]
    dy_v1 = intersect_y - v1.T[1][:, np.newaxis]
    dy_v2 = intersect_y - v2.T[1][:, np.newaxis]
    dx = dx_v1 * dx_v2
    dy = dy_v1 * dy_v2
    dist_x = skel_x[np.newaxis, :] - intersect_x
    dist_y = skel_y[np.newaxis, :] - intersect_y

    non_boundry_points = np.where(np.logical_and(dy > 0, dx > 0))
    dist_matrix = np.sqrt(dist_x ** 2 + dist_y ** 2)
    dist_matrix[non_boundry_points] = np.inf
    nearest_id_x = np.argsort(dist_matrix, axis=0)[:2]
    nearest_id_y = np.linspace(0, dist_matrix.shape[1] - 1, dist_matrix.shape[1]).astype(int)

    pos1_list = np.array([intersect_x[nearest_id_x[0], nearest_id_y], intersect_y[nearest_id_x[0], nearest_id_y]]).T
    pos2_list = np.array([intersect_x[nearest_id_x[1], nearest_id_y], intersect_y[nearest_id_x[1], nearest_id_y]]).T
    midpoints = (pos1_list + pos2_list) / 2
    d_midpoints = np.abs(midpoints - skeleton)
    outliers = np.where(d_midpoints > 1)
    midpoints[outliers] = skeleton[outliers]
    return midpoints

def line_contour_intersection(p1, p2, contour):
    v1, v2 = contour[:-1], contour[1:]
    x1, y1 = v1.T
    x2, y2 = v2.T
    x3, y3 = p1
    x4, y4 = p2
    xy = np.array(line_intersect(x1, y1, x2, y2, x3, y3, x4, y4)).T
    dxy_v1 = xy - v1
    dxy_v2 = xy - v2
    dxy = dxy_v1 * dxy_v2
    intersection_points = xy[np.where(np.logical_and(dxy[:, 0] < 0, dxy[:, 1] < 0))]
    if len(intersection_points) > 2:
        dist = np.sum(np.square(np.tile(p1, (len(intersection_points), 1)) - intersection_points),
                      axis=1)
        intersection_points = intersection_points[np.argsort(dist)[0:2]]
    return intersection_points


@jit(nopython=True, cache=True)
def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):  # Ax+By = C
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3
    intersect_x = (B1 * C2 - B2 * C1) / (A2 * B1 - A1 * B2)
    intersect_y = (A1 * C2 - A2 * C1) / (B2 * A1 - B1 * A2)
    return intersect_x, intersect_y


@jit(nopython=True, cache=True)
def estimate_boundary_length(boundary_coords):
    max_L = len(boundary_coords)
    dist = 0
    for i in range(len(boundary_coords)):
        for j in range(i+1,len(boundary_coords)):
            x1,y1 = boundary_coords[i]
            x2,y2 = boundary_coords[j]
            dist =  np.sqrt((x2-x1)**2+(y2-y1)**2)
            if dist > max_L:
                max_L = dist
    return max_L


def touching_edge(img_shape,optimized_bbox):
    (rows, columns) = img_shape
    (x1, y1, x2, y2) = optimized_bbox
    if min(x1,y1,rows-x2-1, columns-y2-1) <= 5:
        return True
    else:
        return False


def bend_angle(data, window=10):
    p1 = np.concatenate((data[-window:],data[:-window])).T
    p2 = data.copy().T
    p3 = np.concatenate((data[window:],data[0:window])).T
    p1p2 = p1[0]*1+p1[1]*1j - (p2[0]*1+p2[1]*1j)
    p1p3 = p1[0]*1+p1[1]*1j - (p3[0]*1+p3[1]*1j)
    return np.angle(p1p3/p1p2, deg=True)


def bend_angle_open(data, window=5):
    p1 = data[:-2*window].T
    p2 = data[window:-window].T
    p3 = data[2*window:].T
    p1p2 = p1[0]*1+p1[1]*1j - (p2[0]*1+p2[1]*1j)
    p2p3 = p2[0]*1+p2[1]*1j - (p3[0]*1+p3[1]*1j)
    return np.angle(p2p3/p1p2, deg=True)


def standard_rod_complexity(area, L, window, perimeter, correction_factor=0.75):
    _r = np.sqrt(area/np.pi)
    circ_bending_energy_sum = window * 180
    if L > _r:
        r = estimate_r_from_area(area,L)
        l = L-2*r
        circ_perimeter = 2*np.pi*r/correction_factor
        rod_perimeter = 2*l/correction_factor
    else:
        circ_perimeter = perimeter
        rod_perimeter = 0
    return (circ_bending_energy_sum/(circ_perimeter + rod_perimeter))


def estimate_r_from_area(area,L):
    a = np.pi-4
    b = 2*L
    c = -area
    return (-b+np.sqrt(b**2-4*a*c))/(2*a)


def line_length(line):
    v1 = line[:-1]
    v2 = line[1:]
    d = v2-v1
    return np.sum(np.sqrt(np.sum(d**2,axis=1)))

def bilinear_interpolate_numpy(im, x, y):
    h,l = im.shape
    padded = np.zeros((h+1,l+1))
    padded[:h,:l] += im
    im = padded
    x0 = x.astype(int)
    x1 = x0 + 1
    y0 = y.astype(int)
    y1 = y0 + 1
    Ia = im[x0,y0]
    Ib = im[x0,y1]
    Ic = im[x1,y0]
    Id = im[x1,y1]
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    return np.round((Ia*wa) + (Ib*wb) + (Ic*wc) + (Id*wd), 4)


def expand_contour(contour, scale=0.5):
    """
    enlarge or shrink contour
    :param contour:
    :param scale:
    :return:
    """
    dxy = unit_perpendicular_vector(contour, closed=True)
    return contour - scale*dxy


def measure_along_strip(line, img, width = 7, subpixel = 0.5):
    warnings.filterwarnings("ignore")
    unit_dxy = unit_perpendicular_vector(line, closed=False)
    width_normalized_dxy = unit_dxy * subpixel
    copied_img = img.copy()
    data = bilinear_interpolate_numpy(copied_img, line.T[0], line.T[1])
    for i in range(1, 1+int(width * 0.5 / subpixel)):
        dxy = width_normalized_dxy * i
        v1 = line + dxy
        v2 = line - dxy
        p1 = bilinear_interpolate_numpy(copied_img, v1.T[0], v1.T[1])
        p2 = bilinear_interpolate_numpy(copied_img, v2.T[0], v2.T[1])
        data = np.vstack([p1, data, p2])
    return np.average(data, axis=0)


def straighten_cell(img, midline, width):

    # determine vertical boundaries by padding cell width with 1 pixel wide line at each side
    max_width = np.max(width)
    half_max_width = int(round(max_width*0.5+1))
    unit_dxy = unit_perpendicular_vector(midline, closed=False)
    data = bilinear_interpolate_numpy(img, midline.T[0], midline.T[1])
    copied_img = img.copy()
    # remove background if necessary
    for i in range(1, half_max_width):
        dxy = unit_dxy*i
        v1 = midline+dxy
        v2 = midline-dxy
        p1 = bilinear_interpolate_numpy(copied_img, v1.T[0], v1.T[1])
        p2 = bilinear_interpolate_numpy(copied_img, v2.T[0], v2.T[1])
        data = np.vstack([p1, data, p2])
    return data

def straighten_cell_normalize_width(img, midline, width,
                                    subpixel = 0.5,
                                    remove_cap = 0):

    #remove polar regions if necessary
    midline = spline_approximation(midline,
                                   int(len(midline)/subpixel),
                                   smooth_factor=1,
                                   closed=False)


    width = np.interp(np.linspace(0, 1, int(len(width)/subpixel)), xp=np.linspace(0, 1, len(width)), fp=width)
    # previous implementation removed the terminal two points from the profiling line,
    # causing inaccurate signal assignment near the polar regions
    # replacing with new function
    # decapped_midline = midline[remove_cap + 2:-remove_cap - 2]
    # decapped_width = width[remove_cap + 2:-remove_cap - 2]

    decapped_midline = midline[remove_cap: len(midline)-remove_cap].copy()
    decapped_width = width[remove_cap: len(midline)-remove_cap].copy()
    decapped_width[0] = decapped_width[1]
    decapped_width[-1] = decapped_width[-2]

    unit_dxy = subpixel*(unit_perpendicular_vector(midline, closed=False))[remove_cap: len(midline)-remove_cap]
    max_width = np.max(decapped_width)
    #normalize vertical step size by cell width
    normalization_factor = decapped_width / max_width
    width_normalized_dxy = unit_dxy * (np.vstack([normalization_factor, normalization_factor]).T)
    data = bilinear_interpolate_numpy(img, decapped_midline.T[0], decapped_midline.T[1])
    copied_img = img.copy()

    for i in range(1, int(round(0.5*max_width/subpixel)+1)):
        dxy = width_normalized_dxy * i
        v1 = decapped_midline + dxy
        v2 = decapped_midline - dxy
        p1 = bilinear_interpolate_numpy(copied_img, v1.T[0], v1.T[1])
        p2 = bilinear_interpolate_numpy(copied_img, v2.T[0], v2.T[1])
        data = np.vstack([p1, data, p2])
    return data


"""
inefficient local max finder, deprecated
def subpixel_localmax(input_image, mask,
                      forward_sigma=1, background_sigma=5,
                      height=10, radius=3, max_iteration=100,
                      step=0.001, tolerance=0.01):
    data = input_image.copy()
    # increase local contrast using gaussian filters of two different sigma.
    gaussian_filtered = filters.gaussian(data, sigma=forward_sigma) - filters.gaussian(data, sigma=background_sigma)
    gaussian_filtered[gaussian_filtered < 0] = 0

    # find pixelwise local maxima with local contrast constraint h.
    localmax = morphology.extrema.h_maxima(gaussian_filtered, h=height, selem=morphology.disk(radius))

    # find coordinates of local maxima within ROI
    init_points = np.array(np.where((localmax) * morphology.dilation(mask))).astype(np.float)
    evolved_points = init_points.copy()

    # interpolate signal gradient.
    g_x, g_y = np.gradient(data)[0], np.gradient(data)[1]
    fg_x = interpolate_2Dmesh(data_array=g_x, smooth=0)
    fg_y = interpolate_2Dmesh(data_array=g_y, smooth=0)

    # iterative evolution of local maxima
    converged_list = np.zeros(len(init_points[0]))
    converged = False
    while max_iteration > 0:
        for i in range(len(init_points[0])):
            if converged_list[i] == 0:
                x, y = float(evolved_points[0][i]), float(evolved_points[1][i])
                dx = (fg_x(x, y) * step)[0][0]
                dy = (fg_y(x, y) * step)[0][0]
                if min(abs(dx), abs(dy)) >= tolerance:
                    evolved_points[0][i] = x + dx
                    evolved_points[1][i] = y + dy
                else:
                    converged_list[i] = 1
        max_iteration -= 1
        if converged_list.sum() == len(init_points[0]):
            converged = True
            break
    return init_points, evolved_points, converged


def pixel_blob_log(input_image, mask,
                   forward_sigma=1, background_sigma=5,
                   min_sigma=0.5, max_sigma=1.2, threshold=100):

    data = input_image.copy()
    # increase local contrast using gaussian filters of two different sigma.
    gaussian_filtered = (filters.gaussian(data, sigma=forward_sigma) - filters.gaussian(data, sigma=background_sigma))
    gaussian_filtered *= mask
    gaussian_filtered[gaussian_filtered < 0] = 0
    # find blobs using LoG method
    blobs = feature.blob_log(gaussian_filtered, min_sigma=min_sigma, max_sigma=max_sigma, overlap=0.8, threshold=threshold)

    return blobs
"""


def local_puncta_zscore(image, coords, trim_low_bound=5):

    x, y = coords
    data = np.sort((image[x - 4:x + 5, y - 4:y + 5].copy()).flatten())[trim_low_bound:]

    threshold = filters.threshold_isodata(data, nbins=64)
    threshold = 0.5 * (threshold + data[-4:].mean())

    fg = data[data > threshold]
    bg = data[data <= threshold]

    if len(fg) <= 5:
        z = 0
    else:
        z = (np.mean(fg) - np.mean(bg)) / bg.std()
    return z


def subpixel_approximation_quadratic(x, y, energy_map, max_iteration=2):

    local_maxima_found = False
    for i in range(max_iteration):
        cube = energy_map[x - 1:x + 2, y - 1:y + 2]
        if cube[1, 1] == cube.max():
            local_maxima_found = True
            vx0, vx1, vx2 = cube[0, 1], cube[1, 1], cube[2, 1]
            vy0, vy1, vy2 = cube[1, 0], cube[1, 1], cube[1, 2]
            x += quadratic_maxima_approximation(vx0, vx1, vx2)
            y += quadratic_maxima_approximation(vy0, vy1, vy2)
            break
        else:
            dx, dy = np.array(np.unravel_index(cube.argmax(), cube.shape)) - 1
            x += dx
            y += dy
    return x, y, local_maxima_found


def find_puncta(img, mask):

    img_smoothed = filters.gaussian(img, sigma=1)
    img_smoothed = (img_smoothed - img_smoothed.min()) / (img_smoothed.max() - img_smoothed.min())
    LoG = ndi.gaussian_laplace(img_smoothed, sigma=1)
    extrema = feature.peak_local_max(-LoG * mask, min_distance=2, threshold_rel=0.3)
    output = []

    for ex in extrema:
        x, y = ex
        newx, newy, stable_maxima = subpixel_approximation_quadratic(x, y, img_smoothed)
        if stable_maxima:
            z_image = local_puncta_zscore(img_smoothed, ex)
            z_LoG = local_puncta_zscore(LoG, ex)
            output.append([newx, newy, img[x, y], LoG[x, y], z_image, z_LoG])
    return np.array(output)


def moving_window_average(data, window_size=5):
    if len(data) <= window_size+2:
        raise ValueError("Input array does not have enough values")
    else:
        cum_sum = np.cumsum(np.insert(data, 0, 0))
        mw_average = (cum_sum[window_size:]-cum_sum[:-window_size])/float(window_size)
        return np.interp(np.linspace(0, 1, len(data)),
                         xp=np.linspace(0, 1, len(mw_average)),
                         fp=mw_average)


def normalize_data1D(data, re_orient=False, base=10):
    half_l = int(0.5 * len(data))
    if data[:half_l].mean() < data[-half_l:].mean():
        if re_orient:
            data = np.flip(data)
    normalized_data = (data-data.min()+base)/(data.max()-data.min()+base)
    return normalized_data


def normalize_data_2D(data, re_orient=True, percentile_low_bound=1):
    half_l = int(0.5 * data.shape[1])
    if data[:, :half_l].mean() < data[:, -half_l:].mean():
        if re_orient:
            data = np.flip(data, axis=1)

    th_low = np.percentile(data, percentile_low_bound)

    normalized_data = (data - th_low) / (data.max() - th_low)
    normalized_data[normalized_data > 1] = 1
    normalized_data[normalized_data < 0] = 0
    return normalized_data


def pad_data(array, length, max_len=15, max_pixel=512, normalize=True, base=50):
    pixelated_length = min(int(round(length*max_pixel*0.5/max_len))*2, max_pixel)
    pad_length = max(int((max_pixel-pixelated_length)/2), 0)
    interpolated = np.interp(np.linspace(0, 1, pixelated_length),
                             np.linspace(0, 1, len(array)), array)
    if normalize:
        interpolated = (interpolated-interpolated.min()+base)/(interpolated.max()-interpolated.min()+base)
    padded = np.pad(interpolated, (pad_length, pad_length), 'constant', constant_values=(0, 0))
    return padded

def scanning_disk():
    cube = np.ones((3,3))
    x, y = np.where(cube > 0)
    x -= 1
    y -= 1
    weights = np.array([0.5, 1, 0.5,
                        1, 1, 1,
                        0.5, 1, 0.5])
    return x, y, weights


def measure_along_contour(cell, channel):
    contour = cell.optimized_contour[:-1]
    data = filters.gaussian(cell.data[channel], sigma=1,
                            preserve_range=True)
    x, y, weights = scanning_disk()
    _x = x[:, np.newaxis]+contour[:, 0]
    _y = y[:, np.newaxis]+contour[:, 1]
    interpolated = bilinear_interpolate_numpy(data, _x, _y)
    interpolated *= weights[:, np.newaxis]
    return np.sum(interpolated, axis=0)/7
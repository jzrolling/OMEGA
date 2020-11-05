__author__ = "jz-rolling"

from .helper import *
from .cell import Cell
from . import boundary
import skimage as sk
import numpy as np


class Cluster:

    def __init__(self, image = None, colony_regionprop = None):
        self.shape_indexed = None
        self.labeled_seeds = None
        self.particle_counts = 0
        self.watersheded = None
        self.particles = {}
        self.boundary_mask = None
        self.particle_pairwise_dict = {}
        self.boundary_pairwise_dict = {}
        self.merge_particle_info = None
        self.cells = {}
        self.discarded = False
        self.touching_edge =False

        if (image != None) and (colony_regionprop != None):
            self.bbox = optimize_bbox(image.shape, colony_regionprop.bbox)
            (x1, y1, x2, y2) = self.bbox
            self.label = colony_regionprop.label
            self.phase = image.data[image.mask_channel_name][x1:x2, y1:y2].copy()
            self.sobel = image.mask_sobel[x1:x2, y1:y2].copy()
            self.mask_clumpy_zone = image.mask_clumpy_zone[x1:x2, y1:y2].copy()
            self.pixel_microns = image.pixel_microns
            self.config = image.config['cluster']
            self.global_mask_background_median = image.image_background_median[image.mask_channel_name]
            self.global_mask_foreground_median = image.image_foreground_median[image.mask_channel_name]
            self.global_mask_background_std = image.image_background_std[image.mask_channel_name]
            self.global_mask_foreground_std = image.image_foreground_std[image.mask_channel_name]
            self.touching_edge = touching_edge(image.shape, self.bbox)

    def create_seeds(self, image, apply_median_filter=True):

        low_bound = int(self.config['shape_index_low_bound'])
        high_bound = int(self.config['shape_index_high_bound'])
        min_seed_size = int(self.config['seed_size_min'])
        radius = int(self.config['delink_radius'])
        x1, y1, x2, y2 = self.bbox
        self.mask = (image.mask_labeled_clusters[x1:x2, y1:y2] == self.label) * 1

        if apply_median_filter:
            # use smoothed shape-indexed image
            self.shape_indexed = image.mask_shape_indexed_smoothed[x1:x2, y1:y2].copy() * self.mask
        else:
            self.shape_indexed = image.mask_shape_indexed[x1:x2, y1:y2].copy() * self.mask

        # generate watersheded seeds, remove the ones that are too small
        init_seeds = (self.shape_indexed < high_bound) & (self.shape_indexed > low_bound)
        init_seeds = morphology.remove_small_holes(init_seeds, area_threshold=30) * 1
        init_seeds = morphology.opening(init_seeds, sk.morphology.disk(radius)).astype(bool)
        init_seeds = morphology.remove_small_objects(init_seeds, min_size=min_seed_size)

        # label particles
        self.labeled_seeds = sk.measure.label(init_seeds, connectivity=1)

    def recreate_seeds_from_branched_particle(self, cell):
        self.shape_indexed = cell.shape_index
        self.label = cell.cluster_label
        self.bbox = cell.bbox
        self.phase = cell.data[cell.mask_channel_name]
        self.sobel = cell.sobel
        self.mask_clumpy_zone = np.zeros(cell.shape_index.shape)
        self.pixel_microns = cell.pixel_microns
        self.config = cell.config['cluster']
        self.mask = cell.mask * 1
        radius = int(self.config['delink_radius'])

        low_th = int(self.config['shape_index_branch_threshold_low_bound'])
        high_th = int(self.config['shape_index_branch_threshold_high_bound'])

        min_seed_size = int(self.config['seed_size_min'])
        init_seeds = (self.shape_indexed < high_th) & (self.shape_indexed > low_th)
        init_seeds = morphology.remove_small_holes(init_seeds, area_threshold=30) * 1
        init_seeds = morphology.opening(init_seeds, sk.morphology.disk(radius)).astype(bool)
        init_seeds = morphology.remove_small_objects(init_seeds, min_size=min_seed_size)
        self.labeled_seeds = measure.label(init_seeds, connectivity=1)
        del init_seeds


    def recreate_seeds_from_vsnap(self, cell):
        self.shape_indexed = cell.shape_index
        self.label = cell.cluster_label
        self.bbox = cell.bbox
        self.phase = cell.data[cell.mask_channel_name]
        self.sobel = cell.sobel
        self.mask_clumpy_zone = np.zeros(cell.shape_index.shape)
        self.pixel_microns = cell.pixel_microns
        self.config = cell.config['cluster']
        self.mask = cell.mask * 1

        dist_transform = ndi.distance_transform_edt(self.mask)
        threshold = np.percentile(dist_transform[dist_transform > 0], 85)
        init_seeds = dist_transform > threshold
        init_seeds = morphology.remove_small_objects(init_seeds, min_size=10)
        self.labeled_seeds = measure.label(init_seeds, connectivity=2)
        del init_seeds


    def segmentation(self):
        self.particle_counts = self.labeled_seeds.max() # 10+ times faster than computing np.unique()
        if self.particle_counts >= 1:
            #remove subimages whose seeds that are too small

            if self.particle_counts == 1:
                #single particl subimage
                if not self.touching_edge:
                    #remove particles that are too close to the edge
                    self.watersheded = self.mask
                    self.particles[1] = measure.regionprops(self.mask)[0]
                else:
                    self.discarded = True
            else:
                _clump_vs_cluster = self.mask_clumpy_zone[np.where(self.mask>0)]
                if len(np.nonzero(_clump_vs_cluster)[0])/len(_clump_vs_cluster) > 0.9:
                    self.discarded = True
                else:
                    self.watersheded = segmentation.watershed(self.phase, self.labeled_seeds,
                                                              mask=self.mask,
                                                              connectivity=1,
                                                              compactness=0,
                                                              watershed_line=True)

                    particle_info = measure.regionprops(self.watersheded, intensity_image=self.phase)
                    for part in particle_info:
                        self.particles[part.label] = part

                    # record cell boundraies
                    self.boundary_mask = ((self.mask > 0) * 1) - ((self.watersheded > 0) * 1)
                    d1,d2 = boundary.boundary_neighbor_pairwise(self.watersheded, self.boundary_mask)
                    self.particle_pairwise_dict = d1
                    self.boundary_pairwise_dict = d2
                    del d1,d2
        else:
            self.discarded = True

    def compute_boundary_metrics(self):
        for label, boundary in self.boundary_pairwise_dict.items():
            boundary.primary_boundary_metrics(self)

    def remove_false_boundary(self, default=True):
        paired_list = []
        boundary_id_list = []
        for boundary_id, boundary in self.boundary_pairwise_dict.items():
            if default:
                boundary.boundary_classification_default(self)
            if boundary.false_boundary:
                p1, p2 = boundary.p1, boundary.p2
                paired_list.append([p1,p2])
                boundary_id_list.append(boundary_id)

        self.merge_particle_info = merge_joint(paired_list, boundary_id_list)
        particle_list, boundary_list = self.merge_particle_info.newdata,self.merge_particle_info.new_idxlist

        if len(particle_list) != len(boundary_list):
            raise ValueError('Boundary ID allocation failed!')
        else:
            for i in range(len(particle_list)):
                group = sorted(particle_list[i])
                newlabel = int(group[0])
                for label in group[1:]:
                    self.watersheded[self.watersheded==label] = newlabel
                    del self.particles[label]
                for boundary_id in boundary_list[i]:
                    xcoords, ycoords = np.array(self.boundary_pairwise_dict[boundary_id].boundary_coordinates).T
                    self.watersheded[xcoords,ycoords] = newlabel
                self.particles[newlabel] = measure.regionprops((self.watersheded==newlabel).astype(np.int),
                                                                intensity_image=self.phase)[0]

    def filter_particles(self, image):
        min_area = int(self.config['area_pixel_low'])
        for label, particle in self.particles.items():
            _clump_area = self.mask_clumpy_zone[np.where(self.watersheded == label)]
            if (len(np.nonzero(_clump_area))/len(_clump_area) <= 0.01) & (particle.area >= 20):
                x0, y0, _x0, _y0 = self.bbox
                x3, y3, x4, y4 = optimize_bbox(self.phase.shape, particle.bbox)
                optimized_bbox = [x3 + x0, y3 + y0, x4 + x0, y4 + y0]
                if not touching_edge(image.shape, optimized_bbox):
                    if particle.area > min_area:
                        optimized_mask = self.watersheded[x3:x4, y3:y4] == label
                        cell = Cell(image, self.label, str(label),
                                    optimized_mask, optimized_bbox, regionprop=particle)
                        cell.find_contour()
                        cell.extract_skeleton()
                        self.cells[label] = cell

    def split_branches(self, image):
        _old_labels_2b_removed = []
        sub_dict = {}

        if not self.cells == {}:
            max_label = max(list(self.cells.keys()))+1
            for label, cell in self.cells.items():
                if cell.branched and not cell.discarded:
                    mini_cluster = Cluster(image=image, colony_regionprop=cell.regionprop)
                    if cell.vshaped:
                        mini_cluster.recreate_seeds_from_vsnap(cell)
                    else:
                        mini_cluster.recreate_seeds_from_branched_particle(cell)
                    mini_cluster.segmentation()
                    mini_cluster.compute_boundary_metrics()
                    mini_cluster.remove_false_boundary()
                    for minilabel, miniparticle in mini_cluster.particles.items():
                        x0, y0, _x0, _y0 = mini_cluster.bbox
                        x3, y3, x4, y4 = optimize_bbox(mini_cluster.phase.shape, miniparticle.bbox)
                        optimized_bbox = [x3 + x0, y3 + y0, x4 + x0, y4 + y0]
                        optimized_mask = mini_cluster.watersheded[x3:x4, y3:y4] == minilabel
                        if miniparticle.area >= 20:
                            newcell = Cell(image, self.label, max_label, optimized_mask,
                                           optimized_bbox, regionprop=miniparticle)
                            newcell.find_contour()
                            newcell.extract_skeleton()
                            sub_dict[max_label] = newcell
                            max_label += 1
                            if label not in _old_labels_2b_removed:
                                _old_labels_2b_removed.append(label)
            for key in _old_labels_2b_removed:
                del self.cells[key]
            self.cells.update(sub_dict)

    def measure_cells(self,
                      terminal_shapeindex=False,
                      threshold=45,
                      reject_low_quality_cells=False):
        for label, cell in self.cells.items():
            if not cell.branched and not cell.discarded:
                cell.compiled_cell_process()
                if not cell.branched and not cell.discarded:
                    if terminal_shapeindex:
                        shpid = cell.measurements.signal_measurements['Shape_index']['axial']
                        shpid = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(shpid)), shpid)
                        below_th = np.where(shpid < threshold)[0]
                        if len(below_th) > 2:
                            if below_th[0] > 10 and below_th[-1] < 90:
                                cell.segmentation_quality_by_shapeindex = 0
                                if reject_low_quality_cells:
                                    cell.discarded = True
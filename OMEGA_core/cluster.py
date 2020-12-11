__author__ = "jz-rolling"

from .helper import *
from .cell import Cell
from . import boundary
import skimage as sk
import numpy as np
from sklearn import svm,linear_model,tree
import pickle as pk


class Cluster:
    """
    A cluster is a continuous patch of the binary mask that contains one to many cell-like particles
    """
    def __init__(self, image_obj=None,
                 cluster_regionprop=None):
        """
        :param image_obj: corresponding master Image object
        :param cluster_regionprop: Skimage regionprop object of the corresponding Cluster

        Attributes
        ==========

        mask
        ---------- binary mask of the Cluster

        shape_indexed
        ---------- clip of shape index image

        labeled_seeds
        ---------- particle seeds labled by unique integers (1->N)

        particle_counts
        ---------- number of particles found in Cluster

        watershed
        ---------- watershed segmentation output

        particles
        ---------- dictionary of found particles

        boundary_mask
        ---------- binary image of inter-particle boundaries

        particle_pairwise_dict & boundary_pairwise_dict
        ---------- dictionaries recording each directly contacting particles and the assigned boundaries
                   For example: a particle_pairwise_dict entry may look like: {a : {b : c}}, here a,b,c are all
                   integers that denote the numeric labels of particle A, particle B and the boundary C between
                   A and B. A boundary_pairwise_dict entry may look like {c:Boundary_obj} where boundary C is
                   represented by a Boundary class.

        merge_particle_info
        ---------- conflated particle-boundary information

        cells
        ---------- dictionary of Cell object(s)

        discarded
        ---------- Cluster omitted from analysis if True

        touching_edge
        ---------- Cluster contains edge pixels if True

        bbox
        ---------- coordinates of the bounding box corners by the format of (x1, y1, x2, y2)
                   note that the bounding box is based on an inflated ROI
                   rather than the original ROI (regionprops.bbox)

        label
        ---------- integer label of the Cluster

        phase, sobel, mask_clumpy_zone
        ---------- clips of corresponding image data inherited from master Image object

        pixel_microns
        ---------- unit pixel length

        config
        ---------- Cluster configurations

        """
        self.mask = None
        self.shape_indexed = None
        self.labeled_seeds = None
        self.particle_counts = 0
        self.watershed = None
        self.particles = {}
        self.boundary_mask = None
        self.particle_pairwise_dict = {}
        self.boundary_pairwise_dict = {}
        self.merge_particle_info = None
        self.cells = {}
        self.discarded = False
        self.touching_edge = False
        self.false_boundary_neighbors = []
        self.false_boundary_id = []

        if (image_obj is not None) and (cluster_regionprop is not None):
            self.bbox = optimize_bbox(image_obj.shape, cluster_regionprop.bbox)
            (x1, y1, x2, y2) = self.bbox
            self.label = cluster_regionprop.label
            self.phase = image_obj.data[image_obj.mask_channel_name][x1:x2, y1:y2].copy()
            self.sobel = image_obj.mask_sobel[x1:x2, y1:y2].copy()
            self.frangi = image_obj.mask_frangi[x1:x2, y1:y2].copy()
            self.mask_clumpy_zone = image_obj.mask_clumpy_zone[x1:x2, y1:y2].copy()
            self.pixel_microns = image_obj.pixel_microns
            self.DoG = image_obj.mask_DoG[x1:x2, y1:y2].copy()
            self.RoG = image_obj.mask_RoG[x1:x2, y1:y2].copy()
            self.config = image_obj.config['cluster']
            """
            unused stats, might be integrated in the future
            """

            self.global_mask_background_median = image_obj.image_background_median[image_obj.mask_channel_name]
            self.global_mask_foreground_median = image_obj.image_foreground_median[image_obj.mask_channel_name]
            self.global_mask_background_std = image_obj.image_background_std[image_obj.mask_channel_name]
            self.global_mask_foreground_std = image_obj.image_foreground_std[image_obj.mask_channel_name]

            self.touching_edge = touching_edge(image_obj.shape, self.bbox)

    def create_seeds(self,
                     image_obj,
                     apply_median_filter=False):

        """
        This function infers core body parts of cells from the shape-index surface curvature profiles.
        The inferred body parts are used as seeds for watershed segmentation, the segmented particles are subsequently
        subdivided or merged by a collection of criterion.

        :param image_obj: master Image object
        :param apply_median_filter: suppresses local curvature extremities using a median kernel if set to True

        """

        """
        Import configurations
        """
        shape_index_low_bound = int(self.config['shape_index_low_bound'])
        shape_index_high_bound = int(self.config['shape_index_high_bound'])
        frangi_low_bound = float(self.config['frangi_low_bound'])
        frangi_high_bound = float(self.config['frangi_high_bound'])
        min_seed_size = int(self.config['seed_size_min'])
        radius = int(self.config['delink_radius'])
        use_frangi = bool(int(self.config['use_frangi']))

        """
        Inherit the binary mask from the master Image object
        """
        x1, y1, x2, y2 = self.bbox
        self.mask = (image_obj.mask_labeled_clusters[x1:x2, y1:y2] == self.label) * 1

        if apply_median_filter:
            # use smoothed shape-indexed image
            self.shape_indexed = image_obj.mask_shape_indexed_smoothed[x1:x2, y1:y2].copy() * self.mask
        else:
            self.shape_indexed = image_obj.mask_shape_indexed[x1:x2, y1:y2].copy() * self.mask

        """
        generate watershed seeds, remove the ones that are too small
        """
        if use_frangi:
            frangi = self.frangi*self.mask
            init_seeds = (frangi < frangi_high_bound) & (frangi > frangi_low_bound)
        else:
            init_seeds = (self.shape_indexed < shape_index_high_bound) & \
                         (self.shape_indexed > shape_index_low_bound)
        init_seeds = morphology.remove_small_holes(init_seeds, area_threshold=30) * 1
        init_seeds = morphology.opening(init_seeds, morphology.disk(radius)).astype(bool)
        init_seeds = morphology.remove_small_objects(init_seeds, min_size=min_seed_size)

        # label particles
        self.labeled_seeds = sk.measure.label(init_seeds, connectivity=1)

    def create_seeds_from_branched_particle(self, cell):
        """
        This function forces the segmentation of particles of complex shapes (eg. branched) using more stringent
        segmentation criterion.

        Note that the current solution performs suboptimally and often yields inaccurate segmentation
        Future adjustments with supervised SVM or NN are needed to reach higher segmentation accuracy and to
        account for cells with deformed morphology.

        :param cell: Cell object to be subdivided (re-imported as a Cluster object)
        :return:
        """
        self.shape_indexed = cell.shape_index
        self.label = cell.cluster_label
        self.bbox = cell.bbox
        self.phase = cell.data[cell.mask_channel_name]
        self.sobel = cell.sobel
        self.mask_clumpy_zone = np.zeros(cell.shape_index.shape)
        self.pixel_microns = cell.pixel_microns
        self.config = cell.config['cluster']
        self.mask = cell.mask * 1
        self.frangi = cell.frangi

        """
        import configurations for Cel subdivision.
        """
        use_frangi = bool(int(self.config['use_frangi_split_branch']))
        radius = int(self.config['delink_radius'])

        if not use_frangi:
            ref_image = self.shape_indexed
            low_th = int(self.config['shape_index_branch_threshold_low_bound'])
            high_th = int(self.config['shape_index_branch_threshold_high_bound'])
        else:
            ref_image = self.frangi
            low_th = float(self.config['frangi_branched_low_bound'])
            high_th = float(self.config['frangi_branched_high_bound'])
        min_seed_size = int(self.config['seed_size_min'])

        init_seeds = (ref_image < high_th) & (ref_image > low_th)
        init_seeds = morphology.remove_small_holes(init_seeds, area_threshold=30) * 1
        init_seeds = morphology.opening(init_seeds, morphology.disk(radius)).astype(bool)
        init_seeds = morphology.remove_small_objects(init_seeds, min_size=min_seed_size)
        self.labeled_seeds = measure.label(init_seeds, connectivity=1)
        del init_seeds

    def create_seeds_from_vsnap(self, cell):
        """
        This function forces the segmentation of V-shaped particles

        Note that the current solution performs suboptimally and often yields inaccurate segmentation
        Future adjustments with supervised SVM or NN are needed to reach higher segmentation accuracy and to
        account for cells with deformed morphology.

        :param cell: Cell object to be subdivided (re-imported as a Cluster object)
        :return:
        """
        self.shape_indexed = cell.shape_index
        self.label = cell.cluster_label
        self.bbox = cell.bbox
        self.phase = cell.data[cell.mask_channel_name]
        self.sobel = cell.sobel
        self.mask_clumpy_zone = np.zeros(cell.shape_index.shape)
        self.pixel_microns = cell.pixel_microns
        self.config = cell.config['cluster']
        self.mask = cell.mask * 1
        self.frangi = cell.frangi

        """
        import configurations for Cel subdivision.
        """
        use_frangi = bool(int(self.config['use_frangi_split_branch']))
        radius = int(self.config['delink_radius'])

        if not use_frangi:
            ref_image = self.shape_indexed
            low_th = int(self.config['shape_index_vshape_low_bound'])
            high_th = int(self.config['shape_index_vshape_high_bound'])
        else:
            ref_image = self.frangi
            low_th = float(self.config['frangi_branched_low_bound'])
            high_th = float(self.config['frangi_branched_high_bound'])
        min_seed_size = int(self.config['seed_size_min'])

        init_seeds = (ref_image < high_th) & (ref_image > low_th)
        init_seeds = morphology.remove_small_holes(init_seeds, area_threshold=30) * 1
        init_seeds = morphology.opening(init_seeds, morphology.disk(radius)).astype(bool)
        init_seeds = morphology.remove_small_objects(init_seeds, min_size=min_seed_size)
        self.labeled_seeds = measure.label(init_seeds, connectivity=1)
        del init_seeds

    def segmentation(self):
        """
        Cluster segmentation function.

        Given a set of inferred seed(s) (self.labeled_seeds), this function uses the watershed algorithm to
        conduct an exhaustive segmentation on the binary mask of a Cluster. The segmented Cluster may yield one
        (directly inherited by a Cell object) or multiple particles. In the latter case, Boundary objects are
        extracted from the labeled segments and are vetted for their authenticity. False boundaries are removed and
        their corresponding segments are conflated.

        """
        self.particle_counts = self.labeled_seeds.max() # 10+ times faster than computing np.unique()

        if self.particle_counts >= 1:

            if self.particle_counts == 1:
                # single segment
                if not self.touching_edge:
                    # remove particles that are too close to the edge
                    self.watershed = self.mask
                    self.particles[1] = measure.regionprops(self.mask)[0]
                else:
                    self.discarded = True

            else:
                # multple segments
                # discard segments that are close to a "clumpy" zone (by DoG)
                _clump_vs_cluster = self.mask_clumpy_zone[np.where(self.mask>0)]
                if len(np.nonzero(_clump_vs_cluster)[0])/len(_clump_vs_cluster) > 0.9:
                    self.discarded = True
                else:
                    # watershed segmentation
                    self.watershed = segmentation.watershed(self.phase, self.labeled_seeds,
                                                            mask=self.mask,
                                                            connectivity=1,
                                                            compactness=0,
                                                            watershed_line=True)
                    particle_info = measure.regionprops(self.watershed, intensity_image=self.phase)
                    for part in particle_info:
                        self.particles[part.label] = part

                    # record cell boundaries
                    self.boundary_mask = ((self.mask > 0) * 1) - ((self.watershed > 0) * 1)
                    d1, d2 = boundary.boundary_neighbor_pairwise(self.watershed, self.boundary_mask)
                    self.particle_pairwise_dict = d1
                    self.boundary_pairwise_dict = d2
                    del d1, d2

        else:
            self.discarded = True

    def compute_boundary_metrics(self, image_obj):
        """
        Analyze extracted boundaries.
        """
        for label, boundary in self.boundary_pairwise_dict.items():
            boundary.primary_boundary_metrics(self)
            image_obj.boundary_stats.append(list(boundary.metrics.values()))

    def boundary_classification(self,
                                counter,
                                boundary_prediction):
        """
        Remove low-confidence cell-cell boundaries. Merge neighboring segments/particles
        :param method:
        :return:
        """

        self.false_boundary_neighbors = []
        self.false_boundary_id = []
        for boundary_id, boundary in self.boundary_pairwise_dict.items():
            if boundary_prediction[counter]==1:
                p1, p2 = boundary.p1, boundary.p2
                self.false_boundary_neighbors.append([p1, p2])
                self.false_boundary_id.append(boundary_id)
            counter += 1
        return counter

    def remove_false_boundaries(self):
        self.merge_particle_info = merge_joint(self.false_boundary_neighbors,
                                               self.false_boundary_id)
        particle_list, boundary_list = self.merge_particle_info.newdata,\
                                       self.merge_particle_info.new_idxlist

        if len(particle_list) != len(boundary_list):
            raise ValueError('Boundary ID allocation failed!')
        else:
            for i in range(len(particle_list)):
                group = sorted(particle_list[i])
                newlabel = int(group[0])
                for label in group[1:]:
                    self.watershed[self.watershed==label] = newlabel
                    del self.particles[label]
                for boundary_id in boundary_list[i]:
                    xcoords, ycoords = np.array(self.boundary_pairwise_dict[boundary_id].boundary_coordinates).T
                    self.watershed[xcoords, ycoords] = newlabel
                self.particles[newlabel] = measure.regionprops((self.watershed == newlabel).astype(np.int),
                                                               intensity_image=self.phase)[0]

    def filter_particles(self, image_obj):
        """
        primary filtering function that removes particles that are too small and/or found in the "clumpy" zone.
        :param image_obj: master Image object
        """

        """
        minimal particle size filter
        """
        min_area = int(self.config['area_pixel_low'])
        max_area = int(self.config['area_pixel_high'])
        max_width = float(self.config['width_high'])/self.pixel_microns

        for label, particle in self.particles.items():
            _clump_area = self.mask_clumpy_zone[np.where(self.watershed == label)]

            # some basic discriminators
            in_clump = (len(np.nonzero(_clump_area))/len(_clump_area) > 0.01)*1
            too_small = (particle.area < min_area)*1
            too_large = (particle.area > max_area)*1
            too_bulky = ((particle.solidity > 0.75) & \
                         (particle.minor_axis_length > 3*max_width))*1
            # modification needed
            if np.sum([in_clump, too_small,
                       too_large, too_bulky]) == 0:
                x0, y0, _x0, _y0 = self.bbox
                x3, y3, x4, y4 = optimize_bbox(self.phase.shape, particle.bbox)
                optimized_bbox = [x3 + x0, y3 + y0, x4 + x0, y4 + y0]
                if not touching_edge(image_obj.shape, optimized_bbox):
                    optimized_mask = self.watershed[x3:x4, y3:y4] == label
                    # make a cell object
                    cell = Cell(image_obj, self.label, str(label),
                                optimized_mask, optimized_bbox, regionprop=particle)
                    cell.find_contour()
                    cell.extract_skeleton()
                    self.cells[label] = cell

    def split_branches(self, image_obj):

        """
        This function forces the segmentation of Cell objects of complex shapes
        and iteratively updates the Cluster.cells dictionary.
        :param image_obj:
        :return:
        """

        # temporary label list and cells dictionary
        _old_labels_2b_removed = []
        sub_dict = {}
        min_area = int(self.config['area_pixel_low'])
        if not self.cells == {}:
            max_label = max(list(self.cells.keys()))+1
            for label, cell in self.cells.items():

                # forced segmentation of branched object
                if cell.branched and not cell.discarded:

                    # directly transfer data from the branched Cell object to make a Cluster
                    mini_cluster = Cluster(image_obj=image_obj,
                                           cluster_regionprop=cell.regionprop)

                    # subdivide cell

                    if cell.vshaped:
                        mini_cluster.create_seeds_from_vsnap(cell)
                    else:
                        mini_cluster.create_seeds_from_branched_particle(cell)

                    # Standard cluster segmentation and boundary optimization
                    mini_cluster.segmentation()
                    if len(self.boundary_pairwise_dict)> 0:
                        for boundary_id, boundary in self.boundary_pairwise_dict.items():
                            if len(boundary.metrics.values()) > 0:
                                metrics = np.array(list(boundary.metrics.values())).reshape(1, -1)
                                normalized_metrics = image_obj.boundary_normalizer.transform(metrics)
                                false_boundary = image_obj.boundary_prediction_model.predict(normalized_metrics)[0]
                                boundary.false_boundary = bool(false_boundary)
                    mini_cluster.remove_false_boundaries()

                    # append updated Cell objects to the Cluster.cells dictionary
                    for minilabel, miniparticle in mini_cluster.particles.items():
                        x0, y0, _x0, _y0 = mini_cluster.bbox
                        x3, y3, x4, y4 = optimize_bbox(mini_cluster.phase.shape, miniparticle.bbox)
                        optimized_bbox = [x3 + x0, y3 + y0, x4 + x0, y4 + y0]
                        optimized_mask = mini_cluster.watershed[x3:x4, y3:y4] == minilabel
                        if miniparticle.area >= min_area:
                            newcell = Cell(image_obj, self.label, max_label,
                                           optimized_mask,
                                           optimized_bbox,
                                           regionprop=miniparticle)
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
                      polar_shapeindex=False,
                      reject_low_quality_cells=False):
        """
        Multi-metrics quantification of cellular data
        discard low-confidence Cell objects by polar shapeindex measure when specified

        :param polar_shapeindex: record low-confidence Cell objects when True
        :param reject_low_quality_cells: reject low-confidence Cell objects when True
        :return:
        """

        threshold = int(self.config['polar_shape_index_threshold'])
        for label, cell in self.cells.items():
            if not cell.branched and not cell.discarded:
                # Multi-metrics quantification
                cell.compiled_cell_process()

                # note that the Cell.compiled_cell_process may alter the Cell.branched
                # or the Cell.discarded status
                if not cell.branched and not cell.discarded:
                    if polar_shapeindex:
                        # calculates polar shape_index scores
                        # Cells with both poles correctly located should have a pole-proximate
                        # shapeindex lower than the specified threshold (strongly curved)
                        axial_shapeindex = cell.measurements.signal_measurements['Shape_index']['axial']
                        axial_shapeindex = np.interp(np.linspace(0, 1, 100),
                                                     np.linspace(0, 1, len(axial_shapeindex)),
                                                     axial_shapeindex)
                        below_th = np.where(axial_shapeindex < threshold)[0]
                        if len(below_th) >= 2:
                            if below_th[0] > 10 and below_th[-1] < 90:
                                cell.segmentation_quality_by_shapeindex = 0
                                if reject_low_quality_cells:
                                    cell.discarded = True
                        else:
                            cell.segmentation_quality_by_shapeindex = 0
                            if reject_low_quality_cells:
                                cell.discarded = True
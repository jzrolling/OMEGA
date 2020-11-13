from .helper import *


class Boundary:

    def __init__(self, p1, p2, b_coords):
        self.p1 = p1
        self.p2 = p2
        self.boundary_coordinates = b_coords
        self.false_boundary = False
        self.relative_distance = 0.0
        self.deflection_angle = 1.0
        self.center_line_angle = 1.0
        self.bbox = []
        self.length = 0
        self.max_perimeter_fraction = 0.0
        self.boundary_average_intensity = 0.0
        self.phase_intensity_deviation = 0.0
        self.shapeindex_median = 0
        self.metrics = {}

    def primary_boundary_metrics(self, cluster):

        self.length = len(self.boundary_coordinates)
        self.length = estimate_boundary_length(np.array(self.boundary_coordinates))

        # retrieve particle regionprops
        P1, P2 = cluster.particles[self.p1],\
                 cluster.particles[self.p2]

        # centroid coordinates
        x1, y1 = P1.centroid
        x2, y2 = P2.centroid

        # particle bbox
        bbox1 = P1.bbox
        bbox2 = P2.bbox
        self.bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),
                     max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]

        # particle major axis length
        L1 = P1.major_axis_length
        L2 = P2.major_axis_length

        # areas of the two particles
        A1, A2 = P1.area, P2.area

        # mean intensity
        mean_intensity = (P1.mean_intensity*A1 + P2.mean_intensity*A2)/(A1+A2)

        # perimeters of two particles:
        per1, per2 = P1.perimeter, P2.perimeter

        # calculate particle orientation
        theta1, theta2 = orientation_by_eig(P1), orientation_by_eig(P2)

        # orientation of line connecting two centroids
        thetaP1P2 = np.arctan((x2 - x1) / (y2 - y1))

        # xy coordinates of boundary
        boundary_x, boundary_y = np.array(self.boundary_coordinates).T

        # euclidean distance between two centroids
        dP1P2 = distance([x1 + bbox1[0], y1 + bbox1[1]], [x2 + bbox2[0], y2 + bbox2[1]])

        # if distance between two centroids approximates a half of total lengths,
        # the two objects are more likely to be from one cell
        self.relative_distance = 2 * dP1P2 / (L1 + L2)
        if self.relative_distance > 1:
            self.relative_distance == 1

        # deflection angle between two objects, the smaller it is , the more likely
        # the two particles belong to one
        self.deflection_angle = deflection_angle(theta1, theta2) / (0.5 * np.pi)
        self.center_line_angle = max(deflection_angle(theta1, thetaP1P2),
                                     deflection_angle(theta1, thetaP1P2)) / (0.5 * np.pi)

        # if the length of the boundary is larger than over half of the length of the smaller object,
        # the two objects are more likely to be from one cell
        self.max_perimeter_fraction = self.length / min(per1, per2)

        # signal intensities along boundary
        self.boundary_average_intensity = np.mean(cluster.phase[boundary_x, boundary_y].flatten())
        self.boundary_sobel_mean = cluster.sobel[boundary_x, boundary_y].mean()
        self.boundary_sobel_std = cluster.sobel[boundary_x, boundary_y].std()
        self.boundary_normalized_intensity = self.boundary_average_intensity/cluster.global_mask_foreground_median
        self.phase_intensity_deviation = abs(self.boundary_average_intensity-mean_intensity) / \
                                         cluster.global_mask_background_std

        self.shapeindex_std = cluster.shape_indexed[boundary_x, boundary_y].std()
        self.shapeindex_mean = cluster.shape_indexed[boundary_x, boundary_y].mean()
        self.frangi_mean = cluster.frangi[boundary_x, boundary_y].mean()
        self.frangi_std = cluster.frangi[boundary_x, boundary_y].std()
        self.DoG_mean = cluster.DoG[boundary_x, boundary_y].mean()
        self.DoG_std = cluster.DoG[boundary_x, boundary_y].std()
        self.RoG_mean = cluster.RoG[boundary_x, boundary_y].mean()
        self.RoG_std = cluster.RoG[boundary_x, boundary_y].std()
        self.min_neighbor_size = (min(A1, A2))*(cluster.pixel_microns**2)

        self.metrics = {'Maximum_perimeter_fraction': self.max_perimeter_fraction,
                        'Min_neighbor_size': self.min_neighbor_size,
                        'Boundary_length': self.length,
                        'Particle_deflection_angle': self.deflection_angle,
                        'Centroid_deflection_angle': self.center_line_angle,
                        'Normalized_centroid_distance': self.relative_distance,
                        'Intensity': self.boundary_average_intensity,
                        'Intensity_deviation': self.phase_intensity_deviation,
                        'Shapeindex_mean': self.shapeindex_mean,
                        'Shapeindex_variation': self.shapeindex_std,
                        'Sobel_mean': self.boundary_sobel_mean,
                        'Sobel_variation': self.boundary_sobel_std,
                        'Frangi_mean': self.frangi_mean,
                        'Frangi_std': self.frangi_std,
                        'DoG_mean': self.DoG_mean,
                        'DoG_std': self.DoG_std,
                        'RoG_mean': self.RoG_mean,
                        'RoG_std': self.RoG_std}

    """
    def boundary_classification_default(self, cluster):

        min_boundary_length = int(float(cluster.config['width_low']) / (cluster.pixel_microns * np.sqrt(2)))

        if self.max_perimeter_fraction > float(cluster.config['perimeter_fraction_cutoff']):
            self.false_boundary = True

        elif self.length >= min_boundary_length:

            byShape = bool(max(self.deflection_angle, self.center_line_angle) <\
                           float(cluster.config['max_deflection_angle']))

            byDist = bool(self.relative_distance >= float(cluster.config['normalized_centroid_distance_min']))

            byShapeid = bool(self.shapeindex_mean <= float(cluster.config['max_boundary_shape_index']))

            byFrangi = bool(self.frangi_mean >= float(cluster.config['min_boundary_frangi']))

            if np.sum([byShape, byDist, byShapeid, byFrangi]) >= 3:
                self.false_boundary = True


    
    deprecated functions
    def boundary_classification_SVM(self, predictor):

        self.false_boundary = False

    def boundary_classification_decision_tree(self, predictor):

        boundary_data = np.array(list(self.metrics.values())).reshape(1, -1)
        self.false_boundary = bool(predictor.predict(boundary_data)[0])

    def boundary_visualization(self, cluster):
        x1,y1,x2,y2 = self.bbox
        phase_img = cluster.phase[x1:x2,y1:y2]
        mask_img = (cluster.watersheded == self.p1)[x1:x2,y1:y2] & (cluster.watersheded == self.p2)[x1:x2, y1:y2]
        _xcoords, _ycoords = np.array(self.boundary_coordinates).T
        xcoords = _xcoords-x1
        ycoords = _ycoords-y1
        return phase_img, mask_img, xcoords, ycoords
    """

def boundary_neighbor_pairwise(watershed, boundary_mask, connectivity=1):
    particle_pairwise_dict = {}
    boundary_pairwise_dict = {}
    xlist, ylist = np.where(boundary_mask > 0)
    xmax, ymax = watershed.shape
    n = 0
    unique_neighbors = []
    for i in range(len(xlist)):
        x, y = xlist[i], ylist[i]
        # find unique neighbors
        if connectivity == 1:
            m = find_unique([watershed[max(x - 1, 0), y], watershed[min(x + 1, xmax - 1), y],
                             watershed[x, max(y - 1, 0)], watershed[x, min(ymax - 1, y + 1)]])
        elif connectivity == 2:
            m = find_unique(watershed[max(x - 1, 0):min(x + 1, xmax - 1), max(y - 1, 0):max(y - 1, 0)].flatten())

        # discard background pixels
        if (m[0] == 0) & (len(m) >= 3):
            unique_neighbors = m[1:]
        elif (m[0] != 0) & (len(m) >= 2):
            unique_neighbors = m
        if len(unique_neighbors) != 0:
            for pair in combinations(unique_neighbors, 2):
                p1, p2 = pair
                if p1 not in particle_pairwise_dict:
                    particle_pairwise_dict[p1] = {p2: n}
                    n += 1
                elif p2 not in particle_pairwise_dict[p1]:
                    particle_pairwise_dict[p1][p2] = n
                    n += 1
                idx = particle_pairwise_dict[p1][p2]
                if idx not in boundary_pairwise_dict:
                    boundary_pairwise_dict[idx] = Boundary(p1,p2,[[x,y]])
                else:
                    boundary_pairwise_dict[idx].boundary_coordinates += [[x, y]]
    return particle_pairwise_dict, boundary_pairwise_dict


def find_unique(datalist):
    return sorted(list(set(datalist)))


def deflection_angle(theta1, theta2):
    return min(abs(theta1 + theta2), abs(theta1 - theta2))


def boundary_annotation_thumbnail(boundary, cluster):
    x1,y1,x2,y2 = boundary.bbox
    phase = np.round(normalize_data1D(cluster.phase[x1:x2,y1:y2],
                                      re_orient=False, base=0)*255).astype(np.uint8)
    coords = np.array(boundary.boundary_coordinates)
    phase_labeled = phase.copy()
    phase_labeled[coords[:,0]-x1,coords[:,1]-y1] = 255
    return np.concatenate([phase,phase_labeled],axis=1)





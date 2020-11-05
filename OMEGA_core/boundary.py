from .helper import *


def boundary_neighbor_pairwise(mask, labels, connectivity=1):
    particle_pairwise_dict = {}
    boundary_pairwise_dict = {}
    xlist, ylist = np.where(labels > 0)
    xmax, ymax = mask.shape
    n = 0
    unique_neighbors = []
    for i in range(len(xlist)):
        x, y = xlist[i], ylist[i]
        # find unique neighbors
        if connectivity == 1:
            m = find_unique([mask[max(x - 1, 0), y], mask[min(x + 1, xmax - 1), y], \
                             mask[x, max(y - 1, 0)], mask[x, min(ymax - 1, y + 1)]])
        elif connectivity == 2:
            m = find_unique(mask[max(x - 1, 0):min(x + 1, xmax - 1), max(y - 1, 0):max(y - 1, 0)].flatten())

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
        # boundary < 3 pixels are unlikely to be false ones.
        self.length = len(self.boundary_coordinates)
        if self.length >=3:
            self.length = estimate_boundary_length(np.array(self.boundary_coordinates))
            # retrieve particle regionprops
            P1, P2 = cluster.particles[self.p1], cluster.particles[self.p2]

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
            # compute the mean signal intensity along boundary
            self.boundary_average_intensity = np.mean(sorted(cluster.phase[boundary_x, boundary_y].flatten())[:-2])

            self.phase_intensity_deviation = abs(self.boundary_average_intensity-mean_intensity)/cluster.global_mask_background_std
            self.shapeindex_mean = np.mean(cluster.shape_indexed[boundary_x, boundary_y])
            self.metrics = {'Maximum_perimeter_fraction': self.max_perimeter_fraction,
                            'Boundary_length': self.length,
                            'Particle_deflection_angle': self.deflection_angle,
                            'Centroid_deflection_angle': self.center_line_angle,
                            'Normalized_centroid_distance': self.relative_distance,
                            'Intensity_deviation': self.phase_intensity_deviation,
                            'Shapeindex_mean':self.shapeindex_mean}

    def boundary_classification_default(self, cluster):

        min_boundary_length = int(float(cluster.config['width_low']) / (cluster.pixel_microns * np.sqrt(2)))

        if self.max_perimeter_fraction > float(cluster.config['perimeter_fraction_cutoff']):
            self.false_boundary = True

        elif self.length >= min_boundary_length:
            if max(self.deflection_angle, self.center_line_angle) < float(cluster.config['max_deflection_angle']):
                if self.relative_distance >= float(cluster.config['normalized_centroid_distance_min']):
                    if self.shapeindex_mean <= 90:
                        self.false_boundary = True

    def boundary_visualization(self, cluster):
        x1,y1,x2,y2 = self.bbox
        phase_img = cluster.phase[x1:x2,y1:y2]
        mask_img = (cluster.watersheded == self.p1)[x1:x2,y1:y2] & (cluster.watersheded == self.p2)[x1:x2, y1:y2]
        _xcoords, _ycoords = np.array(self.boundary_coordinates).T
        xcoords = _xcoords-x1
        ycoords = _ycoords-y1
        return phase_img, mask_img, xcoords, ycoords




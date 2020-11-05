import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import spatial, stats
from matplotlib import cm, patches
from statsmodels.stats.multitest import fdrcorrection
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import re
from collections import Counter


class miniSAFE:

    def __init__(self, enrichment_threshold=0.1,
                 attribute_enrichment_min_size=5,
                 neighborhood_radius=0.12,
                 attribute_relative_distance_threshold=0.1):
        self.graph = None
        self.locus2attribute = None
        self.attributes_stats = None
        self.node_mapped = None
        self.enrichment_threshold = enrichment_threshold
        self.attribute_enrichment_min_size = attribute_enrichment_min_size

        self.neighborhood_radius = neighborhood_radius
        self.neighborhood_radius_type = None
        self.neighborhoods = None

        self.dataframe = None
        self.gff = None
        self.pos_dict = {}
        self.feature1 = None
        self.feature2 = None
        self.attribute_relative_distance_threshold = attribute_relative_distance_threshold

    def load_input_dataframe(self,
                             filename,
                             locus_key='locus'):
        # load excel
        self.dataframe = pd.read_excel(filename)
        self.nodes = self.dataframe[locus_key].values

    def load_genome_annotation_table(self, filename, locus_key='Locus'):
        # load excel
        self.gff = pd.read_excel(filename)
        self.all_genes = self.gff[locus_key].values

    def get_attributes(self, filename, locus_key='Locus', attribute_key='GO term'):

        # load attribute table
        if filename.endswith('csv'):
            attribute_table = pd.read_csv(filename)
        elif filename.endswith('xls') or filename.endswith('xlsx'):
            attribute_table = pd.read_excel(filename)
        self.locus2attribute = preset_attributes(attribute_table, self.gff, locus_key, attribute_key)
        self.attributes_stats, self.node_mapped = node2attributes(self.locus2attribute,
                                                                  self.nodes)

        # set nodes with no attribute assigned to NaN
        node_attribute_count = np.sum(self.node_mapped.values, axis=1)
        zero_count_index = np.where(node_attribute_count == 0)
        self.node_mapped.loc[self.nodes[zero_count_index], :] = np.nan

    def construct_graph2D(self,
                          feature1, feature2, connectivity=2):
        self.graph = None
        self.feature1 = feature1
        self.feature2 = feature2

        # get values regarding the two features of interests
        xval = self.dataframe[feature1].values
        yval = self.dataframe[feature2].values

        # compute flattened dist matrix, find 5 & 95 percentile
        # distance interval
        xyval = np.array([xval, yval]).T
        distM1D = spatial.distance.pdist(xyval)
        d1, d2 = np.percentile(distM1D, 5), np.percentile(distM1D, 95)
        interval = d2 - d1
        cutoff = self.neighborhood_radius * interval

        # force scattered positions
        self.coordinates = dict(zip(np.arange(len(self.nodes)), np.array([xval, yval]).T))
        self.graph, self.neighborhoods, self.dist_matrix = preset_graph(self.nodes,
                                                                        xval, yval,
                                                                        distance_cutoff=cutoff)
        self.neighborhoods = connectivity_dist_m(self.neighborhoods, connectivity=connectivity)

    def compute_pvalues_by_hypergeom(self):

        # Nodes with not-NaN values in >= 1 attribute
        nodes_not_nan = np.any(~np.isnan(self.node_mapped), axis=1)

        # -- Number of nodes
        # n = self.graph.number_of_nodes()    # total
        n = np.sum(nodes_not_nan)  # with not-NaN values in >=1 attribute

        N = np.zeros([len(self.nodes), len(self.attributes_stats)]) + n

        # -- Number of nodes annotated to each attribute
        N_in_group = np.tile(np.nansum(self.node_mapped, axis=0),
                             (len(self.nodes), 1))

        # -- Number of nodes in each neighborhood
        neighborhood_size = np.dot(self.neighborhoods,
                                   nodes_not_nan.astype(int))[:, np.newaxis]
        N_in_neighborhood = np.tile(neighborhood_size, (1, len(self.attributes_stats)))

        # -- Number of nodes in each neighborhood and  annotated to each attribute
        N_in_neighborhood_in_group = np.dot(self.neighborhoods,
                                            np.where(~np.isnan(self.node_mapped), self.node_mapped, 0))

        self.pvalues_pos = stats.hypergeom.sf(N_in_neighborhood_in_group - 1,
                                              N, N_in_group, N_in_neighborhood)

        self.pvalues_pos_fdr = np.apply_along_axis(fdrcorrection, 1, self.pvalues_pos)[:, 1, :]
        self.nes = -np.log10(self.pvalues_pos)
        self.nes_fdr = -np.log10(self.pvalues_pos_fdr)

        idx = ~np.isnan(self.nes)
        idx_fdr = ~np.isnan(self.nes_fdr)
        self.nes_binary = np.zeros(self.nes.shape)
        self.nes_binary_fdr = np.zeros(self.nes.shape)
        self.nes_binary[idx] = np.abs(self.nes[idx]) > -np.log10(self.enrichment_threshold)
        self.nes_binary_fdr[idx] = np.abs(self.nes_fdr[idx_fdr]) > -np.log10(self.enrichment_threshold)
        self.attributes_stats['num_neighborhoods_enriched'] = np.sum(self.nes_binary, axis=0)
        self.attributes_stats['num_neighborhoods_enriched_fdr'] = np.sum(self.nes_binary_fdr, axis=0)

    def define_top_attributes(self, fdr=True):

        # preset top attributes to false
        self.attributes_stats['top'] = False

        # connectivity stats
        self.attributes_stats['num_connected_components'] = 0
        self.attributes_stats['size_connected_components'] = None
        self.attributes_stats['size_connected_components'] = self.attributes_stats['size_connected_components'].astype(
            object)
        self.attributes_stats['num_large_connected_components'] = 0

        # locate attributes with number of enriched neighborhoods over defined
        # defined threshold, default = 1
        if fdr:
            idx = np.where(self.attributes_stats['num_neighborhoods_enriched_fdr'] >= \
                           self.attribute_enrichment_min_size)[0]
            self.attributes_stats.loc[idx, 'top'] = True
        else:
            idx = np.where(self.attributes_stats['num_neighborhoods_enriched'] >= \
                           self.attribute_enrichment_min_size)[0]
            self.attributes_stats.loc[idx, 'top'] = True

        # forcing all enriched neightborhoods to be interconnected may miss functionally important
        # attributes that are present in two or more separate clusters. Will adapt the connectivity
        # assignment method without forcing connectivity
        for attribute in self.attributes_stats.index.values[self.attributes_stats['top']]:

            # locate all enriched neighborhoods for a given attribute
            if fdr:
                enriched_neighborhoods = list(np.where((self.nes_binary_fdr[:, attribute] > 0))[0])
            else:
                enriched_neighborhoods = list(np.where((self.nes_binary[:, attribute] > 0))[0])

            # create subgraph, find all interconnected regions
            H = nx.subgraph(self.graph, enriched_neighborhoods)
            connected_components = sorted(nx.connected_components(H),
                                          key=len, reverse=True)
            num_connected_components = len(connected_components)

            # calculate sizes of all interconnected regions
            size_connected_components = np.array([len(c) for c in connected_components])

            # remove miniscule interconnected regions
            num_large_connected_components = np.sum(size_connected_components >= self.attribute_enrichment_min_size)
            self.attributes_stats.loc[attribute, 'num_connected_components'] = num_connected_components
            self.attributes_stats.at[attribute, 'size_connected_components'] = size_connected_components
            self.attributes_stats.loc[attribute, 'num_large_connected_components'] = num_large_connected_components

            # deprecated for now
            self.attributes_stats.loc[self.attributes_stats['num_connected_components'] > 1, 'top'] = False

    def define_domains(self, fdr=True):

        # binary matrix of only top attributes
        # cluster domains by hierarchical clusterin
        m = self.nes_binary[:, self.attributes_stats['top']].T
        Z = linkage(m, method='average', metric='euclidean')
        max_d = np.max(Z[:, 2] * self.attribute_relative_distance_threshold)
        domains = fcluster(Z, max_d, criterion='distance')

        # assign domains to attributes
        self.attributes_stats['domain'] = 0
        self.attributes_stats.loc[self.attributes_stats['top'], 'domain'] = domains

        # assign nodes to domains
        if fdr:
            node2nes = pd.DataFrame(data=self.nes_fdr,
                                    columns=[self.attributes_stats.index.values,
                                             self.attributes_stats['domain']])
            node2nes_binary = pd.DataFrame(data=self.nes_binary_fdr,
                                           columns=[self.attributes_stats.index.values,
                                                    self.attributes_stats['domain']])
        else:
            node2nes = pd.DataFrame(data=self.nes,
                                    columns=[self.attributes_stats.index.values,
                                             self.attributes_stats['domain']])
            node2nes_binary = pd.DataFrame(data=self.nes_binary,
                                           columns=[self.attributes_stats.index.values,
                                                    self.attributes_stats['domain']])

        self.node2domain = node2nes_binary.groupby(level='domain', axis=1).sum()

        # From safepy: A node belongs to the domain that contains the highest number of attributes
        # for which the nodes is significantly enriched
        t_max = self.node2domain.loc[:, 1:].max(axis=1)
        t_idxmax = self.node2domain.loc[:, 1:].idxmax(axis=1)
        t_idxmax[t_max == 0] = 0
        self.node2domain['primary_domain'] = t_idxmax

        # From safepy: Get the max NES for the primary domain
        o = node2nes.groupby(level='domain', axis=1).max()
        i = pd.Series(t_idxmax)
        self.node2domain['primary_nes'] = o.lookup(i.index, i.values)

    def trim_domains(self):

        # get total node count of each domain
        domain_counts = np.zeros(len(self.attributes_stats['domain'].unique())).astype(int)
        t = self.node2domain.groupby('primary_domain')['primary_domain'].count()
        domain_counts[t.index] = t.values

        # Remove domains that are the top choice for less than a certain number of neighborhoods
        to_remove = np.flatnonzero(domain_counts < self.attribute_enrichment_min_size)
        self.attributes_stats.loc[self.attributes_stats['domain'].isin(to_remove), 'domain'] = 0
        idx = self.node2domain['primary_domain'].isin(to_remove)
        self.node2domain.loc[idx, ['primary_domain', 'primary_nes']] = 0

        # Rename the domains (simple renumber)
        a = np.sort(self.attributes_stats['domain'].unique())
        b = np.arange(len(a))
        renumber_dict = dict(zip(a, b))

        self.attributes_stats['domain'] = [renumber_dict[k] for k in self.attributes_stats['domain']]
        self.node2domain['primary_domain'] = [renumber_dict[k] for k in self.node2domain['primary_domain']]

        # Make labels for each domain
        domains = np.sort(self.attributes_stats['domain'].unique())
        domains_labels = self.attributes_stats.groupby('domain')['attributes'].apply(str_cat)
        self.domains = pd.DataFrame(data={'id': domains, 'label': domains_labels})
        self.domains.set_index('id', drop=False)

        labels = self.domains['label'].values
        labels[0] = 'background'
        self.domains['label'] = labels


def preset_attributes(attribute_table, gff_table, locus_key='Locus', attribute_key='GO term'):
    locus_set = np.sort(np.array(gff_table.Locus.unique()))
    attribute_set = np.sort(np.array(attribute_table[attribute_key].unique()))
    matrix = np.zeros((len(locus_set), len(attribute_set)))

    locus2attribute = pd.DataFrame(matrix, columns=attribute_set)
    locus2attribute.set_index(locus_set, drop=True, inplace=True)
    for attribute in attribute_set:
        subset = attribute_table[attribute_table[attribute_key] == attribute][locus_key].values
        locus2attribute.loc[subset, attribute] += 1
    return locus2attribute


def node2attributes(locus2attribute, nodes):
    node_mapped = locus2attribute.loc[nodes, :].copy()
    node_mapped = node_mapped.drop(node_mapped.columns[np.where(node_mapped.sum() <= 3)], axis=1)
    attribute_stats = pd.DataFrame({'attributes': node_mapped.columns,
                                    'count': np.array(node_mapped.sum())})
    return attribute_stats, node_mapped


def preset_graph(nodes, xval, yval, distance_cutoff):
    # set nodes to unique locus list
    g = nx.Graph()
    num_nodes = np.arange(len(nodes))
    g.add_nodes_from(num_nodes)

    xdict = dict(zip(num_nodes, xval))
    ydict = dict(zip(num_nodes, yval))
    nx.set_node_attributes(g, xdict, 'x')
    nx.set_node_attributes(g, ydict, 'y')

    # calculate distance matrix, locate pairs with edge distance lower than threshould
    xyval = np.array([xval, yval]).T
    dm = spatial.distance_matrix(xyval, xyval)
    # omit diagonal zeros
    filtered_dm = np.zeros(dm.shape)
    filtered_dm[np.where(dm <= distance_cutoff)] = 1

    # set network edges
    p1 = np.where(filtered_dm != 0)[0]
    p2 = np.where(filtered_dm != 0)[1]
    dm[np.diag_indices_from(dm)] = 0
    w = dm[np.where(filtered_dm != 0)]
    pairs = np.array([p1, p2]).T
    g.add_edges_from(pairs)
    # g.remove_edges_from(nx.selfloop_edges(g))
    return g, filtered_dm, dm


# attribute name simplifier. inherited from safepy with no change
def chop_and_filter(s):
    single_str = s.str.cat(sep=' ')
    single_list = re.findall(r"[\w']+", single_str)

    single_list_count = dict(Counter(single_list))
    single_list_count = [k for k in sorted(single_list_count, key=single_list_count.get, reverse=True)]

    to_exclude = ['of', 'a', 'the', 'an', ',', 'via', 'to', 'into', 'from']
    single_list_words = [w for w in single_list_count if w not in to_exclude]

    return ' | '.join(single_list_words[:5])


# return attribute by directly concatenating strings
def str_cat(s):
    single_str = s.str.cat(sep=' | ')
    return single_str


from skimage import morphology, measure, feature, filters
import tifffile
import OMEGA as om


def scatter2outline(xycoords, base=20, ampl_factor=10):
    # enlarge scatter map
    enlarged = (xycoords + base) * ampl_factor
    # convert to integers
    enlarged_int = np.round(enlarged).astype(int)
    # create canvas
    xmin, xmax, ymin, ymax = enlarged_int[:, 0].min(), \
                             enlarged_int[:, 0].max(), \
                             enlarged_int[:, 1].min(), \
                             enlarged_int[:, 1].max()
    height = xmax + (base * ampl_factor)
    width = ymax + (base * ampl_factor)
    canvas = np.zeros((height, width))
    canvas[enlarged_int[:, 0], enlarged_int[:, 1]] = 1
    canvas = filters.gaussian(canvas, sigma=0.5) > 0
    convex_hull = morphology.convex_hull_image(canvas)
    outline = measure.find_contours(convex_hull, level=0.5)[0]
    outline = om.helper.spline_approximation(outline, n=100, smooth_factor=5, closed=True)
    return (outline / ampl_factor) - base


def connectivity_dist_m(dist_matrix, connectivity=2):
    new_matrix = np.zeros(dist_matrix.shape)
    for i in range(len(dist_matrix)):
        idx = np.array([i])
        for j in range(connectivity):
            neighbors = np.where(np.sum(dist_matrix[idx, :], axis=0) > 0)[0]
            idx = np.concatenate([idx, neighbors])
        new_matrix[i, idx] = 1
    return new_matrix
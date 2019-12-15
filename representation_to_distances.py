import configparser
import numpy as np
from scipy.spatial.distance import cdist, euclidean, mahalanobis, hamming, cityblock, sqeuclidean, \
    correlation
import pandas as pd
from itertools import combinations, product
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import label_data_small_graph

config = configparser.ConfigParser()
config.read('conf.ini')
DIMENSION = int(config['ARGUMENTS']['dimensions'])
TREE_LEN = int(config['ARGUMENTS']['tree_len'])
SAME_GROUP = (0, 0)
SAME_DEPTH_NOT_EQUAL = (1, 1)
MOST_DIFFERENT = (TREE_LEN, TREE_LEN, 0)

plt.style.use('seaborn-deep')


def distances_distribution():
    a = []
    ae_matrix = read_ae_matrix()
    sets_for_communities = organize_sets()
    labels_dict = pickle.load(open(os.path.join('small_graph', "labels.pkl"), 'rb'))
    node_to_matrix_row = {}
    for i in range(ae_matrix.shape[0]):
        node_to_matrix_row[str(int(ae_matrix[i, 0]))] = i
    ae_matrix = ae_matrix[:, 1:]
    submatrices = []
    for s in sets_for_communities:
        indices = [node_to_matrix_row[str(i)] for i in s[:-1]]
        submatrices.append((ae_matrix[indices, :], s[-1]))
    covariance_matrix = np.cov(ae_matrix.transpose())
    metrics = ['euclidean', 'mahalanobis', 'cityblock', 'sqeuclidean', 'correlation']
    methods = ['min', 'max', 'avg', 'centroid']
    fig, ax = plt.subplots(6, 4, figsize=(12, 10))

    pairs_of_sets = to_pairs(submatrices, labels_dict)
    for j in range(4):
        ax[0, j].axis('off')
    for i, j in product(range(5), range(4)):
        s1 = 0
        s2 = 0
        # for w in range(10):
        calc_distances_1 = []
        calc_distances_2 = []
        calc_distances_3 = []

        for (m_1, m_2), dist in pairs_of_sets:
            d = ae_to_distance(m_1, m_2, metric=metrics[i], method=methods[j], cov=covariance_matrix)
            if dist == SAME_GROUP + (2,):
                calc_distances_1.append(d)
            elif dist == SAME_DEPTH_NOT_EQUAL + (1,):
                calc_distances_2.append(d)
            elif dist == MOST_DIFFERENT:
                calc_distances_3.append(d)
            else:
                raise ValueError("Some wierd distance:", dist)

        all_distances = calc_distances_1 + calc_distances_2 + calc_distances_3
        a += all_distances
        # print(all_distances)
        bins = np.linspace(min(all_distances) - 0.1, max(all_distances) + 0.1, 20)
        ax[i + 1, j].hist([calc_distances_1, calc_distances_2, calc_distances_3], bins,
                          label=['same', 'different near', 'different far'])
        ax[i + 1, j].set_title('%s, %s' % (metrics[i], methods[j]), fontsize=10)
        ax[i + 1, j].legend(loc="upper right", fontsize=8)
        # print("calc1: ", len(calc_distances_1), "calc2: ", len(calc_distances_2), "calc3: ", len(calc_distances_3))
        # print("calc1: ", calc_distances_1, "calc2: ", calc_distances_2, "calc3: ", calc_distances_3)
        f_val, p_score = f_oneway(calc_distances_1, calc_distances_2, calc_distances_3)
        s1 += f_val
        s2 += p_score
        ###print(i, j, "values:", s1 / 10, s2 / 10)
    # print(a)
    plt.suptitle('Deviations of the predicted distances from the actual distances', y=0.95)
    plt.tight_layout(pad=0.5, h_pad=0.6, w_pad=1.25)
    plt.savefig("deviations_histogram.png")



def organize_sets():
    '''group labled is in the form of:  [ (1, [7,5,9,6] ), (2, [88, 95,3,2), ..... ], which is list of tuples, each tuple contains the group lable from the tree
    in the 0 index, the group nodes list in the second index'''
    with open('metric_list.pkl', 'rb') as f:
        groups_labeled = pickle.load(f)

    groups_subsets = []
    for group in groups_labeled:
        groups_subset = create_subsets(group[1])
        for subset in groups_subset:
            groups_subsets.append((group[0], subset))

    sets_and_labels = []
    for groups_subset in groups_subsets:
        sets_and_labels.append( groups_subset[1] + [ groups_subset[0]])
    return sets_and_labels


def create_subsets(original_list):
    all_subsets = []
    for i in range(2):
        random_indices = np.random.choice(original_list, round(len(original_list) / (2 * (i + 1))), replace=False)
        complement = [a for a in original_list if a not in random_indices]
        all_subsets += [list(random_indices), complement]
    return all_subsets


def to_pairs(matrices, labels_dict):
    matrices_combinations = list(combinations(matrices, 2))
    combinations_same_commmunities = [(m_1, m_2) for ((m_1, l_1), (m_2, l_2)) in matrices_combinations if l_1 == l_2]
    combinations_different_communities = [(m_1, m_2)
                                          for ((m_1, l_1), (m_2, l_2)) in matrices_combinations if l_1 != l_2]
    corresponding_labels_same = [(l_1, l_2) for ((m_1, l_1), (m_2, l_2)) in matrices_combinations if l_1 == l_2]
    corresponding_labels_different = [(l_1, l_2) for ((m_1, l_1), (m_2, l_2)) in matrices_combinations if l_1 != l_2]
    same_10 = np.random.choice(np.arange(len(combinations_same_commmunities)), 10, replace=False)
    diff_10 = np.random.choice(np.arange(len(combinations_different_communities)), 10, replace=False)
    same_comm = [combinations_same_commmunities[ind] for ind in same_10]
    diff_comm = [combinations_different_communities[ind] for ind in diff_10]
    same_labels = [corresponding_labels_same[ind] for ind in same_10]
    diff_labels = [corresponding_labels_different[ind] for ind in diff_10]
    matrices_pairs = same_comm + diff_comm
    labels_pairs = same_labels + diff_labels
    pairs_and_distances = []
    for (m1, m2), (l_1, l_2) in zip(matrices_pairs, labels_pairs):
        pairs_and_distances.append(((m1, m2), labels_dict[(l_1, l_2)]))
    return pairs_and_distances


def read_ae_matrix():
    ae_file = open(os.path.join("small_graph", "projected_graph"), "r")
    ae_lol = []  # list of lists
    for line in ae_file.readlines():
        ln = line.split(" ")
        if len(ln) == DIMENSION + 1:
            to_append = []
            for ind, val in enumerate(ln):
                if ind == 0:
                    to_append.append(int(val))
                elif ind == DIMENSION:
                    val = val[:-1]
                    to_append.append(float(val))
                else:
                    to_append.append(float(val))
            ae_lol.append(to_append)
    ae_matrix = np.array(ae_lol)
    return ae_matrix


def ae_to_distance(mat1, mat2, metric='euclidean', method='avg', cov=None):
    # Methods: min, max, avg, centroid
    # Metrics: Euclidean, Manhattan (Cityblock), Mahalanobis (requires a covariance matrix) ,Add: Maximum
    if not method == 'centroid':
        vec_dist = cdist(mat1, mat2, metric=metric)
        if method == 'min':
            calc_dist = np.min(vec_dist)
        elif method == 'max':
            calc_dist = np.max(vec_dist)
        elif method == 'avg':
            calc_dist = np.nanmean(vec_dist)
        else:
            raise ValueError("Wrong name of method")
    else:
        cent_1 = np.mean(mat1, axis=0).reshape((1, -1))
        cent_2 = np.mean(mat2, axis=0).reshape((1, -1))
        if metric == 'euclidean':
            calc_dist = euclidean(cent_1, cent_2)
        elif metric == 'cityblock':
            calc_dist = cityblock(cent_1, cent_2)
        elif metric == 'hamming':
            calc_dist = hamming(cent_1, cent_2)
        elif metric == 'correlation':
            calc_dist = correlation(cent_1, cent_2)
        elif metric == 'cityblock':
            calc_dist = cityblock(cent_1, cent_2)
        elif metric == 'sqeuclidean':
            calc_dist = sqeuclidean(cent_1, cent_2)
        elif metric == 'mahalanobis':
            if cov is None:
                raise ValueError("Insert covariance matrix")
            calc_dist = mahalanobis(cent_1, cent_2, VI=np.linalg.inv(cov))
        else:
            raise ValueError("Wrong name of metric")

    return calc_dist


if __name__ == '__main__':

    distances_distribution()

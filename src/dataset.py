from parse import args
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import numpy as np
import scipy.sparse
import scipy.io
import scipy
import json
import csv
import sys
import data_utils
import pickle
from dgl.data.utils import load_graphs

from src.utils import csr_matrix_to_edge_index

SMALL_OFFSET = 1e-10
DATAPATH = args.DATAPATH + "graph/"


def categorical_label(input):
    r, c = np.where(input == 1)
    labels = np.ones(input.shape[0]) * (-1)
    labels[r] = c
    return labels


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)) + SMALL_OFFSET
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
    # return sparse_to_tuple(features)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


class NCDataset(object):
    def __init__(self, name):
        self.name = name
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(dataname):
    if dataname == "t-finance":
        dataset = load_finance_dataset(dataname)
        return dataset
    elif dataname == "gambling":
        dataset = load_gambling_dataset()
        return dataset
    raise Exception('wrong dataname')

def load_gambling_dataset():
    graph = pickle.load(open('../data/gambling/data.pkl', 'rb'))

    dataset = NCDataset('gambling')
    edge_index = torch.from_numpy(csr_matrix_to_edge_index(graph['adj'])).type(torch.long)
    node_feat = graph['card_x'].type(torch.float32)
    num_nodes = len(graph['label'])

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.from_numpy(graph['label']).unsqueeze(1).type(torch.long)
    dataset.label = label
    return dataset

def load_finance_dataset(name, anomaly_alpha=None, anomaly_std=None):
    graph, label_dict = load_graphs('../data/tfinance')
    graph = graph[0]
    graph.ndata['label'] = graph.ndata['label'].argmax(1)

    if anomaly_std:
        feat = graph.ndata['feature'].numpy()
        anomaly_id = graph.ndata['label'][:, 1].nonzero().squeeze(1)
        feat = (feat - np.average(feat, 0)) / np.std(feat, 0)
        feat[anomaly_id] = anomaly_std * feat[anomaly_id]
        graph.ndata['feature'] = torch.tensor(feat)
        graph.ndata['label'] = graph.ndata['label'].argmax(1)

    if anomaly_alpha:
        feat = graph.ndata['feature'].numpy()
        anomaly_id = list(graph.ndata['label'][:, 1].nonzero().squeeze(1))
        normal_id = list(graph.ndata['label'][:, 0].nonzero().squeeze(1))
        label = graph.ndata['label'].argmax(1)
        diff = anomaly_alpha * len(label) - len(anomaly_id)
        import random
        new_id = random.sample(normal_id, int(diff))
        # new_id = random.sample(anomaly_id, int(diff))
        for idx in new_id:
            aid = random.choice(anomaly_id)
            # aid = random.choice(normal_id)
            feat[idx] = feat[aid]
            label[idx] = 1  # 0

    dataset = NCDataset(name)
    num_nodes = graph.number_of_nodes()
    edge_index = torch.cat((graph.edges()[0].unsqueeze(0), graph.edges()[1].unsqueeze(0)), 0)
    node_feat = graph.ndata['feature'].float()

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = graph.ndata['label'].unsqueeze(1)
    dataset.label = label
    return dataset

def load_amazon_dataset(name):
    fulldata = scipy.io.loadmat(f'{DATAPATH}{name}.mat')
    csc_matrix = fulldata['Network']
    edge_index = data_utils.csc_to_edge_index(csc_matrix)
    node_feat = fulldata['Attributes'].toarray()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(fulldata['Label'], dtype=torch.long).squeeze(0).unsqueeze(1)
    dataset.label = label
    return dataset


def load_twitch_dataset(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    A, label, features = load_twitch(lang)
    dataset = NCDataset(lang)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_twitch(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = "{}twitch/{}".format(DATAPATH, lang)
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2] == "True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int)
    with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id: idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]

    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)),
                                 (np.array(src), np.array(targ))),
                                shape=(n, n))
    features = np.zeros((n, 3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    features = features[:, np.sum(features, axis=0) != 0]  # remove zero cols
    new_label = label[reorder_node_ids]
    label = new_label

    return A, label, features


def load_geom_gcn_dataset(name):
    fulldata = scipy.io.loadmat(f'{DATAPATH}{name}.mat')
    edge_index = fulldata['edge_index']
    node_feat = fulldata['node_feat']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset


def load_citation_dataset(dataset_str):
    # downloaded from https://github.com/tkipf/gcn/blob/master/gcn/utils.py
    """
        Loads input data from gcn/data directory
        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
        All objects above must be saved using python pickle module.
        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.{}.{}".format(DATAPATH + "citation/", dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.{}.test.index".format(DATAPATH + "citation_net/", dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))
    stand_idx_dic = {"tst_idx": idx_test, "trn_idx": idx_train, "val_idx": idx_val}

    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(preprocess_features(features).todense(), dtype=torch.float)
    labels = categorical_label(labels)
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(dataset_str)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes,
                     "stand_idx_dic": stand_idx_dic}
    labels = torch.tensor(labels, dtype=torch.long)
    dataset.label = labels
    return dataset


def load_polblogs():
    with np.load(f"{DATAPATH}polblogs_dataset/polblogs.npz") as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])
        edge_index = torch.tensor(adj_matrix.nonzero(), dtype=torch.long)
        node_feat = torch.tensor(adj_matrix.todense(), dtype=torch.float)  # adj-raw as features
        labels = loader.get('labels')
        num_nodes = len(labels)
        dataset = NCDataset("polblogs")
        dataset.graph = {'edge_index': edge_index,
                         'node_feat': node_feat,
                         'edge_feat': None,
                         'num_nodes': num_nodes}
        labels = torch.tensor(labels, dtype=torch.long)
        dataset.label = labels
        return dataset


def load_etg_syn(hr_level=""):
    data_dic = np.load(f"{DATAPATH}synthetic_dataset/hr_{hr_level}/etg_syn{hr_level}.npy", allow_pickle=True).item()
    edge_index = torch.tensor(data_dic["edge_indexes"], dtype=torch.long)
    labels = torch.tensor(data_dic["labels"], dtype=torch.long)
    num_nodes = len(labels)
    node_feat = torch.tensor(data_dic["feat"], dtype=torch.float)  # guassian noise as feature
    dataset = NCDataset("etg_syn_diff_hr_data")
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes,
                     "LR60_splits": data_dic["LR60_splits"],
                     "LR40_splits": data_dic["LR40_splits"],
                     "LR20_splits": data_dic["LR20_splits"]}
    dataset.label = labels
    return dataset

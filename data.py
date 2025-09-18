import numpy as np
import scipy.sparse as sp
import numpy as np
import scipy.io as sio
import pickle as pkl
import sys
import networkx as nx
def load_undirected_map(data_name):
    data_dir = './data/{}.mat'.format(data_name)
    data = sio.loadmat(data_dir)
    features =data['Features']
    labels = data['Label']
    adj= data['Network']
    return adj, features, labels

def load_label_AN(dataset):
    edge_file = open("data/{}.edge".format(dataset), 'r')
    attri_file = open("data/{}.node".format(dataset), 'r')
    label_file = open("data/{}.label".format(dataset), 'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()
    labels = label_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    attribute_number = int(attributes[1].split('\t')[1].strip())
    print("dataset:{}, node_num:{},edge_num:{},attribute_num:{}".format(dataset, node_num, edge_num, attribute_number))
    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row = []
    adj_col = []

    for line in edges:
        node1 = int(line.split('\t')[0].strip())
        node2 = int(line.split('\t')[1].strip())
        adj_row.append(node1)
        adj_col.append(node2)
    adj = sp.csc_matrix((np.ones(edge_num), (adj_row, adj_col)), shape=(node_num, node_num))
        
    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split('\t')[0].strip())
        attribute1 = int(line.split('\t')[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    attribute = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))
    N = len(labels)
    y = np.zeros(N, dtype=int)
    i=0
    for line in labels:
        l = line.strip("\n\r")
        y[i] = int(l)
        i += 1
    
    return adj, attribute, y

def read_label(inputFileName):
    f = open(inputFileName, "r")
    lines = f.readlines()
    f.close()
    N = len(lines)
    y = np.zeros(N, dtype=int)
    i = 0
    for line in lines:
        l = line.strip("\n\r")
        y[i] = int(l)
        i += 1
    return y

def load_AN(dataset):
    edge_file = open("data/{}.edge".format(dataset), 'r')
    attri_file = open("data/{}.node".format(dataset), 'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    attribute_number = int(attributes[1].split('\t')[1].strip())
    print("dataset:{}, node_num:{},edge_num:{},attribute_num:{}".format(dataset, node_num, edge_num, attribute_number))
    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row = []
    adj_col = []

    for line in edges:
        node1 = int(line.split('\t')[0].strip())
        node2 = int(line.split('\t')[1].strip())
        adj_row.append(node1)
        adj_col.append(node2)
    adj = sp.csc_matrix((np.ones(edge_num), (adj_row, adj_col)), shape=(node_num, node_num))
        
    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split('\t')[0].strip())
        attribute1 = int(line.split('\t')[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    attribute = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))
    
    return adj, attribute


def load_citationANEWeight(dataset_str="cora"):
    """
    Load Citation Networks Datasets.

    return adj, features, adj_mat, labels, idx_train, idx_val, idx_test
    adj:初始对称邻接矩阵
    features： 特征
    adj_mat：将特征融入adj的新P
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
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
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        # adj, features = preprocess_citation(adj, features, normalization)
        # features = preprocess_citation_feat(features)

    else:
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

    n = adj.shape[0]
    sum_num = adj.nnz * 1.0
    sparsity = sum_num / ((n) * (n))
    print('---> Sparsity of adj matrix: %.4f' % sparsity)
    

    return features, adj, labels, idx_train, idx_val, idx_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_citation(dataset_str="cora"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
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
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        # adj, features = preprocess_citation(adj, features, normalization)
        # features = preprocess_citation_feat(features)

    elif dataset_str == 'nell.0.001':
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)


        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = adj.astype(float)
        features = features.astype(float)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        # features = preprocess_citation_feat(features)

    else:
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        # adj, features = preprocess_citation(adj, features, normalization)
        # features = preprocess_citation_feat(features)

    return adj, features, labels, idx_train, idx_val, idx_test
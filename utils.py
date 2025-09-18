
import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
import sys
import networkx as nx
import os
import random
import scipy.io as scio
from sklearn.metrics import roc_auc_score,average_precision_score
import time
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn import svm


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def add_self_loops(sparse_mx):
    """Add self-loops to a scipy sparse matrix."""
    num_nodes = sparse_mx.shape[0]
    identity = sp.eye(num_nodes, dtype=sparse_mx.dtype)  # Create identity matrix for self-loops
    sparse_mx_with_loops = sparse_mx + identity  # Add self-loops
    return sparse_mx_with_loops

def to_dense_if_sparse(H):
    if isinstance(H, torch.Tensor) and H.is_sparse:
        H = np.array(H.to_dense())
        return H
    else:
        return H

    
def generate_G_from_H(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    colsum = np.array(mx.sum(0))
    r_inv = np.power(rowsum, -0.5).flatten()
    c_inv = np.power(colsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    c_inv[np.isinf(c_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    c_mat_inv = sp.diags(c_inv)
    # mx = r_mat_inv.dot(mx)
    temp = r_mat_inv.dot(mx)
    temp = temp.dot(c_mat_inv)
    temp = temp.dot(mx.T)
    G = temp.dot(r_mat_inv)
    return G
    
class EdgeSampler(object):
    def __init__(self, train_edges, train_edge_false, batch_size, remain_delet=True, shuffle=True):
        self.shuffle = shuffle
        self.index = 0
        self.index_false = 0
        self.pos_edge = train_edges
        self.neg_edge = train_edge_false
        self.id_index = list(range(train_edges.shape[0]))
        self.data_len = len(self.id_index)
        self.remain_delet = remain_delet
        self.batch_size = batch_size
        if self.shuffle:
            self._shuffle()

    def __iter__(self):
        return self

    def _shuffle(self):
        random.shuffle(self.id_index)

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.remain_delet:
            if self.index + self.batch_size > self.data_len:
                self.index = 0
                self.index_false = 0
                self._shuffle()
                raise StopIteration
            batch_index = self.id_index[self.index: self.index + self.batch_size]
            batch_x = self.pos_edge[batch_index]
            batch_y = self.neg_edge[batch_index]
            self.index += self.batch_size

        else:
            if self.index >= self.data_len:
                self.index = 0
                raise StopIteration
            end_ = min(self.index + self.batch_size, self.data_len)
            batch_index = self.id_index[self.index: end_]
            batch_x = self.pos_edge[batch_index]
            batch_y = self.neg_edge[batch_index]
            self.index += self.batch_size
        return np.array(batch_x), np.array(batch_y)

def loss_function_entropysample(pos_logit, neg_logit, b_xent, loss_type='entropy'):
    pos_logit = pos_logit.view(-1, 1)
    neg_logit = neg_logit.view(pos_logit.shape[0], -1)
    if loss_type == 'entropy':
        logits = torch.cat([pos_logit, neg_logit], dim=1)
        labels = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)], dim=1)
        cost = b_xent(logits, labels)
    else:
        cost = torch.mean(torch.nn.functional.softplus(neg_logit - pos_logit))
    return cost

def emb_get_roc_scoreGNN(net, hyper_node_G, hyper_H, dataloader_val, device):
    net.eval()
    preds = []
    preds_neg = []
    while True:
        try:
            pos_edge, neg_edge = dataloader_val.next()
        except StopIteration:
            break
        pos_src_, pos_dst_ = zip(*pos_edge)
        neg_src_, neg_dst_ = zip(*neg_edge)
        pos_src = torch.LongTensor(pos_src_).to(device)
        pos_dst = torch.LongTensor(pos_dst_).to(device)
        neg_src = torch.LongTensor(neg_src_).to(device)
        neg_dst = torch.LongTensor(neg_dst_).to(device)
        src_emb, dst_emb, src_neg_emb, dst_neg_emb, loss_recon = net(hyper_node_G, pos_src, pos_dst, neg_src, neg_dst, hyper_H)

        pos_logit, neg_logit = net.pred_logits(src_emb, dst_emb,
                                            src_neg_emb, dst_neg_emb)
        
        pos_logit = torch.sigmoid(pos_logit)
        neg_logit = torch.sigmoid(neg_logit)
        pos_logit = pos_logit.data.cpu().numpy().reshape(-1)
        neg_logit = neg_logit.data.cpu().numpy().reshape(-1)
        preds.extend(pos_logit.tolist())
        preds_neg.extend(neg_logit.tolist())

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)).tolist(), np.zeros(len(preds_neg)).tolist()])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score
def get_roc_scoreGNN(net, features, adj, dataloader_val, device):
    net.eval()
    preds = []
    preds_neg = []
    while True:
        try:
            pos_edge, neg_edge = dataloader_val.next()
        except StopIteration:
            break
        pos_src_, pos_dst_ = zip(*pos_edge)
        neg_src_, neg_dst_ = zip(*neg_edge)
        pos_src = torch.LongTensor(pos_src_).to(device)
        pos_dst = torch.LongTensor(pos_dst_).to(device)
        neg_src = torch.LongTensor(neg_src_).to(device)
        neg_dst = torch.LongTensor(neg_dst_).to(device)
        src_emb, dst_emb, src_neg_emb, dst_neg_emb = net(features, adj, pos_src, pos_dst, neg_src, neg_dst)

        pos_logit, neg_logit = net.pred_logits(src_emb, dst_emb,
                                                 src_neg_emb, dst_neg_emb)
        pos_logit = torch.sigmoid(pos_logit)
        neg_logit = torch.sigmoid(neg_logit)
        pos_logit = pos_logit.data.cpu().numpy().reshape(-1)
        neg_logit = neg_logit.data.cpu().numpy().reshape(-1)
        preds.extend(pos_logit.tolist())
        preds_neg.extend(neg_logit.tolist())

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)).tolist(), np.zeros(len(preds_neg)).tolist()])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def output_nodeemb(hyper_H, net, hyper_node_G, device):
    net.eval()
    node_emb = net.get_emb(hyper_H, hyper_node_G)
    return node_emb

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges_net_lp(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))
    # num_val = int(np.floor(edges.shape[0] / 5.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    num_node = adj.shape[0]
    sample_size = int((len(train_edges) * 1.0) / num_node)
    sample_size = sample_size + 10

    neg_list = []
    all_candiate = set(range(adj.shape[0]))
    for i in range(0, num_node):
        adj_csr = adj.tocsr()
        adj_csr = sp.csr_matrix(adj_csr)
        non_zeros = adj_csr.getrow(i).nonzero()[1]
        neg_candi = np.array(list(all_candiate.difference(set(non_zeros))))
        if len(neg_candi) >= sample_size:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=False)
        elif len(neg_candi) == 0:
            pass
        else:
            neg_candi = neg_candi

        neg_candi = [[i, j] for j in neg_candi]
        # print('len_: %d' % len(neg_candi))
        neg_list.extend(neg_candi)
    # neg_list = np.array(neg_list)

    train_edges_false = np.array(random.sample(neg_list, len(train_edges)))

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false
def mask_test_edges_net(adj):
     # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    ##这段代码的整体作用是从邻接矩阵中随机抽取10%的边作为测试集，5%的边作为验证集，剩余的边作为训练集。
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    num_node = adj.shape[0]
    sample_size = int((len(train_edges) * 1.0) / num_node)
    sample_size = sample_size + 10

    ##为每个节点生成一组负样本（不存在边的节点对），并将这些负样本添加到neg_list中。
    neg_list = []
    all_candiate = set(range(adj.shape[0]))

    for i in range(0, num_node):
        adj_csr = adj.tocsr()
        adj_csr = sp.csr_matrix(adj_csr)
        non_zeros = adj_csr.getrow(i).nonzero()[1]
        neg_candi = np.array(list(all_candiate.difference(set(non_zeros))))
        if len(neg_candi) >= sample_size:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=False)
        elif len(neg_candi) == 0:
            pass
        else:
            neg_candi = neg_candi

        neg_candi = [[i, j] for j in neg_candi]
        neg_list.extend(neg_candi)

    train_edges_false = np.array(random.sample(neg_list, len(train_edges)))

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if ismember([idx_i, idx_j], test_edges):
            continue
        if ismember([idx_j, idx_i], test_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_edges_classify(adj, num_node):
    # Function to build test set with 10% positive links
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    # original_adj edges
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, val_edge_idx, axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    adj = adj.multiply(adj_train)

    assert len(adj.data) == len(adj_train.data)
    sample_size = int((len(train_edges) * 1.0) / num_node)
    sample_size = sample_size + 10

    neg_list = []
    all_candiate = set(range(adj.shape[0]))
    for i in range(0, num_node):
        adj_csr = adj.tocsr()
        adj_csr = sp.csr_matrix(adj_csr)
        non_zeros = adj_csr.getrow(i).nonzero()[1]
        neg_candi = np.array(list(all_candiate.difference(set(non_zeros))))
        if len(neg_candi) >= sample_size:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=False)
        elif len(neg_candi) == 0:
            pass
        else:
            neg_candi = neg_candi

        neg_candi = [[i, j] for j in neg_candi]
        neg_list.extend(neg_candi)

    train_edges_false = np.array(random.sample(neg_list, len(train_edges)))
    val_edges_false = np.array(random.sample(neg_list, len(val_edges)))
    test_edges = []
    test_edges_false = []
    return adj, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

#######SVM##############
def test_classify(feature, labels, args):
    shape = len(labels.shape)
    if shape == 2:
        labels = np.argmax(labels, axis=1)
    f1_mac = []
    f1_mic = []
    accs = []
    kf = KFold(n_splits=5, random_state=args.seed, shuffle=True)
    for train_index, test_index in kf.split(feature):
        train_X, train_y = feature[train_index], labels[train_index]
        test_X, test_y = feature[test_index], labels[test_index]
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)

        micro = f1_score(test_y, preds, average='micro')
        macro = f1_score(test_y, preds, average='macro')
        acc = accuracy(preds, test_y)
        f1_mac.append(macro)
        f1_mic.append(micro)
        accs.append(acc)
    f1_mic = np.array(f1_mic)
    f1_mac = np.array(f1_mac)
    accs = np.array(accs)
    f1_mic = np.mean(f1_mic)
    f1_mac = np.mean(f1_mac)
    accs = np.mean(accs)
    print('Testing based on svm: ',
          'f1_micro=%.4f' % f1_mic,
          'f1_macro=%.4f' % f1_mac,
          'acc=%.4f' % accs)
    return f1_mic, f1_mac, accs

def accuracy(preds, labels):
    correct = (preds == labels).astype(float)
    correct = correct.sum()
    return correct / len(labels)

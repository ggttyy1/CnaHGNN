import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = torch.mm(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = torch.spmm(G, x)
        return x
    
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class CustomLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomLinear, self).__init__()
        # 定义自定义权重和偏置
        self.weight = nn.Parameter(torch.ones(input_dim, output_dim))
    
    def forward(self, x, Hyper_H):
        # 计算线性变换
        W = self.weight*Hyper_H
        return torch.matmul(W, x)
    
class CnaHGNN(nn.Module):
    def __init__(self, nfeat, nattri, num_nodes, nlayer, dropout, hid1, hid2, pos_weight, act = 'relu'):
        super(CnaHGNN, self).__init__()
        self.num_attri = nattri
        self.num_nodes = num_nodes
        self.latent_dim = nfeat
        self.dropout = dropout
        self.ratio = 1
        self.agg = 'add'

        hgc_node_layer = []
        for i in range(nlayer):
            if i%2 == 0:
                hgc_node_layer.append(HGNN_conv(2*self.latent_dim, self.latent_dim))
            else:
                hgc_node_layer.append(HGNN_conv(self.latent_dim, self.latent_dim))
        self.hgc_node = nn.ModuleList(hgc_node_layer)

        self.nlayer = nlayer
        self.recon_pre = InnerProductDecoder(beta = pos_weight)
        self.agg_model = CustomLinear(self.num_nodes,self.num_attri)

        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.PReLU()

        #############
        n_layer = (nlayer)*(nlayer)
        self.mlp1 = nn.Linear(self.latent_dim * n_layer, hid1)
        self.mlp2 = nn.Linear(hid1, hid2)
        self.mlp3 = nn.Linear(hid2, 1)
        ###############

        self.embedding_attri = torch.nn.Embedding(
            num_embeddings=self.num_attri, embedding_dim=self.latent_dim)
        self.embedding_node = torch.nn.Embedding(
            num_embeddings=self.num_nodes, embedding_dim=self.latent_dim)
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def __dropout(self, graph):
        graph = self.__dropout_x(graph)
        return graph

    def __dropout_x(self, x):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + self.dropout
        # random_index = random_index.int().bool()
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    def forward(self, hyper_node_G, pos_src, pos_dst, neg_src, neg_dst, Hyper_attr_H):
        if self.training:
            hyper_node_G = self.__dropout(hyper_node_G)
        

        node_emb = []
        src_emb = []
        dst_emb = []
        src_neg_emb = []
        dst_neg_emb = []


        #2层属性超图
        attr_x = self.embedding_attri.weight
        node_x = self.embedding_node.weight
        loss_recon = self.recon_pre(attr_x, node_x, Hyper_attr_H)
        
        for i in range(self.nlayer):
            if i%2 ==0:
                node_x = torch.cat([node_x, torch.mm(Hyper_attr_H.T, attr_x)], dim=1)
                node_x = self.hgc_node[i](node_x, hyper_node_G)
                node_emb.append(F.normalize(node_x, p=2, dim=1))
            else:
                node_x = self.hgc_node[i](node_x, hyper_node_G)
                node_emb.append(F.normalize(node_x, p=2, dim=1))


        for i,node in enumerate(node_emb):
            src_emb.append(node[pos_src])
            dst_emb.append(node[pos_dst])
            src_neg_emb.append(node[neg_src])
            dst_neg_emb.append(node[neg_dst])


        return src_emb, dst_emb, src_neg_emb, dst_neg_emb, loss_recon
    
    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer
    
    def cross_layer(self, src_x, dst_x):
        bi_layer = self.bi_cross_layer(src_x, dst_x)
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer
    
    def compute_logits(self, emb):
        emb = self.mlp1(emb)
        emb = self.act(emb)
        emb = self.mlp2(emb)
        emb = self.act(emb)
        preds = self.mlp3(emb)
        return preds
    
    def pred_logits(self, src_emb, dst_emb, src_neg_emb, dst_neg_emb):
        emb_pos = self.cross_layer(src_emb, dst_emb)
        emb_neg = self.cross_layer(src_neg_emb, dst_neg_emb)
        logits_pos = self.compute_logits(emb_pos)
        logits_neg = self.compute_logits(emb_neg)
        return logits_pos, logits_neg

    
    def get_emb(self, hyper_H, hyper_node_G):

        node_feat = self.embedding_node.weight
        attr_feat = self.embedding_attri.weight
        node_feat = torch.cat([node_feat, torch.spmm(hyper_H.T, attr_feat)], dim=1)
        node_feat = self.hgc_node[0](node_feat, hyper_node_G)

        node_feat = node_feat.data.cpu().numpy()
        return node_feat

    def reg_loss(self):
        reg_loss = (1 / 2) * (self.embedding_node.weight.norm(2).pow(2) +
                              self.embedding_attri.weight.norm(2).pow(2) / float(self.num_nodes + self.num_attri))
        return reg_loss

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self, beta, act=F.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.act = act
        self.beta = beta
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, attr_x, node_x, Hyper_attr_H):
        attr_x = self.dropout(attr_x)
        node_x = self.dropout(node_x)
        outputs = torch.mm(attr_x, node_x.t())
        outputs = self.act(outputs)
        loss_recon = recon_loss(Hyper_attr_H, outputs, beta_positive = self.beta)
        # x = tf.reshape(x, [-1])
        return loss_recon
    
def recon_loss(y_true, y_pred, beta_positive=2, beta_negative=0.1):
    # 正样本掩码 (假设正样本的标签是 1)
    pos_mask = (y_true == 1).float()
    
    # 负样本掩码 (假设负样本的标签是 0)
    neg_mask = (y_true == 0).float()
    
    # 计算误差
    mse_loss = nn.MSELoss(reduction='none')(y_true, y_pred)
    
    # 正负样本加权
    pos_loss = beta_positive * (mse_loss * pos_mask).mean()
    neg_loss = beta_negative * (mse_loss * neg_mask).mean()
    
    # 返回加权后的总损失
    total_loss = pos_loss + neg_loss
    return total_loss


class MymodelTune(nn.Module):
    def __init__(self, nfeat, nnode, nattri, nlayer, dropout, drop, hid1=512, hid2=128, act='relu'):
        super(MymodelTune, self).__init__()
        self.latent_dim = nfeat
        self.decoder = InnerProductDecoder(nfeat, dropout)
        self.dropout = dropout
        self.nlayer = nlayer
        self.drop = drop
        self.hid1 = hid1
        self.hid2 = hid2
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.PReLU()

        layer = []
        for i in range(self.nlayer):
            layer.append(HGNN_conv(nfeat, nfeat))
        self.gc1 = nn.ModuleList(layer)
        self.num_node = nnode
        self.num_attri = nattri
        n_layer = 1
        self.mlp1 = nn.Linear( n_layer * self.latent_dim, self.hid1)
        # self.mlp2 = nn.Linear(self.hid1, 1, bias=False)
        self.mlp2 = nn.Linear(self.hid1, self.hid2)
        self.mlp3 = nn.Linear(self.hid2, 1, bias=True)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self):
        torch.nn.init.normal_(self.embedding_node.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_attri.weight, std=0.1)
    def __dropout(self, graph):
        graph = self.__dropout_x(graph)
        return graph

    def __dropout_x(self, x):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + self.dropout
        # random_index = random_index.int().bool()
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    def forward(self, x1, adj, pos_src, pos_dst, neg_src, neg_dst):
        src_emb = []
        dst_emb = []
        src_neg_emb = []
        dst_neg_emb = []
        src_emb.append(F.normalize(x1[pos_src], p=2, dim=1))
        dst_emb.append(F.normalize(x1[pos_dst], p=2, dim=1))
        src_neg_emb.append(F.normalize(x1[neg_src], p=2, dim=1))
        dst_neg_emb.append(F.normalize(x1[neg_dst], p=2, dim=1))

        
        return src_emb, dst_emb, src_neg_emb, dst_neg_emb

    def comute_hop_emb(self, src_adj, dst_adj, src_neg_adj, dst_neg_adj):
        if self.training:
            if self.drop:
                src_adj = self.__dropout(src_adj)
                dst_adj = self.__dropout(dst_adj)
                src_neg_adj = self.__dropout(src_neg_adj)
                dst_neg_adj = self.__dropout(dst_neg_adj)
        x1 = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)

        src_emb_2 = self.gc2(x1, src_adj)
        dst_emb_2 = self.gc2(x1, dst_adj)
        src_neg_emb_2 = self.gc2(x1, src_neg_adj)
        dst_neg_emb_2 = self.gc2(x1, dst_neg_adj)
        return [src_emb_2], [dst_emb_2], [src_neg_emb_2], [dst_neg_emb_2]

    def get_emb(self, x1, node_index, adj):
        node_emb = []
        node_emb.append(F.normalize(x1[node_index], p=2, dim=1))
        for i, layer in enumerate(self.gc1):
            x1 = layer(x1, adj)
            node_emb.append(F.normalize(x1[node_index], p=2, dim=1))

        return node_emb

    def get_emb2(self, adj):
        xx = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)
        node_emb = self.gc2(xx, adj)
        return node_emb

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def cross_layer(self, src_x, dst_x):
        bi_layer = self.bi_cross_layer(src_x, dst_x)
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def compute_logits(self, emb):
        emb = self.mlp1(emb)
        emb = self.act(emb)
        emb = self.mlp2(emb)
        emb = self.act(emb)
        preds = self.mlp3(emb)
        return preds

    def pred_logits(self, src_emb, dst_emb, src_neg_emb, dst_neg_emb):
        emb_pos = self.cross_layer(src_emb, dst_emb)
        emb_neg = self.cross_layer(src_neg_emb, dst_neg_emb)
        logits_pos = self.compute_logits(emb_pos)
        logits_neg = self.compute_logits(emb_neg)
        return logits_pos, logits_neg

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)


    def reg_loss(self):
        reg_loss = (1 / 2) * (self.embedding_node.weight.norm(2).pow(2) +
                              self.embedding_attri.weight.norm(2).pow(2) / float(self.num_node + self.num_attri))
        return reg_loss
    

import os
from tqdm import tqdm
import argparse
import numpy as np
import torch
from utils import *
from model import *
import torch.optim as optim
import time
from data import *
# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=str, default='0', help='specify cuda devices')
parser.add_argument('--dataset', type=str, default="cora",
                    help='Dataset to use.')
parser.add_argument('--model_type', type=str, default="cnahgnn",
                    help='Dataset to use.')
parser.add_argument('--mode', type=str, default="train",
                    help='Dataset to use.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--nlayer', type=int, default=1, help='layer of HGNN.')
parser.add_argument('--ratio', type=int, default=1, help='loss ratio.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--activate', type=str, default="relu",
                    help='relu | prelu')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lambda_reg', type=float, default=1e-6,
                   help='weight of l2')
parser.add_argument('--trade_weight', type=float, default=0.8,
                    help='trade_off parameters).')
parser.add_argument('--hid1', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--hid2', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dim', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--use_cpu', type=int, default=0,
                    help='Use attribute or not')
parser.add_argument('--loss_type', type=str, default="entropy",
                    help='entropy | BPR')
parser.add_argument('--patience', type=int, default=50,
                    help='Use attribute or not')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--drop', type=int, default=1,
                    help='Indicate whether drop out or not')
parser.add_argument('--save_num', type=int, default=20,
                    help='Use attribute or not')

def train_embed(hyper_H, hyper_node_G, dataloader, dataloader_val, dataloader_test, device, args,
                              pos_weight, norm, save_path):
    '''
    hyper_H:关联矩阵
    hyper_node_G:超图
    dataloader:训练集
    dataloader_val:验证集
    dataloader_test:测试集
    device:设备
    args:参数
    pos_weight:正样本权重
    norm:归一化系数
    '''
    lambda_reg = args.lambda_reg  # 正则化系数，需根据实验调优
    num_attri = hyper_H.shape[0]
    num_nodes = hyper_H.shape[1]
    model = CnaHGNN(nfeat = args.dim,
                nattri = num_attri,
                num_nodes = num_nodes,
                nlayer = args.nlayer, 
                dropout=args.dropout, 
                hid1=args.hid1, 
                hid2=args.hid2,
                pos_weight=pos_weight,
                act=args.activate)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss()
    model.to(device)
    hyper_node_G = hyper_node_G.to(device)
    hyper_H = hyper_H.to(device)
    max_auc = 0.0
    max_ap = 0.0
    best_epoch = 0
    cnt_wait = 0
    auc_list = []
    ap_list = []
    for epoch in tqdm(range(args.epochs)):
        steps = 0
        epoch_loss = 0.0
        model.train()
        while True:
            try:
                pos_edge, neg_edge = dataloader.next()
            except StopIteration:
                break
            pos_src_, pos_dst_ = zip(*pos_edge)
            neg_src_, neg_dst_ = zip(*neg_edge)
            pos_src = torch.LongTensor(pos_src_).to(device)
            pos_dst = torch.LongTensor(pos_dst_).to(device)
            neg_src = torch.LongTensor(neg_src_).to(device)
            neg_dst = torch.LongTensor(neg_dst_).to(device)
            src_emb, dst_emb, src_neg_emb, dst_neg_emb, loss_recon = model(
                                                               hyper_node_G=hyper_node_G, 
                                                               pos_src = pos_src, 
                                                               pos_dst= pos_dst, 
                                                               neg_src = neg_src, 
                                                               neg_dst = neg_dst, 
                                                               Hyper_attr_H=hyper_H)
            
            pos_logit, neg_logit = model.pred_logits(src_emb, dst_emb, src_neg_emb, dst_neg_emb)

            loss_train = loss_function_entropysample(pos_logit, neg_logit, b_xent, loss_type=args.loss_type)
            # loss_train = loss_train + lambda_reg * loss_recon
            loss_train = loss_train + norm * loss_recon + lambda_reg*model.reg_loss()
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            epoch_loss += loss_train.item()
            #print('--> Epoch %d Step %5d loss: %.3f' % (epoch + 1, steps + 1, loss_train.item()))
            steps += 1
        auc_train, ap_train = emb_get_roc_scoreGNN(model, hyper_node_G, hyper_H, dataloader, device)
        auc_, ap_ = emb_get_roc_scoreGNN(model, hyper_node_G, hyper_H, dataloader_val, device)
        auc_list.append(auc_)
        ap_list.append(ap_)
        if auc_ > max_auc:
            max_auc = auc_
            max_ap = ap_
            best_epoch = epoch
            cnt_wait = 0
        else:
            cnt_wait += 1

        print('Epoch %d / %d' % (epoch, args.epochs),
              'current_best_epoch: %d' % best_epoch,
              'train_loss: %.4f' % (epoch_loss / steps),
              'train_auc: %.4f' % auc_train,
              'train_ap: %.4f' % ap_train,
              'valid_acu: %.4f' % auc_,
              'valid_ap: %.4f' % ap_)

        if cnt_wait == args.patience:
            print('Early stopping!')
            break      
    print('!!! Training finished',
          'best_epoch: %d' % best_epoch,
          'best_auc: %.4f' % max_auc,
          'best_ap: %.4f' % max_ap)
    auc_test, ap_test = emb_get_roc_scoreGNN(model, hyper_node_G, hyper_H, dataloader_test, device)
    print(auc_test, ap_test)
    emb = output_nodeemb(hyper_H, model, hyper_node_G, device)
    return emb, epoch

def finetune_GCLWeightGNN(features, device, args):
    print('---> Loading dataset for gcn...')
    # adj, _, labels, idx_train, idx_val, idx_test = load_citation(args.dataset)
    if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
        _, adj, labels, idx_train, idx_val, idx_test = load_citationANEWeight(args.dataset)
    elif args.dataset == 'ACM':
        adj, _, labels = load_undirected_map(args.dataset)
        labels = labels.reshape(-1)
    else:
        adj, _ = load_AN(args.dataset)
    print('--->Generate train/valid links for unsupervised learning...')
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_net_lp(
        adj)
    adj = adj_train

    features = sparse_mx_to_torch_sparse_tensor(features).float().to_dense()

    #adj_norm = add_self_loops(adj)
    #adj_norm = adj_norm.tolil()
    #adj_norm = adj_norm.tocsr()
    #adj_norm = generate_G_from_H(adj_norm)
    #adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm).float()
    adj_norm = []
    dataloader = EdgeSampler(train_edges, train_edges_false, args.batch_size)
    dataloader_val = EdgeSampler(val_edges, np.array(val_edges_false), args.batch_size, remain_delet=False)
    dataloader_test = EdgeSampler(test_edges, np.array(test_edges_false), args.batch_size, remain_delet=False)
    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    save_path = "./weights/LP_%s_" % args.model_type + args.dataset + '_%d_' % args.dim + '_%d_' % args.hid1\
                + '_%d_' % args.hid2 + '{}'.format(args.trade_weight) + '.pth'
    auc_test, ap_test = train_GCLGNN(features, adj_norm, dataloader, dataloader_val, dataloader_test, save_path, device, args,
                              pos_weight, norm)
    return auc_test, ap_test

def train_GCLGNN(features, hyper_adj, dataloader, dataloader_val, dataloader_test, save_path, device, args, pos_weight, norm):
    num_node = features.shape[0]
    num_attri = features.shape[1]
    model = MymodelTune(nfeat=num_attri,
                                nnode=num_node,
                                nattri=num_attri,
                                nlayer=args.nlayer,
                                dropout=args.dropout,
                                drop=args.drop,
                                hid1=args.hid1,
                                hid2=args.hid2,
                                act=args.activate)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = pos_weight.to(device)
    b_xent = nn.BCEWithLogitsLoss()
   # hyper_adj = hyper_adj.to(device)
    model.to(device)
    features = features.to(device)
    max_auc = 0.0
    max_ap = 0.0
    best_epoch = 0
    cnt_wait = 0
    auc_list = []
    ap_list = []
    auc_test_list = []
    ap_test_list = []
    for epoch in tqdm(range(args.epochs)):
        steps = 0
        epoch_loss = 0.0
        model.train()
        while True:
            try:
                pos_edge, neg_edge = dataloader.next()
            except StopIteration:
                break
            pos_src_, pos_dst_ = zip(*pos_edge)
            neg_src_, neg_dst_ = zip(*neg_edge)
            pos_src = torch.LongTensor(pos_src_).to(device)
            pos_dst = torch.LongTensor(pos_dst_).to(device)
            neg_src = torch.LongTensor(neg_src_).to(device)
            neg_dst = torch.LongTensor(neg_dst_).to(device)
            src_emb, dst_emb, src_neg_emb, dst_neg_emb = model(features, hyper_adj, pos_src, pos_dst, neg_src, neg_dst)
            pos_logit, neg_logit = model.pred_logits(src_emb, dst_emb, src_neg_emb, dst_neg_emb)
            loss_train = loss_function_entropysample(pos_logit, neg_logit, b_xent, loss_type=args.loss_type)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            epoch_loss += loss_train.item()
           # print('--> Epoch %d Step %5d loss: %.3f' % (epoch + 1, steps + 1, loss_train.item()))
            steps += 1
        auc_train, ap_train = get_roc_scoreGNN(model, features, hyper_adj, dataloader, device)
        auc_, ap_ = get_roc_scoreGNN(model, features, hyper_adj, dataloader_val, device)
        auc_test, ap_test = get_roc_scoreGNN(model, features, hyper_adj, dataloader_test, device)
        auc_test_list.append(auc_test)
        ap_test_list.append(ap_test)
        auc_list.append(auc_)
        ap_list.append(ap_)
        if auc_ > max_auc:
            max_auc = auc_
            max_ap = ap_
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            cnt_wait += 1

        print('Epoch %d / %d' % (epoch, args.epochs),
              'current_best_epoch: %d' % best_epoch,
              'train_loss: %.4f' % (epoch_loss / steps),
              'train_auc: %.4f' % auc_train,
              'train_ap: %.4f' % ap_train,
              'valid_acu: %.4f' % auc_,
              'valid_ap: %.4f' % ap_)

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    print('!!! Training finished',
          'best_epoch: %d' % best_epoch,
          'best_auc: %.4f' % max_auc,
          'best_ap: %.4f' % max_ap)
    print('auc and ap list:', auc_list)
    print('auc_test and ap_test list:', auc_test_list)
    model.load_state_dict(torch.load(save_path))
    auc_test, ap_test = get_roc_scoreGNN(model, features, hyper_adj, dataloader_test, device)
    return auc_test, ap_test

if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        cuda_name = 'cuda:' + args.cuda
        device = torch.device(cuda_name)
        print('--> Use GPU %s' % args.cuda)
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")
        print("--> No GPU")

    if args.use_cpu:
        device = torch.device("cpu")
    
    print('---> Loading %s dataset...' % args.dataset)
    if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
    # if args.dataset == 'cora':
        features, adj, labels, idx_train, idx_val, idx_test = load_citationANEWeight(args.dataset)
    elif args.dataset == 'ACM':
        adj, features, labels = load_undirected_map(args.dataset)
        labels = labels.reshape(-1)
    else:
        adj, features = load_AN(args.dataset)
    # features, adj, labels, idx_train, idx_val, idx_test = load_citationANEWeight(args.dataset)

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_net(
        adj)

    hyper_H = features.T
    adj_norm = adj_train
    num_node = adj_train.shape[0]

###############
    adj_norm = add_self_loops(adj_norm)
    adj_norm = adj_norm.tolil()
    adj_norm = adj_norm.tocsr()
    adj_norm = generate_G_from_H(adj_norm)
    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm).float()

    hyper_H = hyper_H.tolil()
    hyper_H = hyper_H.tocsr()
    hyper_H = sparse_mx_to_torch_sparse_tensor(hyper_H).float().to_dense()
################

    dataloader = EdgeSampler(train_edges, train_edges_false, args.batch_size)
    dataloader_val = EdgeSampler(val_edges, np.array(val_edges_false), args.batch_size, remain_delet=False)
    dataloader_test = EdgeSampler(test_edges, np.array(test_edges_false), args.batch_size, remain_delet=False)
    pos_weight = torch.Tensor([float(features.shape[0] * features.shape[1] - features.sum()) / features.sum()])
    pos_weight = pos_weight.item()
    norm = features.shape[0] * features.shape[1] / float((features.shape[0] * features.shape[1] - features.sum()) * 2)
    print('pos_weight:', pos_weight)
    print('norm:', norm)
    save_path = "./weights/linkwet_%s_" % args.model_type + args.dataset + '%d' % args.nlayer + '_%d_' % args.dim + '_%d_' % args.hid1 + '_%d_' % args.hid2 + '{}'.format(args.trade_weight)+ '.pth'
    auc_mean = []
    ap_mean = []
    for _ in range(5):
        node_embed, epoch = train_embed(hyper_H, adj_norm, dataloader, dataloader_val, dataloader_test, device, args,
                              pos_weight, norm, save_path)
        new_features = sp.csr_matrix(node_embed)
        auc_test, ap_test = finetune_GCLWeightGNN(new_features, device, args)
        auc_mean.append(auc_test)
        ap_mean.append(ap_test)
        print('!!! Link prediction results for ',
          'auc_gcl: %.4f' % (auc_test),
          'ap_gcl: %.4f' % ap_test
          )
    print('!!! Finish')
    print('!!! Mean auc_gcl: %.4f' % (np.mean(auc_mean)))
    print('!!! Mean ap_gcl: %.4f' % (np.mean(ap_mean)))
    print('list auc', auc_mean)
    print('list ap', ap_mean)
    auc = np.mean(auc_mean)
    ap = np.mean(ap_mean)
    with open("LP_RESULT.txt", "a") as file:  # "a" 模式用于追加写入
        file.write(f"Dataset: {args.dataset},5 epoch, nlayer: {args.nlayer}, auc: {auc} ap: {ap}\n")

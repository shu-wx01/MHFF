import os
import argparse
import yaml
import numpy as np
import random
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, recall_score, precision_score, accuracy_score
import matplotlib.pyplot as plt

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon')
#     parser.add_argument('--gamma', type=float, default=1)
#     parser.add_argument('--C', type=int, default=1)
#     parser.add_argument('--K', type=int, default=1)
    args_input = parser.parse_args()
    config_path = '../config/'+args_input.dataset+'.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    args = argparse.Namespace(**config)
#     args.gamma = args_input.gamma
#     args.C = args_input.C
#     args.K = args_input.K
    print('----------------------------------')
    print('              args')
    print('----------------------------------')
    print(f'dataset:\t{args.dataset}')
    print(f'seed:\t{args.seed}')
    print(f'epoch:\t{args.epoch}')
    print(f'early_stop:\t{args.early_stop}')
    print(f'lr:\t{args.lr}')
    print(f'weigth_decay:{args.weight_decay}')
    print(f'gamma:\t{args.gamma}')
    print(f'C:\t{args.C}')
    print(f'K:\t{args.K}')
    print(f'intra_dim:\t{args.intra_dim}')
    print(f'dropout:\t{args.dropout}')
    print(f'cuda:\t{args.cuda}')
    print('----------------------------------')
    return args

class EarlyStop():
    def __init__(self, early_stop, if_more=True) -> None:
        self.best_eval = 0
        self.best_epoch = 0
        self.if_more = if_more
        self.early_stop = early_stop
        self.stop_steps = 0
    
    def step(self, current_eval, current_epoch):
        do_stop = False
        do_store = False
        if self.if_more:
            if current_eval > self.best_eval:
                self.best_eval = current_eval
                self.best_epoch = current_epoch
                self.stop_steps = 1
                do_store = True
            else:
                self.stop_steps += 1
                if self.stop_steps >= self.early_stop:
                    do_stop = True
        else:
            if current_eval < self.best_eval:
                self.best_eval = current_eval
                self.best_epoch = current_epoch
                self.stop_steps = 1
                do_store = True
            else:
                self.stop_steps += 1
                if self.stop_steps >= self.early_stop:
                    do_stop = True
        return do_store, do_stop

def conf_gmean(conf):
    tn, fp, fn, tp = conf.ravel()
    return (tp*tn/((tp+fn)*(tn+fp)))**0.5
def prob2pred(prob, threshhold=0.5):
    pred = np.zeros_like(prob, dtype=np.int32)
    pred[prob >= threshhold] = 1
    pred[prob < threshhold] = 0
    return pred
def evaluate(labels, logits, result_path = ''):
    probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    if len(result_path)>0:
        np.save(result_path+'_result_preds', preds)
        np.save(result_path+'_result_probs', probs)
    conf = confusion_matrix(labels, preds)
    recall = recall_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    auc = roc_auc_score(labels, probs)
    gmean = conf_gmean(conf)
    return f1_macro, auc, gmean, recall

def hinge_loss(labels, scores):
    margin = 1
    ls = labels*scores
    
    loss = F.relu(margin-ls)
    loss = loss.mean()
    return loss

def normalize(mx):
    """
        Row-normalize sparse matrix
        Code from https://github.com/williamleif/graphsage-simple/
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def count_neighbors(graph, label):
    num_nodes = graph.num_nodes()

    # 创建结果列表
    same_count_1st = []
    different_count_1st = []

    same_count_2st = []
    different_count_2st = []

    for node in range(num_nodes):
        if label[node] == 0:
            continue

        # 获取节点的一阶邻居
        neighbors = graph.successors(node)
        # 获取节点的二阶邻居
        neighbors_2nd = []

        # 计算邻居中标签相同和不同的个数
        same_count = 0
        different_count = 0
        for neighbor in neighbors:
            neighbors_2nd.extend(graph.successors(neighbor))

            if label[node] == label[neighbor]:
                same_count += 1
            else:
                different_count += 1

        same_count_1st.append(same_count)
        different_count_1st.append(different_count)

        # 去除重复的邻居
        neighbors_2nd = list(set(neighbors_2nd))
        same_count = 0
        different_count = 0
        for neighbor in neighbors_2nd:
            if label[node] == label[neighbor]:
                same_count += 1
            else:
                different_count += 1

        same_count_2st.append(same_count)
        different_count_2st.append(different_count)
    fraud_per_1st = []
    fraud_per_2st = []
    for i in range(len(same_count_1st)):
        fraud_per_1st.append(same_count_1st[i] / (different_count_1st[i] + same_count_1st[i]))
        fraud_per_2st.append(same_count_2st[i] / (different_count_2st[i] + same_count_2st[i]))
    fraud_per_1st_avg = sum(fraud_per_1st) / len(fraud_per_1st)
    fraud_per_2st_avg = sum(fraud_per_2st) / len(fraud_per_2st)
    print("end")

def csr_matrix_to_edge_index(csr_matrix):
    # 获取 CSR 矩阵的行数和列数
    num_rows, num_cols = csr_matrix.shape

    # 将 CSR 矩阵转换为 COO 矩阵
    coo_matrix = csr_matrix.tocoo()

    # 提取 COO 矩阵中的非零元素的行索引和列索引
    row_indices = coo_matrix.row
    col_indices = coo_matrix.col

    # 创建 edge_index 列表
    edge_index = np.vstack((row_indices, col_indices))

    return edge_index

def draw_curve(metric='', dataset=''):
    epochs = [j for j in range(100)]
    file_names = ['model_results_GCN','model_results_GAT','model_results_PCGNN','model_results_ESGNN','model_results_BWGNN','model_results_MHFF']
    # file_names = ['model_results','model_results_GCN', 'model_results_GAT', 'model_results_PCGNN', 'model_results_BWGNN','model_results_ESGNN']
    labels_dict = {0: 'GCN', 1: 'GAT', 2: 'PCGNN',
                   3: 'ESGNN', 4: 'BWGNN', 5: 'MHFF'}
    colors_dict = {0: '#C7C1DE', 1: '#B5CE4E', 2: '#97D0C5', 3: '#7FABD1', 4: '#BD7795', 5: '#F39865'}
    if metric == 'loss':
        loss_results = []
        for file_name in file_names:
            file_name = '../result/' + dataset + '/' + file_name + '_loss' + '_' + dataset + '.txt'
            loss_list = []
            with open(file_name, "r") as file:
                for line in file:
                    line = line.rstrip('\n')
                    parts = line.split(",")

                    loss = parts[1].split(':')[1]
                    loss_list.append(float(loss))
            loss_results.append(loss_list)
        for idx in range(len(file_names)):
            plt.plot([j for j in epochs], loss_results[idx], label=labels_dict[idx], color=colors_dict[idx])

        plt.xlim((0, 100))
        plt.ylim((0, 10))
        plt.xticks(np.arange(0, 100, step=25))
        plt.yticks(np.arange(0, 10, step=2))
        # # plt.grid()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.savefig('./loss_curve' + '_' + dataset + '.pdf', dpi=1200)
        plt.show()
    elif metric == 'auc':
        auc_results = []
        for file_name in file_names:
            file_name = '../result/' + dataset + '/' + file_name + '_' + dataset + '.txt'
            auc_list = []
            with open(file_name, "r") as file:
                for line in file:
                    line = line.rstrip('\n')
                    parts = line.split(",")

                    auc = parts[1].split(':')[1]
                    auc_list.append(float(auc)*100)
            auc_results.append(auc_list)
        for idx in range(len(file_names)):
            plt.plot([j for j in epochs], auc_results[idx], label=labels_dict[idx], color=colors_dict[idx])

        plt.xlim((0, 100))
        plt.ylim((40, 100))
        plt.xticks(np.arange(0, 100, step=25))
        plt.yticks(np.arange(40, 100, step=10))
        # plt.grid()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('auc')
        plt.savefig('./auc_curve' + '_' + dataset + '.pdf', dpi=1200)
        plt.show()
    elif metric == 'recall':
        recall_results = []
        for file_name in file_names:
            file_name = '../result/' + dataset + '/' + file_name + '_' + dataset + '.txt'
            recall_list = []
            with open(file_name, "r") as file:
                for line in file:
                    line = line.rstrip('\n')
                    parts = line.split(",")

                    recall = parts[2].split(':')[1]
                    recall_list.append(float(recall)*100)
            recall_results.append(recall_list)
        for idx in range(len(file_names)):
            plt.plot([j for j in epochs], recall_results[idx], label=labels_dict[idx], color=colors_dict[idx])
        plt.xlim((0, 100))
        plt.ylim((20, 100))
        plt.xticks(np.arange(0, 100, step=25))
        plt.yticks(np.arange(20, 100, step=10))
        # # plt.grid()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('recall')
        plt.savefig('./recall_curve' + '_' + dataset + '.pdf', dpi=1200)
        plt.show()
    elif metric == 'f1':
        f1_results = []
        for file_name in file_names:
            file_name = '../result/' + dataset + '/' + file_name + '_' + dataset + '.txt'
            f1_list = []
            with open(file_name, "r") as file:
                for line in file:
                    line = line.rstrip('\n')
                    parts = line.split(",")

                    f1 = parts[3].split(':')[1]
                    f1_list.append(float(f1)*100)
            f1_results.append(f1_list)
        for idx in range(len(file_names)):
            plt.plot([j for j in epochs], f1_results[idx], label=labels_dict[idx], color=colors_dict[idx])
        plt.xlim((0, 100))
        plt.ylim((10, 100))
        plt.xticks(np.arange(0, 100, step=25))
        plt.yticks(np.arange(10, 100, step=10))
        # plt.grid()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('f1')
        plt.savefig('./f1_curve' + '_' + dataset + '.pdf', dpi=1200)
        plt.show()


if __name__ == '__main__':
    draw_curve('f1', 't-finance')


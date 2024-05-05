import argparse
import os
import sys
import dgl
import torch
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
from dataset import load_nc_dataset
import pickle

def generate_edges_labels(edges, labels, train_idx):
    row, col = edges
    edge_labels = []
    edge_train_mask = []
    train_idx = set(train_idx)
    for i, j in zip(row, col):
        i = i.item()
        j = j.item()
        if labels[i] == labels[j]:
            edge_labels.append(1)
        else:
            edge_labels.append(-1)
        if i in train_idx and j in train_idx:
            edge_train_mask.append(1)
        else:
            edge_train_mask.append(0)
    edge_labels = torch.Tensor(edge_labels).long()
    edge_train_mask = torch.Tensor(edge_train_mask).bool()
    return edge_labels, edge_train_mask


if __name__ == '__main__':
    dataset_path = '../data/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gambling')
    args = parser.parse_args()
    print('**********************************')
    print(f'Generate {args.dataset}')
    print('**********************************')
    if args.dataset == 't-finance':
        dataset = load_nc_dataset(args.dataset)
        edge = dataset.graph["edge_index"]
        g = dgl.graph((edge[0], edge[1]))
        in_degrees = g.in_degrees()
        out_degrees = g.out_degrees()
        labels = dataset.label

        positive_degrees = []
        negative_degrees = []
        # 获取目标节点的一阶邻居节点
        target_nodes = [i for i in range(dataset.graph['num_nodes'])]
        for node in target_nodes:
            neighbor_nodes = g.successors(node)
            # 计算与目标节点具有相同标签的邻居节点数量
            same_label_count = (labels[neighbor_nodes] == labels[node]).sum()
            diff_label_count = (labels[neighbor_nodes] != labels[node]).sum()
            positive_degrees.append(same_label_count)
            negative_degrees.append(diff_label_count)
        positive_degrees = torch.stack(positive_degrees)
        negative_degrees = torch.stack(negative_degrees)

        dataset.graph['g'] = g
        dataset.graph['degrees'] = positive_degrees
        dataset.graph['positive_degrees'] = positive_degrees
        dataset.graph['negative_degrees'] = negative_degrees

        # 保存数据结构
        with open('../data/t-finance_processed_data.pkl', 'wb') as file:
            pickle.dump(dataset, file)

    elif args.dataset == 'gambling':
        dataset = load_nc_dataset(args.dataset)
        edge = dataset.graph["edge_index"]
        g = dgl.graph((edge[0], edge[1]))
        in_degrees = g.in_degrees()
        out_degrees = g.out_degrees()
        labels = dataset.label

        positive_degrees = []
        negative_degrees = []
        # 获取目标节点的一阶邻居节点
        target_nodes = [i for i in range(dataset.graph['num_nodes'])]
        for node in target_nodes:
            neighbor_nodes = g.successors(node)
            # 计算与目标节点具有相同标签的邻居节点数量
            same_label_count = (labels[neighbor_nodes] == labels[node]).sum()
            diff_label_count = (labels[neighbor_nodes] != labels[node]).sum()
            positive_degrees.append(same_label_count)
            negative_degrees.append(diff_label_count)
        positive_degrees = torch.stack(positive_degrees)
        negative_degrees = torch.stack(negative_degrees)

        dataset.graph['g'] = g
        dataset.graph['degrees'] = positive_degrees
        dataset.graph['positive_degrees'] = positive_degrees
        dataset.graph['negative_degrees'] = negative_degrees

        # 保存数据结构
        with open('../data/gambling_processed_data.pkl', 'wb') as file:
            pickle.dump(dataset, file)

    print('***************endl****************')



from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import parse
import torch.optim as optim
from model import *
from utils import *
import torch.nn.functional as F
import pickle
import dgl

import warnings
warnings.filterwarnings('ignore')
random.seed(42)

if __name__ == '__main__':

    args = parse.args
    setup_seed(args.seed)
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device('cuda' if use_cuda else 'cpu')
    args.device = device

    # 加载数据结构
    with open('../data/' + args.dataset + '_processed_data.pkl', 'rb') as file:
        dataset = pickle.load(file)

    # load splits
    if args.dataset == 't-finance':
        index = list(range(dataset.label.size()[0]))
        idx_train, idx_test, y_train, y_test = train_test_split(index, dataset.label.squeeze(1).numpy(),
                                                                stratify=dataset.label.squeeze(1).numpy(),
                                                                test_size=0.60, random_state=2, shuffle=True)
        split_dic = dict()
        split_dic["trn_idx"] = idx_train
        split_dic["tst_idx"] = idx_test
    elif args.dataset == 'gambling':
        index = list(range(dataset.label.size()[0]))
        idx_train, idx_test, y_train, y_test = train_test_split(index, dataset.label.squeeze(1).numpy(),
                                                                stratify=dataset.label.squeeze(1).numpy(),
                                                                test_size=0.60, random_state=2, shuffle=True)
        split_dic = dict()
        split_dic["trn_idx"] = idx_train
        split_dic["tst_idx"] = idx_test
    else:
        raise Exception('wrong dataname')
    trn_idx, tst_idx = np.array(split_dic["trn_idx"]), np.array(split_dic["tst_idx"])
    trn_idx = trn_idx.astype(np.int64)
    tst_idx = tst_idx.astype(np.int64)

    # pre-processing
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    # to-cuda
    trn_idx = torch.from_numpy(trn_idx).to(device)
    tst_idx = torch.from_numpy(tst_idx).to(device)
    dataset.label = dataset.label.to(device)
    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

    # create graph
    edge = dataset.graph["edge_index"]
    g = dataset.graph['g'].to(device)

    # train model
    print('Start training model...')
    model = SplitGNN(args, dataset)
    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    batch_size = 1024 * 2
    trn_dataloader = dgl.dataloading.DataLoader(
        g, trn_idx, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False)

    tst_dataloader = dgl.dataloading.DataLoader(
        g, tst_idx, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False)

    for t in range(args.epoch):
        print('train start, epoch:{}'.format(t))
        pbar = tqdm(total=trn_idx.size(0), ncols=80)
        model.train()
        # 将数据划分为批次并进行训练
        loss_all = 0
        for e, (input_nodes, output_nodes, blocks) in enumerate(trn_dataloader):
            batch_size = output_nodes.size(0)
            block = blocks[0]

            # degree_train = dataset.graph['degrees'][output_nodes]
            # input_y = dataset.label.squeeze(1)[output_nodes].numpy()
            # lf_train = (input_y.sum() - len(input_y)) * input_y + len(input_y)
            # smp_prob = np.array(degree_train) / lf_train
            # output_nodes = random.choices(output_nodes, weights=smp_prob, k=3)
            # output_nodes = torch.tensor(output_nodes)

            loss = model.loss(block, dataset.graph["node_feat"][input_nodes], batch_size, input_nodes, output_nodes, device)
            loss_all += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(batch_size)
            print("batch: {:d}, batch-loss: {:.4f}".format(e, loss.item()))

        loss_result = 'epoch:{}, loss:{:.4}'.format(t, loss_all)
        with open('../result/model_results_loss' + '_' + args.dataset  +'.txt', 'a') as file:
            file.write(loss_result + '\n')

        print('test start, epoch:{}'.format(t))
        pbar = tqdm(total=tst_idx.size(0), ncols=80)
        # 将数据划分为批次并进行测试
        true_label_all = []
        tst_pre_all = []
        tst_probs_all = []
        for e, (input_nodes, output_nodes, blocks) in enumerate(tst_dataloader):
            with torch.no_grad():
                model.eval()
                batch_size = output_nodes.size(0)
                block = blocks[0]
                test_labels = dataset.label[tst_idx].cpu().numpy()
                logits = model(block, dataset.graph["node_feat"][input_nodes], batch_size, input_nodes, output_nodes)
                preds = logits[:batch_size].argmax(1).type_as(dataset.label[output_nodes])
                tst_pre_all.append(preds)
                true_label_all.append(dataset.label[output_nodes])
                tst_probs_all.append(F.softmax(logits[:batch_size], dim=1)[:,1])
                pbar.update(batch_size)

        auc = roc_auc_score(torch.cat(true_label_all, dim=0).cpu().numpy(), torch.cat(tst_probs_all, dim=0).cpu().numpy())
        recall = recall_score(torch.cat(true_label_all, dim=0).cpu().numpy(), torch.cat(tst_pre_all, dim=0).cpu().numpy(), average='macro')
        f1_macro = f1_score(torch.cat(true_label_all, dim=0).cpu().numpy(), torch.cat(tst_pre_all, dim=0).cpu().numpy(), average='macro')
        pre = precision_score(torch.cat(true_label_all, dim=0).cpu().numpy(), torch.cat(tst_pre_all, dim=0).cpu().numpy(), average='macro')
        acc = accuracy_score(torch.cat(true_label_all, dim=0).cpu().numpy(), torch.cat(tst_pre_all, dim=0).cpu().numpy())
        print(metrics.classification_report(torch.cat(true_label_all, dim=0).cpu().numpy(), torch.cat(tst_pre_all, dim=0).cpu().numpy(), digits=4))
        result = 'epoch:{}, auc:{:.4}, recall:{:.4}, f1:{:.4}, pre:{:.4}, acc:{:.4}'.format(t, auc, recall, f1_macro, pre, acc)
        print(result)
        with open('../result/model_results' + '_' + args.dataset  +'.txt', 'a') as file:
                file.write(result + '\n')
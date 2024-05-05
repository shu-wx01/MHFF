import torch
import sympy
import torch.nn as nn
import torch.nn.functional as F
import scipy
import dgl.function as fn

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas

class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 relation_aware,
                 dataset,
                 thetas,
                 K,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=True):
        super(PolyConv, self).__init__()
        self._theta = thetas
        self._k = len(self._theta[0])
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.relation_aware = relation_aware
        self.K = K
        self.linear = nn.Linear(in_feats*len(thetas), out_feats, bias)
        self.linear1 = nn.Linear(in_feats*len(thetas), out_feats, bias)
        self.transh = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin
        self.dataset = dataset

    def forward(self, graph, dataset, feat, batch_size, input_nodes, output_nodes):
        def unnLaplacian(feat, D_invsqrt, graph, flag, batch_size):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            h = feat
            h = h * D_invsqrt
            graph.srcdata.update({'h': h})
            graph.dstdata.update({'h': h[:batch_size]})

            if flag==0:
                graph.update_all(fn.copy_u('feat', 'm'), fn.sum('m', 'h'))
            elif flag==1:
                graph.update_all(self.message_positive, fn.sum('p', 'h'))
            elif flag==2:
                graph.update_all(self.message_negative, fn.sum('n', 'h'))
            return feat - graph.ndata.pop('h')['_N'] * D_invsqrt

        graph.apply_edges(self.sign_edges)
        graph.apply_edges(self.judge_edges)
        with graph.local_scope():


            # graph.update_all(message_func=fn.copy_e('positive', 'positive'), reduce_func=self.positive_reduce)
            # graph.update_all(message_func=fn.copy_e('negative', 'negative'), reduce_func=self.negative_reduce)

            in_degrees = dataset.graph['degrees'][input_nodes]
            positive_in_degrees = dataset.graph['positive_degrees'][input_nodes]
            negative_in_degrees = dataset.graph['negative_degrees'][input_nodes]

            D_invsqrt = torch.pow(in_degrees.float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            D_invsqrt_positive = torch.pow(positive_in_degrees.float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            D_invsqrt_negative = torch.pow(negative_in_degrees.float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            
            hs_o = []
            hs_p = []
            hs_n = []
            
            transh = self.transh(feat)
            
            for theta in self._theta:
                h_o = theta[0]*feat
                
                for k in range(1, self._k):
                    feat = unnLaplacian(feat, D_invsqrt, graph, 0, batch_size)
                    h_o += theta[k]*feat
                hs_o.append(h_o)
            
            feat = graph.ndata['feat']['_N']
            for theta in self._theta[0:self.K+1]:
                h_p = theta[0]*feat
                
                for k in range(1, self._k):
                    feat = unnLaplacian(feat, D_invsqrt_positive, graph, 1, batch_size)
                    h_p += theta[k]*feat
                hs_p.append(h_p)

            feat = graph.ndata['feat']['_N']
            for theta in self._theta[self.K+1:]:
                h_n = theta[0]*feat

                for k in range(1, self._k):
                    feat = unnLaplacian(feat, D_invsqrt_negative, graph, 2, batch_size)
                    h_n += theta[k]*feat
                hs_n.append(h_n)
        
            hs_o = torch.cat(hs_o, dim=1)
            if self.K != len(self._theta) - 1 and self.K != -1:
                hs_p = torch.cat(hs_p, dim=1)
                hs_n = torch.cat(hs_n, dim=1) 
                hs_pn = torch.cat([hs_p, hs_n], dim=1)
            elif self.K == -1:
                hs_pn = torch.cat(hs_n, dim=1)
            else:
                hs_pn = torch.cat(hs_p, dim=1)

        if self.lin:
            hs_o = self.linear(hs_o)
            hs_o = self.activation(hs_o)
            hs_pn = self.linear1(hs_pn)
            hs_pn = self.activation(hs_pn)

        return hs_o, hs_pn, transh

    def sign_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        # score = self.relation_aware(src, dst)
        # return {'sign':torch.sign(score)}
        src_label = self.dataset.label[edges.src['_ID']]
        dst_label = self.dataset.label[edges.dst['_ID']]
        sign = torch.where(src_label == dst_label, 1, -1)
        return {'sign': sign}

    def judge_edges(self, edges):
        return {'positive': (edges.data['sign'] >= 0).float(), 'negative': (edges.data['sign'] < 0).float()}

    def positive_reduce(self, nodes):
        return {'positive_in_degree': nodes.mailbox['positive'].sum(1)}

    def negative_reduce(self, nodes):
        return {'negative_in_degree': nodes.mailbox['negative'].sum(1)}
    
    def message_positive(self, edges):
        mask = (edges.data['sign'] >= 0).float().view(-1, 1)
        masked_src_feats = edges.src['h'] * mask
        return {'p': masked_src_feats}
    
    def message_negative(self, edges):
        mask = (edges.data['sign'] < 0).float().view(-1, 1)
        masked_src_feats = edges.src['h'] * mask
        return {'n': masked_src_feats}
    
    
class RelationAware(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.d_liner = nn.Linear(input_dim, output_dim)
        self.f_liner = nn.Linear(3*output_dim, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, dst):
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        diff = src-dst
        e_feats = torch.cat([src, dst, diff], dim=1)
        e_feats = self.dropout(e_feats)
        score = self.f_liner(e_feats).squeeze()
        score = self.tanh(score)
        return score


class MultiRelationSplitGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dataset, dropout, thetas, K, if_sum=False):
        super().__init__()
        self.dataset = dataset
        self.liner = nn.Linear(output_dim*3, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relation_aware = RelationAware(input_dim, output_dim, dropout)
        self.minelayer = PolyConv(input_dim, output_dim, self.relation_aware, self.dataset, thetas, K, lin=True)
        self.dropout = nn.Dropout(dropout)


    def forward(self, g, h, batch_size, input_nodes, output_nodes):
        hs_o = []
        hs_lh = []
        hs_trans = []

        h_o, h_lh, h_trans = self.minelayer(g, self.dataset, h, batch_size, input_nodes, output_nodes)
        hs_o.append(h_o)
        hs_lh.append(h_lh)
        hs_trans.append(h_trans)

        h = torch.cat([torch.cat(hs_o, dim=1),torch.cat(hs_lh, dim=1), torch.cat(hs_trans, dim=1)], dim=1)
        h = self.dropout(h)
        h = self.liner(h)
        return h
    
    def loss(self, g, h, batch_size, input_nodes, output_nodes):
        with g.local_scope():
            g.srcdata.update({"feat": h})
            g.dstdata.update({"feat": h[:batch_size]})

            agg_h = self.forward(g, h, batch_size, input_nodes, output_nodes)

            g.apply_edges(self.score_edges)
            edges_score = g.edata['score']

            # edge_train_mask = g.edges['homo'].data['train_mask'].bool()
            # edge_train_label = g.edges['homo'].data['label'][edge_train_mask]
            # edge_train_pos = edge_train_label == 1
            # edge_train_neg = edge_train_label == -1
            # edge_train_pos_index = edge_train_pos.nonzero().flatten().detach().cpu().numpy()
            # edge_train_neg_index = edge_train_neg.nonzero().flatten().detach().cpu().numpy()
            # edge_train_pos_index = np.random.choice(edge_train_pos_index, size=len(edge_train_neg_index))
            # index = np.concatenate([edge_train_pos_index, edge_train_neg_index])
            # index.sort()
            # edge_train_score = edges_score[edge_train_mask]
            # # hinge loss
            # edge_diff_loss = hinge_loss(edge_train_label[index], edge_train_score[index])

            return agg_h, 0
            
    def score_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        score = self.relation_aware(src, dst)
        return {'score':score}


class SplitGNN(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()
        self.input_dim = dataset.graph['node_feat'].shape[1]  # nodes['company'] for FDCompCN
        self.intra_dim = args.hidden_channels
        self.gamma = args.gamma
        self.C = args.C
        self.K = args.K
        self.n_class = args.n_class
        self.thetas = calculate_theta2(d=self.C)
        self.mine_layer = MultiRelationSplitGNNLayer(self.intra_dim, self.intra_dim, dataset, args.dropout, self.thetas, self.K)
        self.linear = nn.Linear(self.input_dim, self.intra_dim)
        self.linear2 = nn.Linear(self.intra_dim, self.n_class)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU()
        self.dataset = dataset

    def forward(self, g, feats, batch_size, input_nodes, output_nodes):
        h = self.linear(feats)

        g.srcdata.update({"feat": h})
        g.dstdata.update({"feat": h[:batch_size]})

        h = self.mine_layer(g, h, batch_size, input_nodes, output_nodes)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return h 
    
    def loss(self, g, feats, batch_size, input_nodes, output_nodes, device):
        h = self.linear(feats)
        h, edge_loss = self.mine_layer.loss(g, h, batch_size, input_nodes, output_nodes)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        
        # train_mask = g.ndata['train_mask'].bool()
        # train_label = g.ndata['label'][train_mask]
        # train_pos = train_label == 1
        # train_neg = train_label == 0
        #
        # pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
        # neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
        # neg_index = np.random.choice(neg_index, size=len(pos_index), replace=False)
        # index = np.concatenate([pos_index, neg_index])
        # index.sort()

        weight = (1 - self.dataset.label[output_nodes]).sum().item() / self.dataset.label[output_nodes].sum().item()
        model_loss = F.cross_entropy(h[:batch_size], self.dataset.label[output_nodes].squeeze(1), weight=torch.tensor([1., weight]).to(device))
        loss = model_loss + self.gamma*edge_loss
        return loss





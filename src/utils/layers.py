from math import sqrt
from torch import FloatTensor
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(2*out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(linear, self).__init__()
        self.weight = Parameter(FloatTensor(in_features, out_features))
        self.register_parameter('bias', None)
        stdv = 1. / sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, x):
        x = x.matmul(self.weight)
        return x

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, residual=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if not residual:
            self.residual = lambda x: 0
        elif (in_features == out_features):
            self.residual = lambda x: x
        else:
            # self.residual = linear(in_features, out_features)
            self.residual = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=5, padding=2)
    def reset_parameters(self):
        # stdv = 1. / sqrt(self.weight.size(1))
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.1)

    def forward(self, input, adj):
        # To support batch operations
        support = input.matmul(self.weight)
        output = adj.matmul(support)

        if self.bias is not None:
            output = output + self.bias
        if self.in_features != self.out_features and self.residual:
            input = input.permute(0,2,1)
            res = self.residual(input)
            res = res.permute(0,2,1)
            output = output + res
        else:
            output = output + self.residual(input)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

######################################################

class SimilarityAdj(Module):

    def __init__(self, in_features, out_features):
        super(SimilarityAdj, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight0 = Parameter(FloatTensor(in_features, out_features))
        self.weight1 = Parameter(FloatTensor(in_features, out_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / sqrt(self.weight0.size(1))
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, input, seq_len):
        # To support batch operations
        theta = torch.matmul(input, self.weight0)
        phi = torch.matmul(input, self.weight0)
        phi2 = phi.permute(0, 2, 1)
        sim_graph = torch.matmul(theta, phi2)

        theta_norm = torch.norm(theta, p=2, dim=2, keepdim=True)  # B*T*1
        phi_norm = torch.norm(phi, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = theta_norm.matmul(phi_norm.permute(0, 2, 1))
        sim_graph = sim_graph / (x_norm_x + 1e-20)

        output = torch.zeros_like(sim_graph)
        if seq_len is None:
            for i in range(sim_graph.shape[0]):
                tmp = sim_graph[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = F.softmax(adj2, dim=1)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = sim_graph[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = F.softmax(adj2, dim=1)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class DistanceAdj(Module):

    def __init__(self):
        super(DistanceAdj, self).__init__()
        self.sigma = Parameter(FloatTensor(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size, max_seqlen):
        # To support batch operations
        self.arith = np.arange(max_seqlen).reshape(-1, 1)
        dist = pdist(self.arith, metric='cityblock').astype(np.float32)
        self.dist = torch.from_numpy(squareform(dist)).to('cuda')
        self.dist = torch.exp(-self.dist / torch.exp(torch.tensor(1.)))
        self.dist = torch.unsqueeze(self.dist, 0).repeat(batch_size, 1, 1).to('cuda')
        return self.dist
    
if __name__ == '__main__':
    d = DistanceAdj()
    dist = d(1, 256).squeeze(0)
    print(dist.softmax(dim=-1))
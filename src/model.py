import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class ICRN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ICRN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data, data_b,data_b_1, id_new_value_old, id_new_value_old_1, lam, adj):
        x0, edge_index = data.x, data.edge_index
        edge_index1 = data_b.edge_index
        edge_index2 = data_b_1.edge_index

        x = F.dropout(x0, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        x_b_0 = x0[id_new_value_old]
        x_b = F.dropout(x_b_0, p=self.dropout, training=self.training)
        x_b = F.relu(self.lin1(x_b))
        x_b = F.dropout(x_b, p=self.dropout, training=self.training)
        x_b = self.lin2(x_b)

        x_b_1 = x0[id_new_value_old_1]
        x_b_1 = F.dropout(x_b_1, p=self.dropout, training=self.training)
        x_b_1 = F.relu(self.lin1(x_b_1))
        x_b_1 = F.dropout(x_b_1, p=self.dropout, training=self.training)
        x_b_1 = self.lin2(x_b_1)

        if self.dprate == 0.0:

            x = self.prop1(x, edge_index)
            x1 = self.prop1(x_b, edge_index1)
            x2 = self.prop1(x_b_1, edge_index2)
            x_mix = lam * x + (1 - lam) * x1
            x_mix1 = lam * x + (1 - lam) * x2
            bt_c = torch.mm(F.normalize(x_mix, dim=1), F.normalize(x_mix1, dim=1).t())
            bt_loss = torch.diagonal(bt_c).add(-1).pow(2).mean() + off_diagonal(bt_c).pow(2).mean()
            return F.log_softmax(x_mix, dim=1), bt_loss


        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            x1 = F.dropout(x_b, p=self.dprate, training=self.training)
            x1 = self.prop1(x1, edge_index1)

            x2 = F.dropout(x_b_1,p=self.dprate, training=self.training)
            x2 = self.prop1(x2, edge_index2)

            x_mix = lam * x + (1 - lam) * x1
            x_mix1 = lam * x + (1 - lam) *x2

            bt_c = torch.mm(F.normalize(x_mix, dim=0), F.normalize(x_mix1, dim=0).t())
            bt_loss = torch.diagonal(bt_c).add(-1).pow(2).mean() + off_diagonal(bt_c).pow(2).mean()

            return F.log_softmax(x_mix, dim=1), bt_loss

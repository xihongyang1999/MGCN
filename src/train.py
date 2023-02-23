import argparse
from dataset_utils import DataLoader
from utils import random_planetoid_splits
from model import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np
import copy

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def idNode(data, id_new_value_old):
    data = copy.deepcopy(data)
    data.x = None
    data.y[data.val_id] = -1
    data.y[data.test_id] = -1
    data.y = data.y[id_new_value_old]

    data.train_id = None
    data.test_id = None
    data.val_id = None

    id_old_value_new = torch.zeros(id_new_value_old.shape[0], dtype = torch.long)
    id_old_value_new[id_new_value_old] = torch.arange(0, id_new_value_old.shape[0], dtype = torch.long)
    row = data.edge_index[0]
    col = data.edge_index[1]
    row = id_old_value_new[row]
    col = id_old_value_new[col]
    data.edge_index = torch.stack([row, col], dim=0)

    return data

def shuffleData(data):
    data_copy = copy.deepcopy(data)
    id_new_value_old = np.arange(data_copy.num_nodes)
    # print(data_copy.train_id)
    train_id_shuffle = copy.deepcopy(data_copy.train_id)
    np.random.shuffle(train_id_shuffle)
    id_new_value_old[data_copy.train_id] = train_id_shuffle #id是新的，值是old，按照训练数据的下标使得节点序列里面
    data_copy = idNode(data_copy, id_new_value_old)

    return data_copy, id_new_value_old


def RunExp(args, dataset, data, data_b, data_b_1, id_new_value_old,id_new_value_old_1, adj, lam, Net, percls_trn, val_lb):

    def train(model, optimizer, data, data_b,data_b_1, id_new_value_old,id_new_value_old_1, lam, adj, dprate):
        model.train()
        optimizer.zero_grad()
        data_b = data_b.cuda()
        data_b_1 = data_b_1.cuda()
        out, bt_loss = model(data, data_b, data_b_1, id_new_value_old, id_new_value_old_1,lam,adj)
        nll = F.nll_loss(out[data.train_id], data.y[data.train_id]) * lam \
              + F.nll_loss(out[data.train_id], data_b.y[data.train_id]) * (1 - lam) + 0.5 * bt_loss
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data, data_b, data_b_1, id_new_value_old,id_new_value_old_1, lam, adj):
        model.eval()
        logits, accs, losses, preds = model(data, data_b, data_b_1, id_new_value_old, id_new_value_old_1, lam, adj), [], [], []

        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[0][mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            out, bt_loss = model(data, data_b, data_b_1, id_new_value_old,id_new_value_old_1, lam, adj)
            loss = F.nll_loss(out[data.train_id], data.y[data.train_id]) * lam \
              + F.nll_loss(out[data.train_id], data_b.y[data.train_id]) * (1 - lam) + 0.5 * bt_loss

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_planetoid_splits
    data, train_id, val_id, test_id = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'ICRN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []


    for epoch in range(args.epochs):


        train(model, optimizer, data, data_b, data_b_1, id_new_value_old, id_new_value_old_1, lam, adj, args.dprate)


        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data, data_b,data_b_1, id_new_value_old, id_new_value_old_1,lam,adj)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    return test_acc, best_val_acc, Gamma_0


setup_seed(2)
if __name__ == '__main__':
    Results0 = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--train_rate', type=float, default=0.025)
    parser.add_argument('--val_rate', type=float, default=0.025)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--dprate', type=float, default=0.7)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                            choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                            default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop',
                            choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', type=str, default='ICRN')

    args = parser.parse_args()
    gnn_name = args.net
    Net = ICRN

    dname = args.dataset
    dataset = DataLoader(dname)
    data = dataset[0]

    RPMAX = args.RPMAX
    Init = args.Init

    Gamma_0 = None
    alpha = args.alpha

    # data split
    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(val_rate*len(data.y)))
    TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
    print('True Label rate: ', TrueLBrate)
    permute_masks = random_planetoid_splits
    data, train_id, test_id, val_id = permute_masks(data, dataset.num_classes, percls_trn, val_lb)
    # ------------------------------------------------------------------------

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    data.train_id = train_id
    data.test_id = test_id
    data.val_id = val_id

    values = torch.tensor([1 for i in range(data.edge_index.shape[1])])
    adj = torch.sparse.FloatTensor(data.edge_index, values, torch.Size([data.x.shape[0], data.x.shape[0]])).to_dense().cuda()
    adj += torch.eye(adj.shape[0]).long().cuda()
    adj = adj.float()

    adj_train = copy.deepcopy(adj)
    adj = (data.y == data.y.unsqueeze(1)).float().cuda()
    adj_train[data.train_id, :][:, data.train_id] = adj[data.train_id, :][:, data.train_id]


    data_b, id_new_value_old = shuffleData(data)
    data_b1, id_new_value_old_1 = shuffleData(data)
    lam = 0.95
    for RP in tqdm(range(10)):
        test_acc, best_val_acc, Gamma_0 = RunExp(
                args, dataset, data, data_b, data_b1, id_new_value_old, id_new_value_old_1, adj_train, lam, Net, percls_trn, val_lb)

        Results0.append([test_acc, best_val_acc, Gamma_0])

    test_acc_mean, val_acc_mean, _ = np.mean(Results0, axis=0) * 100
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
    print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')


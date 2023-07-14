import sys
import torch
import argparse
import numpy as np
import os

from rmc import RMC, RMC_Sparse
from deeprobust.graph.defense import GCN
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.utils import preprocess, encode_onehot, get_train_val_test

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='meta',
        choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=0.15, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=1000, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=0.1, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of gcn loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--rank', type=int, default=100, help='the truncated rank for W')
parser.add_argument('--k', type=int, default=10, help='sample k*|E| entries for sparse version')
parser.add_argument('--sparse', action='store_true', default=False, help='whether use sparse version')
parser.add_argument('--multi_first', action='store_true', default=False, help='calculate the sample inner product if true')
parser.add_argument('--symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--type', type=int, default=0)

args = parser.parse_args()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    torch.cuda.manual_seed(args.seed)

np.random.seed(15)
data = Dataset(root='./dataset/nettack/', name=args.dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
 
if args.dataset == 'pubmed':
    # just for matching the results in the paper; seed details in https://github.com/ChandlerBang/Pro-GNN/issues/2
    idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0],
            val_size=0.1, test_size=0.8, stratify=encode_onehot(labels), seed=15)

if args.ptb_rate == 0:
    perturbed_adj = adj
    if args.attack == 'nettack':
        perturbed_data = PrePtbDataset(root='./dataset/nettack/',
            name=args.dataset,
            attack_method=args.attack,
            ptb_rate=1.0)
        idx_test = perturbed_data.get_target_nodes()
    args.attack = "no"
            
if args.attack == 'random':
    from deeprobust.graph.global_attack import Random
    # to fix the seed of generated random attack, you need to fix both np.random and random
    # you can uncomment the following code
    # import random; random.seed(args.seed)
    # np.random.seed(args.seed)
    attacker = Random()
    n_perturbations = int(args.ptb_rate * (adj.sum()//2))
    attacker.attack(adj, n_perturbations, type='add')
    perturbed_adj = attacker.modified_adj
    
if args.attack == 'meta' or args.attack == 'nettack':
    perturbed_data = PrePtbDataset(root='./dataset/nettack/',
            name=args.dataset,
            attack_method=args.attack,
            ptb_rate=args.ptb_rate)
    perturbed_adj = perturbed_data.adj
    if args.attack == 'nettack':
        idx_test = perturbed_data.get_target_nodes()

perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False)
features, labels = features.to(device), labels.to(device)

if args.rank > perturbed_adj.size(0):
    args.rank = 'full'

if args.sparse:
    e_num = torch.nonzero(perturbed_adj, as_tuple=True)[0].size(0)//2
    if 2*args.k*e_num > perturbed_adj.size(0)**2:
        args.sparse = False
        print('The sample size is bigger than the matrix size, thus the code swith to the dense version!!!', file=sys.stderr)
    elif 2*args.k*e_num*args.rank > perturbed_adj.size(0)**2:
        args.multi_first = True
        print('The size of inner sample matrix is too big, thus the code calculate L first!!!', file=sys.stderr)

if not args.sparse:
    perturbed_adj = perturbed_adj.to(device)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout, device=device).to(device)
if args.sparse:
    rmc = RMC_Sparse(model, perturbed_adj, args, device=device)
    rmc.fit(features, perturbed_adj, labels, idx_train, idx_val, idx_test, device)
    acc_test = rmc.test(features, labels, idx_val, idx_test)
    acc_val = rmc.best_val
else:
    rmc = RMC(model, perturbed_adj, args, device=device)
    rmc.fit(features, perturbed_adj, labels, idx_train, idx_val, idx_test, device)
    acc_test = rmc.test(features, labels, idx_val, idx_test)
    acc_val = rmc.best_val
print(acc_test)

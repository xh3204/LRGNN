import time
import torch
import numpy as np
import tensorly as tl
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F

from deeprobust.graph.utils import *
from deeprobust.graph.defense.pgd import PGD, prox_operators

class RMC(nn.Module):
    def __init__(self, model, adj, args, svd=False, device='cpu'):
        super(RMC, self).__init__()
        self.device = device
        self.args = args
        self.best_val = 0
        self.best_val_loss = 10
        self.best_adj = None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)
        
        if args.rank != 'full':
            if svd:
                U, S, V = torch.svd(adj)
                self.side = U[:, :self.args.rank].to(device)
            else:
                if args.type == 0:
                    tl.set_backend('pytorch')
                    U, S, V = tl.tenalg.svd_interface(adj.cpu(), n_eigenvecs=self.args.rank)
                    side= torch.FloatTensor(U).to(adj.device)
                elif args.type == 1:
                    side= torch.eye(adj.size(0), args.rank).to(adj.device)
                else:
                    side= torch.randn(adj.size(0), args.rank).to(adj.device)
        
        if args.rank == 'full':
            self.weight = torch.nn.parameter.Parameter(adj.to(device))
        else:
            self.weight = torch.nn.parameter.Parameter(
                (side.t()).matmul(adj).matmul(side).to(self.device)
            )
            self.side = side.to(self.device)
        self.S = torch.nn.parameter.Parameter(
            torch.zeros_like(adj, device=self.device)
        )
        
        self.optimizer_gcn = optim.Adam(self.model.parameters(),
                                   lr=self.args.lr,
                                   weight_decay=self.args.weight_decay)
        self.optimizer_adj = optim.SGD([{'params': self.weight}],
                                  momentum=0.9,
                                  lr=self.args.lr_adj)
        self.optimizer_sparse = optim.SGD([{'params': self.S}],
                                     momentum=0.9,
                                     lr=self.args.lr_adj)
        self.optimizer_nuclear = PGD([{'params': self.weight}],
                                proxs=[prox_operators.prox_nuclear_cuda],
                                lr=self.args.lr_adj, alphas=[self.args.beta])
        self.optimizer_l1 = PGD([{'params': self.S}],
                           proxs=[prox_operators.prox_l1],
                           lr=self.args.lr_adj, alphas=[self.args.alpha])
    
    def fit(self, features, adj, labels, idx_train, idx_val, idx_test, device):
        print('start training!!!')
        time_start = time.time()
        if self.args.debug:
            for epoch in range(self.args.epochs):
                self.epoch = epoch+1
                for _ in range(self.args.outer_steps):
                    self.train_adj(features, adj, labels, idx_train, idx_val, idx_test)

                for _ in range(self.args.inner_steps):
                    self.train_gcn(features, self.L_nor, labels, idx_train, idx_val, idx_test)
        else:
            for epoch in tqdm(range(self.args.epochs)):
                self.epoch = epoch+1
                for _ in range(self.args.outer_steps):
                    self.train_adj(features, adj, labels, idx_train, idx_val, idx_test)

                for _ in range(self.args.inner_steps):
                    self.train_gcn(features, self.L_nor, labels, idx_train, idx_val, idx_test)
        print('Finished, cost {:.2f} seconds!'.format(time.time()-time_start))
                
    def train_adj(self, features, adj, labels, idx_train, idx_val, idx_test):
        if self.epoch == 1:
            if self.args.rank != 'full':
                L_theta = torch.clamp(self.side.matmul(self.weight
                                                      ).matmul(self.side.t()),
                                      min=0, max=1)
            else:
                L_theta = torch.clamp(self.weight, min=0, max=1)
        else:
            L_theta = self.L
        self.optimizer_adj.zero_grad()
        loss_fro = torch.norm(adj-self.S-L_theta, p='fro')
        self.model.train()
        output = self.model(features, self.nor_adj(L_theta))
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_adj = loss_fro + self.args.gamma*loss_train

        loss_adj.backward()
        self.optimizer_adj.step()
        # 更新L的nuclear norm
        self.optimizer_nuclear.zero_grad()
        self.optimizer_nuclear.step()
        
        if self.args.rank != 'full':
            L_theta = self.side.matmul(self.weight).matmul(self.side.t())
        else:
            L_theta = self.weight

        # 更新S
        self.optimizer_sparse.zero_grad()
        loss_fro = torch.norm(adj-self.S-L_theta.detach(), p='fro')
        loss_fro.backward()
        self.optimizer_sparse.step()

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()
        
        L_theta = torch.clamp(L_theta, min=0, max=1)
        L_theta_nor = self.nor_adj(L_theta.detach())
        
        self.L = L_theta
        self.L_nor = L_theta_nor
        
        self.model.eval()
        output = self.model(features, L_theta_nor)
        loss_val =  F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        if acc_val > self.best_val:
            self.best_val = acc_val.item()
            self.best_adj = L_theta.detach()
            self.weights = deepcopy(self.model.state_dict())
        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val.item()
            self.best_adj = L_theta.detach()
            self.weights = deepcopy(self.model.state_dict())
        
        if self.args.debug:
            print('epoch: {}, loss_fro: {}, loss_train: {}, acc_val: {}, acc_test: {}'.format(self.epoch, loss_fro, loss_train, acc_val, acc_test))
    
    def train_gcn(self, features, adj, labels, idx_train, idx_val, idx_test):
        self.optimizer_gcn.zero_grad()
        self.model.train()
        output = self.model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer_gcn.step()

        self.model.eval()
        output = self.model(features, adj)
        acc_val = accuracy(output[idx_val], labels[idx_val])
        loss_val =  F.nll_loss(output[idx_val], labels[idx_val])

        if acc_val > self.best_val:
            self.best_val = acc_val.item()
            self.best_adj = self.L.detach()
            self.weights = deepcopy(self.model.state_dict())
        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val.item()
            self.best_adj = self.L.detach()
            self.weights = deepcopy(self.model.state_dict())
            
    def test(self, features, labels, idx_val, idx_test):
        print("\t=== testing ===")
        self.model.load_state_dict(self.weights)
        self.model.eval()
        adj = self.nor_adj(self.best_adj)
        output = self.model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        val_test = accuracy(output[idx_val], labels[idx_val])
        print("\tVal set results:",
              "accuracy= {:.4f}".format(val_test.item()))
        print("\tTest set results:",
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test
    
    def nor_adj(self, adj):
        if is_sparse_tensor(adj):
            adj_norm = normalize_sparse_tensor(adj)
        else:
            adj_norm = normalize_adj_tensor(adj)
        return adj_norm


class RMC_Sparse(RMC):
    def __init__(self, model, adj, args, svd=False, sparse=False, device='cpu'):
        super(RMC_Sparse, self).__init__(model, adj, args, svd=svd, device=device)
        # sample
        if sparse:
            adj_sparse = adj
        else:
            adj_sparse = adj.to_sparse()
        adj_indices = adj_sparse.indices()
        adj_size = int(adj_indices.size(1)/2)
        # 采样
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        sample_size = adj_size*args.k
        sample_indices = (torch.rand(2, sample_size)*adj.size(0)).type(torch.int64)
        # 去掉自环
        sample_indices = sample_indices[:, sample_indices[0] != sample_indices[1]]
        indices_ = torch.cat([sample_indices, sample_indices[[1,0], :]], dim=-1)
        values_ = torch.zeros(indices_.size(1))
        # 提取索引
        indices = torch.cat([adj_sparse.indices(), indices_], dim=-1).to(device)
        values = torch.cat([adj_sparse.values(), values_], dim=-1).to(device)
        self.adj_sparse_cuda = torch.sparse_coo_tensor(indices, values, adj.size()).coalesce()
        self.indices = self.adj_sparse_cuda.indices()
        self.values = self.adj_sparse_cuda.values()
        self.s_index, self.t_index = self.indices
        self.s_index_list, self.t_index_list = self.s_index.tolist(), self.t_index.tolist()
        self.S = torch.nn.parameter.Parameter(torch.zeros_like(self.adj_sparse_cuda.values(), device=device))
        self.optimizer_sparse = optim.SGD([{'params': self.S}],
                                     momentum=0.9,
                                     lr=self.args.lr_adj)
        self.optimizer_l1 = PGD([{'params': self.S}],
                           proxs=[prox_operators.prox_l1],
                           lr=self.args.lr_adj, alphas=[self.args.alpha])
    
    def fit(self, features, adj, labels, idx_train, idx_val, idx_test, device):
        print('start training!!!')
        time_start = time.time()
        if self.args.debug:
            for epoch in range(self.args.epochs):
                self.epoch = epoch+1
                for _ in range(self.args.outer_steps):
                    self.train_adj(features, adj, labels, idx_train, idx_val, idx_test)

                for _ in range(self.args.inner_steps):
                    self.train_gcn(features, self.L_sparse_nor, labels, idx_train, idx_val, idx_test)
        else:
            for epoch in tqdm(range(self.args.epochs)):
                self.epoch = epoch+1
                for _ in range(self.args.outer_steps):
                    self.train_adj(features, adj, labels, idx_train, idx_val, idx_test)

                for _ in range(self.args.inner_steps):
                    self.train_gcn(features, self.L_sparse_nor, labels, idx_train, idx_val, idx_test)
        print('Finished, cost {:.2f} seconds!'.format(time.time()-time_start))
    
    def train_adj(self, features, adj, labels, idx_train, idx_val, idx_test):
        if self.epoch == 1:
            if not self.args.multi_first:
                L_value = torch.clamp(torch.sum(self.side[self.s_index].matmul(self.weight
                                                                    )*self.side[self.t_index],
                                    dim=-1).reshape(1,-1).squeeze(), min=0, max=1)
            else:
                if self.args.rank != 'full':
                    L_value = torch.clamp((self.side @ self.weight @ self.side.t())[self.s_index_list, self.t_index_list],
                                          min=0, max=1)
                else:
                    L_value = torch.clamp(self.weight[self.s_index_list, self.t_index_list], min=0, max=1)
        else:
            L_value = self.L_value
        self.optimizer_adj.zero_grad()
        loss_fro = torch.norm(self.adj_sparse_cuda.values()-self.S-L_value, p='fro')
        self.model.train()
        L_sparse = torch.sparse_coo_tensor(self.indices, L_value, adj.size()).coalesce().requires_grad_(True)
        output = self.model(features, self.nor_adj(L_sparse))
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_adj = loss_fro + loss_train

        loss_adj.backward()
        self.optimizer_adj.step()
        # 更新L的nuclear norm
        self.optimizer_nuclear.zero_grad()
        self.optimizer_nuclear.step()
        
        if not self.args.multi_first:
            L_value = torch.sum(self.side[self.s_index].matmul(self.weight
                                                                    )*self.side[self.t_index],
                                    dim=-1).reshape(1,-1).squeeze()
        else:
            if self.args.rank != 'full':
                L_value = (self.side @ self.weight @ self.side.t())[self.s_index_list, self.t_index_list]
            else:
                L_value = self.weight[self.s_index_list, self.t_index_list]

        # 更新S
        self.optimizer_sparse.zero_grad()
        loss_fro = torch.norm(self.adj_sparse_cuda.values()-self.S-L_value.detach(), p='fro')
        loss_fro.backward()
        self.optimizer_sparse.step()

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()
        
        L_value = torch.clamp(L_value, min=0, max=1)
        L_sparse.values().data.copy_(L_value.detach())
        L_sparse_nor = self.nor_adj(L_sparse.detach())
        
        self.L_value = L_value
        self.L_sparse = L_sparse
        self.L_sparse_nor = L_sparse_nor
        
        self.model.eval()
        output = self.model(features, L_sparse_nor)
        loss_val =  F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        if acc_val > self.best_val:
            self.best_val = acc_val.item()
            self.best_adj = L_sparse.detach()
            self.weights = deepcopy(self.model.state_dict())
        
        if self.args.debug:
            print('epoch: {}, loss_fro: {}, loss_train: {}, acc_val: {}, acc_test: {}'.format(self.epoch, loss_fro, loss_train, acc_val, acc_test))
            
    def train_gcn(self, features, adj, labels, idx_train, idx_val, idx_test):
        self.optimizer_gcn.zero_grad()
        self.model.train()
        output = self.model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer_gcn.step()
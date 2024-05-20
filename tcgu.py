import numpy as np
import random
import time
import argparse
import time
import deeprobust.graph.utils as utils
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.transforms as T
import torch_sparse
import os 
import gc
import math
from collections import Counter

from utils import *
from torch_sparse import SparseTensor
from copy import deepcopy
from torch_geometric.utils import coalesce

from models.basicgnn import GCN as GCN_PYG, GIN as GIN_PYG, SGC as SGC_PYG, GraphSAGE as SAGE_PYG, GAT as GAT_PYG
from models.parametrized_adj import PGE

import warnings
warnings.filterwarnings('ignore')

class TCGU:
    
    def __init__(self,args) -> None:
        super(TCGU, self).__init__()
        self.args=args
        self.device=self.args.device
        self.root='.'
        
        self.load_data()
        self.split_train_test_val(self.args.train_ratio,self.args.test_ratio)
        
        self.determine_teacher_model()
        self.determine_embed_model()

        self.unlearning_request()
        self.get_unlearn_graph()
    
        print("Traning Teacher!")
        self.train_teacher()
        self.teacher_model.load_state_dict(torch.load(f'{self.root}/saved_model/teacher/{self.args.dataset}_{self.args.teacher_model}_{args.seed}.pt'))

        print("Evaluate Teacher!")
        self.eval_teacher()
        
        print("Initialize Syn Graph!")
        self.generate_labels_syn()
        self.init_graph_syn()
        print("Initialization Finished")
        
        # condense process
        self.run_load_syn()
        
        # initialize fine-tune setting
        print("Change to Fine-tuning mode")
        self.init_fine_tune()
        
        # Fine-tuning
        self.run_load_ft_syn()
        
        # Retrain on condensed Data
        print("Begining Retraining")
        self.train_model_on_cond_data()
        
    def load_data(self):

        data = get_dataset(self.args.dataset, self.args.normalize_features)
        adj, self.feat=utils.to_tensor(data.adj, data.features, device='cpu')
        self.num_nodes=self.feat.shape[0] 
        self.d=self.feat.shape[1]
        self.labels=torch.LongTensor(data.labels).to(self.device)
        self.nclass= int(self.labels.max()+1)
        
        self.edge_index=copy.deepcopy(data.edge_index)
        
        if utils.is_sparse_tensor(adj):
            adj = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj = utils.normalize_adj_tensor(adj)
        self.adj=SparseTensor(row=adj._indices()[0], col=adj._indices()[1],value=adj._values(), sparse_sizes=adj.size()).t() 
        
        self.norm_edge_index=adj._indices()
        self.norm_edge_weight=adj._values()
        
        del data
        del adj
        gc.collect()

    def unlearning_request(self):

        self.feat_unlearn = self.feat.clone() 
        self.edge_index_unlearn = self.edge_index.clone()
        edge_index = self.edge_index.numpy()
        unique_indices = np.where(edge_index[0] < edge_index[1])[0]

        if self.args.unlearn_task == 'node':
            
            unique_nodes = np.random.choice(len(self.idx_train),
                                            int(len(self.idx_train) * self.args.unlearn_ratio),
                                            replace=False)
          
            self.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)

        if self.args.unlearn_task == 'edge':
            remove_indices = np.random.choice(
                unique_indices,
                int(unique_indices.shape[0] * self.args.unlearn_ratio),
                replace=False)
            remove_edges = edge_index[:, remove_indices]
            unique_nodes = np.unique(remove_edges)

            self.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes, remove_indices)

        if self.args.unlearn_task == 'feature':
            unique_nodes = np.random.choice(len(self.idx_train),
                                            int(len(self.idx_train) * self.args.unlearn_ratio),
                                            replace=False)
            self.feat_unlearn[unique_nodes] = 0.

        self.temp_node = unique_nodes
        
    def update_edge_index_unlearn(self, delete_nodes, delete_edge_index=None):
        edge_index = self.edge_index.numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]

        if self.args.unlearn_task == 'edge':
            remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
        else:
            unique_edge_index = edge_index[:, unique_indices]
            delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                                np.isin(unique_edge_index[1], delete_nodes))
            remain_indices = np.logical_not(delete_edge_indices)
            remain_indices = np.where(remain_indices == True)

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[
            0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)
        remain_indices_not = unique_indices_not[
            sort_indices[np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)]]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        return torch.from_numpy(edge_index[:, remain_indices])
    
    def get_unlearn_graph(self):
   
        adj_unlearn=sp.csr_matrix((np.ones(self.edge_index_unlearn.shape[1]),
            (self.edge_index_unlearn[0], self.edge_index_unlearn[1])),shape=(self.num_nodes,self.num_nodes))

        if sp.issparse(adj_unlearn):
            adj_unlearn = sparse_mx_to_torch_sparse_tensor(adj_unlearn)
        else:
            adj_unlearn = torch.FloatTensor(adj_unlearn)
            
        if utils.is_sparse_tensor(adj_unlearn):
            adj_unlearn = utils.normalize_adj_tensor(adj_unlearn, sparse=True)
        else:
            adj_unlearn = utils.normalize_adj_tensor(adj_unlearn)
        self.adj_unlearn=SparseTensor(row=adj_unlearn._indices()[0], col=adj_unlearn._indices()[1],value=adj_unlearn._values(), sparse_sizes=adj_unlearn.size()).t() # 处理后的邻接矩阵
        
        if self.args.unlearn_task=='node':
            self.idx_train_unlearn=np.setdiff1d(self.idx_train,self.temp_node)
            self.labels_train_unlearn=self.labels[self.idx_train_unlearn]
        else:
            self.idx_train_unlearn=self.idx_train
            self.labels_train_unlearn=self.labels_train
        
        del adj_unlearn
        gc.collect()
    
    def split_train_test_val(self,train_ratio=0.7,test_ratio=0.2):

        num_nodes=len(self.labels)
    
        self.idx_train, self.idx_val, self.idx_test=split_train_test_val(num_nodes,train_ratio,test_ratio)

        self.labels_train, self.labels_val, self.labels_test=self.labels[self.idx_train], self.labels[self.idx_val], self.labels[self.idx_test]
    
    def determine_teacher_model(self):

        if self.args.model=='GCN':
            self.teacher_model = GCN_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers, norm='BatchNorm').to(self.device)
        elif self.args.model=='GAT':
            self.teacher_model = GAT_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers,norm='BatchNorm').to(self.device)
        elif self.args.model=='GIN':
            self.teacher_model = GIN_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers, norm='BatchNorm', act=self.args.activation).to(self.device)
        elif self.args.model=='SGC':
            self.teacher_model = SGC_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, nlayers=self.args.nlayers, sgc=True).to(self.device)

        if self.args.inference:
   
            self.inference_loader=NeighborSampler(self.adj,
                sizes=[-1], 
                batch_size=self.args.batch_size,
                num_workers=12, 
                return_e_id=False,
                num_nodes=len(self.labels),
                shuffle=False
            )
            
        print('Teacher Model: {}'.format(self.args.teacher_model))
        
    def determine_embed_model(self):

        if self.args.embed_model=='SGC':
            self.embed_model = SGC_PYG(nfeat=self.d, nhid=256, nclass=self.nclass, dropout=0, nlayers=self.args.nlayers, norm=None, sgc=True).to(self.device) 
        else:
            raise ValueError('Undefined Embedding Model')
        
        print('Embedding Model: {}'.format(self.args.embed_model))
    
    def determine_random_model(self):
        
        if self.args.model=='GCN':
            self.random_model = GCN_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers, norm='BatchNorm').to(self.device)
        elif self.args.model=='GAT':
            self.random_model = GAT_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers,norm='BatchNorm').to(self.device)
        elif self.args.model=='GIN':
            self.random_model = GIN_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers, norm='BatchNorm', act=self.args.activation).to(self.device)
        elif self.args.model=='SGC':
            self.random_model = SGC_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, nlayers=self.args.nlayers, sgc=True).to(self.device)
        
        if self.args.fs_mode=='tfs':
            k=random.choice([i for i in range(len(self.func_space))])
            self.random_model.load_state_dict(self.func_space[k])
    
        for name,parameters in self.random_model.named_parameters():
            parameters.requires_grad=False
    
    def train_teacher(self):
        
        start = time.perf_counter() 
        optimizer_origin=torch.optim.Adam(self.teacher_model.parameters(), lr=self.args.lr_teacher_model)

        best_val=0
        best_test=0
        for it in range(self.args.teacher_model_loop+1):
            self.teacher_model.train()
            optimizer_origin.zero_grad()
            output = self.teacher_model.forward(self.feat.to(self.device), self.adj.to(self.device))[self.idx_train]
            loss = F.nll_loss(output, self.labels_train)
            loss.backward()
            optimizer_origin.step()

            if(it%self.args.teacher_val_stage==0):
                if self.args.inference==True:
                    output = self.teacher_model.inference(self.feat, self.inference_loader, self.device)
                else:
                    output = self.teacher_model.predict(self.feat.to(self.device), self.adj.to(self.device))
                acc_train = utils.accuracy(output[self.idx_train], self.labels_train)
                acc_val = utils.accuracy(output[self.idx_val], self.labels_val)
                acc_test = utils.accuracy(output[self.idx_test], self.labels_test)
                print(f'Epoch: {it:02d}, '
                        f'Loss: {loss.item():.4f}, '
                        f'Train: {100 * acc_train.item():.2f}%, '
                        f'Valid: {100 * acc_val.item():.2f}% '
                        f'Test: {100 * acc_test.item():.2f}%')
                if(acc_val>best_val):
                    best_val=acc_val
                    best_test=acc_test
                    if self.args.save:
                        torch.save(self.teacher_model.state_dict(), f'{self.root}/saved_model/teacher/{self.args.dataset}_{self.args.teacher_model}_{self.args.seed}.pt')
            
        end = time.perf_counter()
        print("Best Test:", best_test)
        print('Time for Training Teacher Model:', round(end-start), 's')

        self.teacher_model.load_state_dict(torch.load(f'{self.root}/saved_model/teacher/{self.args.dataset}_{self.args.teacher_model}_{self.args.seed}.pt'))   
        
    def eval_teacher(self):
        if self.args.inference==True:
            output = self.teacher_model.inference(self.feat, self.inference_loader, self.device)
        else:
            output = self.teacher_model.predict(self.feat.to(self.device), self.adj.to(self.device))
        torch.save(output,f'{self.root}/saved_ours/embed_{self.args.dataset}_{self.args.teacher_model}_{self.args.seed}.pt')
        acc_test = utils.accuracy(output[self.idx_test], self.labels_test)
        print("Teacher model test set results:","accuracy= {:.4f}".format(acc_test.item()))
        
    def generate_labels_syn(self):

        counter = Counter(self.labels_train.cpu().numpy())
        num_class_dict = {}

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        labels_syn = []
        syn_class_indices = {}

        for ix, (c, num) in enumerate(sorted_counter):
            num_class_dict[c] = math.ceil(num * self.args.reduction_rate)
            syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

        self.labels_syn = torch.LongTensor(labels_syn).to(self.device)
        self.num_class_dict=num_class_dict
    
    def init_graph_syn(self):
   
        nnodes_syn = len(self.labels_syn)
        self.n = nnodes_syn
        
        self.feat_syn = nn.Parameter(torch.FloatTensor(self.n, self.d).to(self.device))
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size())) 
        
        self.pge = PGE(nfeat=self.d, nnodes=self.n, device=self.device, args=self.args).to(self.device)
        self.tune=False
    
    def init_fine_tune(self):
        
        self.r=self.args.rank
        self.lora_alpha=self.args.lora_alpha
        
        std_dev = 1 / torch.sqrt(torch.tensor(self.r).float())
        self.feat_syn_a=torch.randn(size=(self.n,self.r),dtype=torch.float,requires_grad=True,device=self.device)
        self.feat_syn_b=torch.zeros(size=(self.r,self.d),dtype=torch.float,requires_grad=True,device=self.device)

        self.feat_syn.requires_grad=False
        self.base_feat=copy.deepcopy(self.feat_syn.detach())
            
        self.tune=True
    
    def save_condense_graph(self):
        
        if self.args.alignment == 1 and self.args.smoothness == 1:
            torch.save(self.feat_syn, f'{self.root}/saved_ours/feat_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt')
            torch.save(self.pge.state_dict(), f'{self.root}/saved_ours/pge_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt')
        if self.args.alignment == 1 and self.args.smoothness == 0:
            torch.save(self.feat_syn, f'{self.root}/saved_ours/feat_without_smoothness_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt')
            torch.save(self.pge.state_dict(), f'{self.root}/saved_ours/pge_without_smoothness_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt')
        if self.args.alignment == 0 and self.args.smoothness == 1:
            torch.save(self.feat_syn, f'{self.root}/saved_ours/feat_without_alignment_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt')
            torch.save(self.pge.state_dict(), f'{self.root}/saved_ours/pge_without_alignment_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt')
        if self.args.alignment == 0 and self.args.smoothness == 0:
            torch.save(self.feat_syn, f'{self.root}/saved_ours/feat_without_both_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt')
            torch.save(self.pge.state_dict(), f'{self.root}/saved_ours/pge_without_both_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt')
    
    def eval_condense_retrain(self,epoch,
                              validation_model,
                              optimizer):
        
        adj_syn=self.pge.inference(self.feat_syn).detach().to(self.device)
        adj_syn[adj_syn<self.args.threshold]=0
        adj_syn.requires_grad=False
        edge_index_syn=torch.nonzero(adj_syn).T
        edge_weight_syn= adj_syn[edge_index_syn[0], edge_index_syn[1]]
        edge_index_syn, edge_weight_syn=gcn_norm(edge_index_syn, edge_weight_syn, self.n)

        teacher_output_syn = self.teacher_model.predict(self.feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
        acc = utils.accuracy(teacher_output_syn, self.labels_syn)
        print('Epoch {}'.format(epoch),"Teacher on syn accuracy= {:.4f}".format(acc.item()))
    
        validation_model.initialize()
        for j in range(self.args.student_model_loop):
            validation_model.train()
            optimizer.zero_grad()
            output_syn = validation_model.forward(self.feat_syn.detach(), edge_index_syn, edge_weight=edge_weight_syn)
            loss = F.nll_loss(output_syn,self.labels_syn)
            loss.backward()
            optimizer.step()

            if j%self.args.student_val_stage==0:
                if self.args.inference==True:
                    output = validation_model.inference(self.feat, self.inference_loader, self.device)
                else:
                    if self.tune==False:
                        output = validation_model.predict(self.feat.to(self.device), self.adj.to(self.device))
                    else:
                        output = validation_model.predict(self.feat.to(self.device), self.adj_unlearn.to(self.device))
                acc_val = utils.accuracy(output[self.idx_val], self.labels_val)
                acc_test = utils.accuracy(output[self.idx_test], self.labels_test)
                
                #print('acc_val: {}; acc_test: {}'.format(acc_val,acc_test))
                
                if(acc_val>self.best_val):
                    self.best_val=acc_val
                    self.best_test=acc_test
                    if self.args.save:
                        self.save_condense_graph()
                        torch.save(validation_model.state_dict(), f'{self.root}/saved_model/student/{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_3_256_0.5_relu_1.pt')
        
        print('Epoch {}'.format(epoch), "Best test acc:", self.best_test)
    
    def random_proj_cov(self,z):
        
        k = torch.tensor(int(z.shape[0] * self.args.ratio))
        p = (1/torch.sqrt(k))*torch.randn(k, z.shape[0]).to(self.args.device)
        
        z_p = p @ z
        cov=z_p.T@z_p
        
        return cov
    
    def smooth_loss(self,edge_index_syn,edge_weight_syn):

        feat_difference = torch.exp(-0.5 * torch.pow(self.feat_syn[edge_index_syn[0]] - self.feat_syn[edge_index_syn[1]], 2))
        smoothness_loss = torch.dot(edge_weight_syn,torch.mean(feat_difference,1).flatten())/torch.sum(edge_weight_syn)
        
        return smoothness_loss
    
    def logits_loss(self,edge_index_syn,edge_weight_syn):

        output_syn = self.teacher_model.forward(self.feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
        hard_loss = F.nll_loss(output_syn, self.labels_syn)
        return hard_loss
    
    def representation_loss(self):

        concat_feat_loss=torch.tensor(0.0).to(self.device)
        concat_feat_std_loss=torch.tensor(0.0).to(self.device)
        concat_feat_cov_loss=torch.tensor(0.0).to(self.device)
        loss_fn=nn.MSELoss()
        for c in range(self.nclass):
            if c in self.num_class_dict:
                index=torch.where(self.labels_syn==c)
                concat_feat_mean_loss=self.coeff[c]*loss_fn(self.concat_feat_mean[c],self.concat_feat_syn[index].mean(dim=0))
                
                if self.args.align_mode in ['std','cs']:
                    self.concat_feat_syn_std=self.concat_feat_syn[index].std(dim=0)
                    concat_feat_std_loss=self.coeff[c]*loss_fn(self.concat_feat_std[c],self.concat_feat_syn_std)
                elif self.args.align_mode in ['cov','cs']:
                    z=self.concat_feat_syn[index]
                    self.concat_feat_syn_cov=z.T@z 
                    concat_feat_cov_loss=self.coeff[c]*loss_fn(self.concat_feat_cov[c],self.concat_feat_syn_cov)*self.args.cov_gamma
                    
                if self.feat_syn[index].shape[0]!=1:
                    concat_feat_loss+=(concat_feat_mean_loss+concat_feat_std_loss+concat_feat_cov_loss)
                else:
                    concat_feat_loss+=(concat_feat_mean_loss)
        concat_feat_loss=concat_feat_loss/self.coeff_sum
        
        return concat_feat_loss
    
    def cal_sim_dist_loss(self):
        
        sim_dist_loss=torch.tensor(0.0).to(self.device)
        loss_fn=nn.MSELoss()

        for c in range(self.nclass):
            if c in self.num_class_dict:
                
                mmd_loss=loss_fn(self.sim_mu_prior[c],self.sim_mu_poster[c])
                sim_dist_loss+=mmd_loss

        return sim_dist_loss
    
    def estimate_poster_dist(self,embed):
        
        ep=torch.ones(self.nclass)*self.args.epsilon
        ep=ep.to(self.device)
        
        sim_mu=[]
        
        syn_prototype=[]
        for c in range(self.nclass):
            if c in self.num_class_dict:
                index=torch.where(self.labels_syn==c)[0]
                syn_prototype.append(embed[index].mean(dim=0))
                
        prototype=torch.stack([syn_prototype[i] for i in range(len(syn_prototype))])
        
        for c in range(self.nclass):
            if c in self.num_class_dict:
                index=torch.where(self.labels_syn==c)[0]
                
                anchor=embed[index]
                sim=cosine_similarity(anchor,prototype) / self.args.tau_sim
                
                mu=sim.mean(dim=0)
                
                sim_mu.append(mu)
                
        return sim_mu
    
    def estimate_prior_dist(self,embed):

        ep=torch.ones(self.nclass)*self.args.epsilon
        ep=ep.to(self.device)
        
        sim_mu=[]

        prior_prototype=[]
        train_indices=torch.from_numpy(self.idx_train_unlearn).to(self.device)

        for c in range(self.nclass):
            if c in self.num_class_dict:
                index=train_indices[torch.where(self.labels_train_unlearn==c)[0]]
                prior_prototype.append(embed[index].mean(dim=0))
                    
        prototype=torch.stack([prior_prototype[i] for i in range(len(prior_prototype))])
            
        for c in range(self.nclass):
            if c in self.num_class_dict:
                index=train_indices[torch.where(self.labels_train_unlearn==c)[0]]
                
                anchor=embed[index]
                sim=cosine_similarity(anchor,prototype) / self.args.tau_sim
                
                mu=sim.mean(dim=0)
                
                sim_mu.append(mu)
        
        return sim_mu
    
    def contrast_reg_loss(self,embed,mode='soft'):

        intra_contrast_loss=torch.tensor(0.0).to(self.device)
        for c in range(self.nclass):
            if c in self.num_class_dict:
                index=torch.where(self.labels_syn==c)[0]
                
                anchor=embed[index]
                exp_sim=torch.exp(cosine_similarity(anchor,embed) / self.args.tau_intra)

                if mode=='soft':
                    loss_c=torch.log(exp_sim[:,index].sum(dim=1)/exp_sim.sum(dim=1))
                    loss_c=-loss_c.sum()
                else:
                    loss_c=-torch.log(exp_sim[:,index]/exp_sim.sum(dim=1)).sum()*(1/self.num_class_dict[c])
                
                intra_contrast_loss+=loss_c
        
        return intra_contrast_loss
    
    def train_syn(self):

        start = time.perf_counter()
        
        if self.args.model=='GCN':
            validation_model = GCN_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers, norm='BatchNorm').to(self.device)
        elif self.args.model=='GAT':
            validation_model = GAT_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout,nlayers=self.args.nlayers,norm='BatchNorm').to(self.device)
        elif self.args.model=='SGC':
            validation_model = SGC_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, nlayers=self.args.nlayers, sgc=True).to(self.device)
        elif self.args.model=='GIN':
            validation_model = GIN_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers, norm='BatchNorm', act=self.args.activation).to(self.device)
        
        
        optimizer = optim.Adam(validation_model.parameters(), lr=self.args.lr_model)
        optimizer_feat = optim.Adam([self.feat_syn], lr=self.args.lr_feat)
        optimizer_pge = optim.Adam(self.pge.parameters(), lr=self.args.lr_adj)
        
        if self.args.embed_model=='SGC':
            
            self.concat_feat=self.feat.to(self.device) 
            temp=self.feat
            for i in range(self.args.nlayers):
                aggr=self.embed_model.convs[0].propagate(self.adj.to(self.device), x=temp.to(self.device)).detach() 
                self.concat_feat=torch.cat((self.concat_feat,aggr),dim=1)
                temp=aggr
        else:
            raise ValueError('Undefined Embedding Model')
        
        self.concat_feat_mean=[]
        self.concat_feat_std=[]
        self.concat_feat_cov=[]
        self.coeff=[]
        self.coeff_sum=0
        
        for c in range(self.nclass):
            if c in self.num_class_dict:
                
                all_idx_c=torch.where(self.labels==c)[0]
                train_indices=torch.from_numpy(self.idx_train).to(self.device)
                index=all_idx_c[torch.isin(all_idx_c,train_indices)]
                
                coe = self.num_class_dict[c] / max(self.num_class_dict.values())
                self.coeff_sum+=coe
                self.coeff.append(coe)
                self.concat_feat_mean.append(self.concat_feat[index].mean(dim=0).to(self.device))
                if self.args.align_mode in ['std','cs']:
                    self.concat_feat_std.append(self.concat_feat[index].std(dim=0).to(self.device))
                if self.args.align_mode in ['cov','cs']:
                    z=self.concat_feat[index].to(self.device)
                    self.concat_feat_cov.append(self.random_proj_cov(z))
            else:
                self.coeff.append(0)
                self.concat_feat_mean.append([])
                self.concat_feat_std.append([])
                self.concat_feat_std.append([])
        self.coeff_sum=torch.tensor(self.coeff_sum).to(self.device)
        
        self.best_val=0
        self.best_test=0
        
        for i in range(self.args.condensing_loop+1):
            
            self.teacher_model.eval()
            optimizer_pge.zero_grad()
            optimizer_feat.zero_grad()
            
            adj_syn = self.pge(self.feat_syn).to(self.device)
            adj_syn[adj_syn<self.args.threshold]=0
            edge_index_syn = torch.nonzero(adj_syn).T
            edge_weight_syn = adj_syn[edge_index_syn[0], edge_index_syn[1]]

            #smoothness_loss = self.smooth_loss(edge_index_syn,edge_weight_syn)

            edge_index_syn, edge_weight_syn = gcn_norm(edge_index_syn, edge_weight_syn, self.n)
            
            if self.args.embed_model=='SGC':
                
                self.concat_feat_syn=self.feat_syn.to(self.device)
                temp=self.feat_syn
                for j in range(self.args.nlayers):
                    aggr_syn=self.embed_model.convs[0].propagate(edge_index_syn, x=temp, edge_weight=edge_weight_syn, size=None)
                    self.concat_feat_syn=torch.cat((self.concat_feat_syn,aggr_syn),dim=1)
                    temp=aggr_syn
            else:
                raise ValueError('Undefined Embedding Model')

            hard_loss=self.logits_loss(edge_index_syn,edge_weight_syn)
    
            concat_feat_loss=self.representation_loss()

            loss=self.args.logit_alpha*hard_loss+self.args.feat_alpha*concat_feat_loss
            loss.backward()
            
            if i%(self.args.tau_s+self.args.tau_f)<self.args.tau_s:
                optimizer_pge.step()
            else:
                optimizer_feat.step()    

            if i>=100 and i%self.args.val_gap==0:
                
                print('Epoch {}'.format(i),"Training Loss= {:.4f}".format(loss.item()))
                
                if self.args.val_mode=='retrain':
                    self.eval_condense_retrain(i,validation_model,optimizer)
                elif self.args.val_mode=='gntk':
                    pass
                else:
                    ValueError('Undefined Evaluation Method')
        
        end = time.perf_counter()
        print('Condensation Duration:',round(end-start), 's')

    def fine_tune_syn(self):
        
        self.func_space=[]
        
        start = time.perf_counter()
        
        if self.args.model=='GCN':
            validation_model = GCN_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers, norm='BatchNorm').to(self.device)
        elif self.args.model=='GAT':
            validation_model = GAT_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout,nlayers=self.args.nlayers,norm='BatchNorm').to(self.device)
        elif self.args.model=='SGC':
            validation_model = SGC_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, nlayers=self.args.nlayers, sgc=True).to(self.device)
        elif self.args.model=='GIN':
            validation_model = GIN_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers, norm='BatchNorm', act=self.args.activation).to(self.device)
        
        optimizer = optim.Adam(validation_model.parameters(), lr=self.args.lr_model)
        optimizer_feat=optim.Adam([self.feat_syn_a,self.feat_syn_b], lr=self.args.lr_feat)
        optimizer_pge = optim.Adam(self.pge.parameters(), lr=self.args.lr_adj)
        
        if self.args.embed_model=='SGC':
            self.concat_feat=self.feat.to(self.device)
            temp=self.feat
            
            for i in range(self.args.nlayers):
                aggr=self.embed_model.convs[0].propagate(self.adj_unlearn.to(self.device), x=temp.to(self.device)).detach() 
                self.concat_feat=torch.cat((self.concat_feat,aggr),dim=1)
                temp=aggr
        else:
            raise ValueError('Undefined Embedding Model')

        self.concat_feat_mean=[]
        self.concat_feat_std=[]
        self.concat_feat_cov=[]
        self.coeff=[]
        self.coeff_sum=0
        
        train_indices=torch.from_numpy(self.idx_train_unlearn).to(self.device)
        
        for c in range(self.nclass):
            if c in self.num_class_dict:

                index=train_indices[torch.where(self.labels_train_unlearn==c)[0]]
                
                coe = self.num_class_dict[c] / max(self.num_class_dict.values()) 
                self.coeff_sum+=coe
                self.coeff.append(coe)
                self.concat_feat_mean.append(self.concat_feat[index].mean(dim=0).to(self.device))
                if self.args.align_mode in ['std','cs']:
                    self.concat_feat_std.append(self.concat_feat[index].std(dim=0).to(self.device))
                if self.args.align_mode in ['cov','cs']:
                    z=self.concat_feat[index].to(self.device)
                    self.concat_feat_cov.append(self.random_proj_cov(z))
            else:
                self.coeff.append(0)
                self.concat_feat_mean.append([])
                self.concat_feat_std.append([])
                self.concat_feat_cov.append([])
        self.coeff_sum=torch.tensor(self.coeff_sum).to(self.device)
    
        # Fine-tune
        for i in range(self.args.finetune_loop+1):
            
            '''Function Sampling'''
            if self.args.ps_mode=='random':
                self.determine_random_model()
            elif self.args.ps_mode=='tps':
                if i==0 or (i+1)%self.args.update_interval==0:
                    self.trajectory_func_sampling()
                self.determine_random_model()
            else:
                raise ValueError('Not implemented SDM Method')
            
            optimizer_pge.zero_grad()
            optimizer_feat.zero_grad()
            
            self.feat_syn=self.base_feat+(self.feat_syn_a.matmul(self.feat_syn_b))*self.lora_alpha

            adj_syn = self.pge(self.feat_syn).to(self.device)
            adj_syn[adj_syn<self.args.threshold]=0
            edge_index_syn = torch.nonzero(adj_syn).T
            edge_weight_syn = adj_syn[edge_index_syn[0], edge_index_syn[1]]
            
            edge_index_syn, edge_weight_syn = gcn_norm(edge_index_syn, edge_weight_syn, self.n)
            self.concat_feat_syn=self.feat_syn.to(self.device)
            temp=self.feat_syn
            
            if self.args.embed_model=='SGC':
                for j in range(self.args.nlayers):
                    aggr_syn=self.embed_model.convs[0].propagate(edge_index_syn, x=temp, edge_weight=edge_weight_syn, size=None)
                    self.concat_feat_syn=torch.cat((self.concat_feat_syn,aggr_syn),dim=1)
                    temp=aggr_syn
            else:
                raise ValueError('Undefined Embedding Model')
            
            concat_feat_loss=self.representation_loss()
               
            if self.args.model=='SGC':
                sim_dist_loss=torch.tensor(0.0).to(self.device)
                intra_contrast_loss=torch.tensor(0.0).to(self.device)
            else:
                src_embeds=self.random_model.generate_embeddings(self.feat.to(self.device),self.adj_unlearn.to(self.device))
                syn_embeds=self.random_model.generate_embeddings(self.feat_syn.to(self.device),edge_index_syn)
                
                self.sim_mu_prior=self.estimate_prior_dist(src_embeds)
                self.sim_mu_poster=self.estimate_poster_dist(syn_embeds)
                
                sim_dist_loss=self.cal_sim_dist_loss()
                
                intra_contrast_loss=self.contrast_reg_loss(syn_embeds,mode=self.args.intra_mode)

            loss=self.args.sim_alpha*sim_dist_loss+\
                self.args.ft_feat_alpha*concat_feat_loss+\
                self.args.reg_alpha*intra_contrast_loss
                    
            loss.backward()
            
            if i%(self.args.tau_ts+self.args.tau_tf)<self.args.tau_ts:
                optimizer_pge.step()
            else:
                optimizer_feat.step()    

        end = time.perf_counter()
        running_time=end-start
        print(f'Finetune Duration:{running_time:.3f}s')

    def run_load_syn(self):

        if self.args.alignment == 0:
            self.args.feat_alpha = 0
        if self.args.smoothness == 0:
            self.args.smoothness_alpha = 0
        if self.args.alignment == 1 and self.args.smoothness == 1:
            #if not os.path.exists(self.root+'/saved_ours/feat_'+self.args.dataset+'_'+self.args.teacher_model+'_'+self.args.validation_model+'_'+str(self.args.reduction_rate)+'_'+str(self.args.seed)+'.pt'):
            print("Condensing!")
            self.train_syn()
            self.feat_syn=torch.load(f'{self.root}/saved_ours/feat_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt').to(self.device)
            self.pge.load_state_dict(torch.load(f'{self.root}/saved_ours/pge_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt'))
        elif self.args.alignment == 0 and self.args.smoothness == 1:
            if not os.path.exists(self.root+'/saved_ours/feat_without_alignment_'+self.args.dataset+'_'+self.args.teacher_model+'_'+self.args.validation_model+'_'+str(self.args.reduction_rate)+'_'+str(self.args.seed)+'.pt'):
                print("Condensing!")
                self.train_syn()
            self.feat_syn=torch.load(f'{self.root}/saved_ours/feat_without_alignment_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt').to(self.device)
            self.pge.load_state_dict(torch.load(f'{self.root}/saved_ours/pge_without_alignment_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt'))
        elif self.args.alignment == 1 and self.args.smoothness == 0:
            #if not os.path.exists(self.root+'/saved_ours/feat_without_smoothness_'+self.args.dataset+'_'+self.args.teacher_model+'_'+self.args.validation_model+'_'+str(self.args.reduction_rate)+'_'+str(self.args.seed)+'.pt'):
            print("Condensing!")
            self.train_syn()
            self.feat_syn=torch.load(f'{self.root}/saved_ours/feat_without_smoothness_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt').to(self.device)
            self.pge.load_state_dict(torch.load(f'{self.root}/saved_ours/pge_without_smoothness_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt'))
        else:
            if not os.path.exists(self.root+'/saved_ours/feat_without_both_'+self.args.dataset+'_'+self.args.teacher_model+'_'+self.args.validation_model+'_'+str(self.args.reduction_rate)+'_'+str(self.args.seed)+'.pt'):
                print("Condensing!")
                self.train_syn()
            self.feat_syn=torch.load(f'{self.root}/saved_ours/feat_without_both_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt').to(self.device)
            self.pge.load_state_dict(torch.load(f'{self.root}/saved_ours/pge_without_both_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt'))   

    def run_load_ft_syn(self):

        if self.args.alignment == 0:
            self.args.feat_alpha = 0
        if self.args.smoothness == 0:
            self.args.smoothness_alpha = 0
        if self.args.alignment == 1 and self.args.smoothness == 1:
            #if not os.path.exists(self.root+'/saved_ours/feat_'+self.args.dataset+'_'+self.args.teacher_model+'_'+self.args.validation_model+'_'+str(self.args.reduction_rate)+'_'+str(self.args.seed)+'.pt'):
            print("Fine-tuning!")
            self.fine_tune_syn()
            self.feat_syn=torch.load(f'{self.root}/saved_ours/feat_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt').to(self.device)
            self.pge.load_state_dict(torch.load(f'{self.root}/saved_ours/pge_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt'))
        elif self.args.alignment == 0 and self.args.smoothness == 1:
            #if not os.path.exists(self.root+'/saved_ours/feat_without_alignment_'+self.args.dataset+'_'+self.args.teacher_model+'_'+self.args.validation_model+'_'+str(self.args.reduction_rate)+'_'+str(self.args.seed)+'.pt'):
            print("Fine-tuning!")
            self.fine_tune_syn()
            self.feat_syn=torch.load(f'{self.root}/saved_ours/feat_without_alignment_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt').to(self.device)
            self.pge.load_state_dict(torch.load(f'{self.root}/saved_ours/pge_without_alignment_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt'))
        elif self.args.alignment == 1 and self.args.smoothness == 0:
            #if not os.path.exists(self.root+'/saved_ours/feat_without_smoothness_'+self.args.dataset+'_'+self.args.teacher_model+'_'+self.args.validation_model+'_'+str(self.args.reduction_rate)+'_'+str(self.args.seed)+'.pt'):
            print("Fine-tuning!")
            self.fine_tune_syn()
            self.feat_syn=torch.load(f'{self.root}/saved_ours/feat_without_smoothness_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt').to(self.device)
            self.pge.load_state_dict(torch.load(f'{self.root}/saved_ours/pge_without_smoothness_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt'))
        else:
            #if not os.path.exists(self.root+'/saved_ours/feat_without_both_'+self.args.dataset+'_'+self.args.teacher_model+'_'+self.args.validation_model+'_'+str(self.args.reduction_rate)+'_'+str(self.args.seed)+'.pt'):
            print("Fine-tuning!")
            self.fine_tune_syn()
            self.feat_syn=torch.load(f'{self.root}/saved_ours/feat_without_both_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt').to(self.device)
            self.pge.load_state_dict(torch.load(f'{self.root}/saved_ours/pge_without_both_{self.args.dataset}_{self.args.teacher_model}_{self.args.validation_model}_{self.args.reduction_rate}_{self.args.seed}.pt'))   
    
    def trajectory_func_sampling(self):

        if self.args.model=='GCN':
            model = GCN_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers, norm='BatchNorm', act=self.args.activation).to(self.device)
        elif self.args.model=='GAT':
            model = GAT_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout,nlayers=self.args.nlayers,norm='BatchNorm',act=self.args.activation).to(self.device)
        elif self.args.model=='SGC':
            model = SGC_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=0, nlayers=self.args.nlayers, sgc=True).to(self.device)
        elif self.args.model=='GIN':
            model = GIN_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers, norm='BatchNorm', act=self.args.activation).to(self.device)
        else:
            pass
        model.initialize()
        optimizer=optim.Adam(model.parameters(), lr=self.args.lr_model)

        adj_syn=self.pge.inference(self.feat_syn).detach().to(self.device)
        adj_syn[adj_syn<self.args.threshold]=0
        adj_syn.requires_grad=False
        edge_index_syn=torch.nonzero(adj_syn).T
        edge_weight_syn= adj_syn[edge_index_syn[0], edge_index_syn[1]]
        edge_index_syn, edge_weight_syn=gcn_norm(edge_index_syn, edge_weight_syn, self.n)
        
        memory = self.feat_syn.element_size() * self.feat_syn.nelement()
        memory1 = edge_index_syn.element_size() * edge_index_syn.nelement()
        memory2 = edge_weight_syn.element_size() * edge_weight_syn.nelement()
        print(memory+memory1+memory2) 
        
        start = time.perf_counter()
        
        best_test=0
        for j in range(self.args.traject_len+1):
            
            model.train()
            optimizer.zero_grad()
            if self.args.model!='MLP':
                output_syn = model.forward(self.feat_syn.detach(), edge_index_syn, edge_weight=edge_weight_syn)
            else:
                output_syn = model.forward(self.feat_syn.detach())
            loss=F.nll_loss(output_syn, self.labels_syn)
            loss.backward()
            optimizer.step()
            
            if j%self.args.sample_interval==0:
                if len(self.func_space)==self.args.param_set_vol:
                    self.func_space.pop(0)
                self.func_space.append(model.state_dict())
 
        end = time.perf_counter()
        
        running_time=end-start
        print(f'Training on the condensed graph:{running_time:.2f}s')
        print("Best Test Acc:",best_test)
        
    def train_model_on_cond_data(self):

        if self.args.model=='GCN':
            model = GCN_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers, norm='BatchNorm', act=self.args.activation).to(self.device)
        elif self.args.model=='GAT':
            model = GAT_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout,nlayers=self.args.nlayers,norm='BatchNorm',act=self.args.activation).to(self.device)
        elif self.args.model=='SGC':
            model = SGC_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=0, nlayers=self.args.nlayers, sgc=True).to(self.device)
        elif self.args.model=='GIN':
            model = GIN_PYG(nfeat=self.d, nhid=self.args.hidden, nclass=self.nclass, dropout=self.args.dropout, nlayers=self.args.nlayers, norm='BatchNorm', act=self.args.activation).to(self.device)
        else:
            pass
        model.initialize()
        optimizer=optim.Adam(model.parameters(), lr=self.args.lr_model)

        adj_syn=self.pge.inference(self.feat_syn).detach().to(self.device)
        adj_syn[adj_syn<self.args.threshold]=0
        adj_syn.requires_grad=False
        edge_index_syn=torch.nonzero(adj_syn).T
        edge_weight_syn= adj_syn[edge_index_syn[0], edge_index_syn[1]]
        edge_index_syn, edge_weight_syn=gcn_norm(edge_index_syn, edge_weight_syn, self.n)
        
        teacher_output_syn=self.teacher_model.predict(self.feat_syn, edge_index_syn, edge_weight=edge_weight_syn)
        acc = utils.accuracy(teacher_output_syn, self.labels_syn)
        print('Teacher on syn accuracy= {:.4f}'.format(acc.item()))
        memory = self.feat_syn.element_size() * self.feat_syn.nelement()
        memory1 = edge_index_syn.element_size() * edge_index_syn.nelement()
        memory2 = edge_weight_syn.element_size() * edge_weight_syn.nelement()
        print(memory+memory1+memory2) 
        
        start = time.perf_counter()

        best_val=0
        best_test=0
        for j in range(self.args.student_model_loop+1):
            
            model.train()
            optimizer.zero_grad()
            if self.args.model!='MLP':
                output_syn = model.forward(self.feat_syn.detach(), edge_index_syn, edge_weight=edge_weight_syn)
            else:
                output_syn = model.forward(self.feat_syn.detach())
            loss=F.nll_loss(output_syn, self.labels_syn)
            loss.backward()
            optimizer.step()

            if j%self.args.student_val_stage==0:
                if self.args.inference==False:
                    if self.args.model!='MLP':
                        if self.tune==False:
                            output = model.predict(self.feat.to(self.device), self.adj.to(self.device))
                        else:
                            output = model.predict(self.feat.to(self.device), self.adj_unlearn.to(self.device))
                    else:
                        output = model.predict(self.feat.to(self.device))
                else:
                    output = model.inference(self.feat,self.inference_loader,self.device)

                if self.tune==False:
                    acc_train = utils.accuracy(output[self.idx_train], self.labels_train)
                else:
                    acc_train = utils.accuracy(output[self.idx_train_unlearn], self.labels_train_unlearn)
                acc_val = utils.accuracy(output[self.idx_val], self.labels_val)
                acc_test = utils.accuracy(output[self.idx_test], self.labels_test)
                
                print(f'Epoch: {j:02d}, '
                        f'Train: {100 * acc_train.item():.2f}%, '
                        f'Valid: {100 * acc_val.item():.2f}% '
                        f'Test: {100 * acc_test.item():.2f}%')
                
                if(acc_val>best_val):
                    best_val=acc_val
                    best_test=acc_test
                    if self.args.save:
                        torch.save(model.state_dict(), f'{self.root}/saved_model/student/{self.args.dataset}_{self.args.teacher_model}_{self.args.model}_{self.args.reduction_rate}_{self.args.nlayers}_{self.args.hidden}_{self.args.dropout}_{self.args.activation}_{self.args.seed}.pt')

        end = time.perf_counter()
        
        self.model=model 
        
        running_time=end-start
        print(f'Training on the condensed graph:{running_time:.3f}s')
        print("Best Test Acc:",best_test)
    
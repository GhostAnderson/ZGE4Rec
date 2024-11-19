"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    def zinb_loss(self, A, pi, mu, theta):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class ZINB_encoder(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = 1
        self.lantent_dim = 64

        self.layer1 = nn.Linear(self.input_dim, self.lantent_dim)

        self.pi_o = nn.Linear(self.lantent_dim, self.output_dim)
        self.disp_o = nn.Linear(self.lantent_dim, self.output_dim)
        self.mean_o = nn.Linear(self.lantent_dim, self.output_dim)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_normal_(self.pi_o.weight)
        nn.init.xavier_normal_(self.disp_o.weight)
        nn.init.xavier_normal_(self.mean_o.weight)
    
    def forward(self, x):
        
        x = F.relu(self.layer1(x))

        pi = torch.sigmoid(torch.clip(self.pi_o(x), -10, 10))
        theta = torch.clip(F.softplus(self.disp_o(x)), 1e-4, 10)
        mu = torch.clip(F.softplus(self.mean_o(x)), 1e-5, 10)

        return pi, mu, theta

class NBLoss(nn.Module):
    def __init__(self, eps=1e-10, scale_factor=1.0, reduction='none') -> None:
        super().__init__()
        self.eps = eps
        self.scale_factor = scale_factor
        self.reduction = reduction
    
    def forward(self, A, mu, theta):
        theta = torch.minimum(theta, torch.Tensor([1e6]).to(world.device))
        mu = mu * self.scale_factor

        t1 = torch.lgamma(theta+self.eps) + torch.lgamma(A+1.0) - torch.lgamma(A+theta+self.eps)
        t2 = (theta+A) * torch.log(1.0 + (mu/(theta+self.eps))) + (A * (torch.log(theta+self.eps) - torch.log(mu+self.eps)))

        output = torch.nan_to_num(t1+t2, nan=0)

        if self.reduction == 'mean':
            output = torch.mean(output)
        elif self.reduction == 'sum':
            output = torch.sum(output)
        elif self.reduction != 'none':
            raise NotImplementedError('Reduction methods: {} is not implemented.')

        return output

class ZINBLoss(nn.Module):
    def __init__(self, ridge_lambda=0.0, eps=1e-10, scale_factor=1, reduction='mean') -> None:
        super().__init__()
        self.nbloss = NBLoss(eps, scale_factor, reduction='none')
        self.ridge_lambda = ridge_lambda
        self.scale_factor = scale_factor
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, A, pi, mu, theta):
        nonzero_zinb = self.nbloss(A, mu, theta) - torch.log(1.0 - pi + self.eps)
        mu = mu * self.scale_factor
        theta = torch.minimum(theta, torch.Tensor([1e6]).to(world.device))

        zero_nb = torch.pow(theta/(theta+mu+self.eps), theta)
        zero_zinb = -torch.log(pi + ((1.0-pi)*zero_nb) + self.eps)
        result = torch.where(A<1e-6, zero_zinb, nonzero_zinb)

        result = torch.nan_to_num(result, nan=0)

        if self.reduction == 'mean':
            result = torch.mean(result)
        elif self.reduction == 'sum':
            result = torch.sum(result)
        elif self.reduction != 'none':
            raise NotImplementedError('Reduction methods: {} is not implemented.')

        return result


class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        self.user_zinb_decoder = ZINB_encoder(self.latent_dim)
        self.item_zinb_decoder = ZINB_encoder(self.latent_dim)

        self.zinbloss = ZINBLoss()

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        # u_pi, u_mu, u_theta, v_pi, v_mu, v_theta = None, None, None, None, None, None
        u_pi, u_mu, u_theta = self.user_zinb_decoder(all_users[users])
        v_pi, v_mu, v_theta = self.item_zinb_decoder(all_items[torch.hstack([pos_items,neg_items])])
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego,\
              u_pi, u_mu, u_theta, v_pi, v_mu, v_theta
    
    def bpr_loss(self, users, pos, neg, A):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0, 
        u_pi, u_mu, u_theta,
        v_pi, v_mu, v_theta) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        
        user_zinb_reg_loss = 0
        for user_param in self.user_zinb_decoder.parameters():
            assert not torch.isnan(user_param).any()
            user_zinb_reg_loss += torch.norm(user_param)
        item_zinb_reg_loss = 0
        for item_param in self.item_zinb_decoder.parameters():
            assert not torch.isnan(item_param).any()
            item_zinb_reg_loss += torch.norm(item_param)

        # reg_loss += 0.1 * (user_zinb_reg_loss + item_zinb_reg_loss)

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        user_A = A[users]
        user_zinb_loss = self.zinbloss(user_A, u_pi, u_mu, u_theta)

        item_A = A.T[torch.hstack([pos, neg])]
        item_zinb_loss = self.zinbloss(item_A, v_pi, v_mu, v_theta)
        
        return loss, reg_loss, user_zinb_loss, item_zinb_loss
       
    # def forward(self, users, items):
    #     # compute embedding

    #     u_pi, u_mu, u_theta = self.user_zinb_decoder(users)
    #     v_pi, v_mu, v_theta = self.item_zinb_decoder(items)

    #     # print('forward')
    #     #all_users, all_items = self.computer()
    #     users_emb = all_users[users]
    #     items_emb = all_items[items]
    #     inner_pro = torch.mul(users_emb, items_emb)
    #     gamma     = torch.sum(inner_pro, dim=1)
    #     return gamma, u_pi, u_mu, u_theta, v_pi, v_mu, v_theta
    
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
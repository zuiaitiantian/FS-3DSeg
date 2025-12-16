""" MPTI with/without attention Learner for Few-shot 3D Point Cloud Semantic Segmentation

Author: Zhao Na, 2020
"""
import os
import torch
from torch import optim
from torch.nn import functional as F
import numpy as np
from models.mldb import ProtoNetAlignQGPASR
from utils.checkpoint_util import load_pretrain_checkpoint, load_model_checkpoint

class Learner(object):
    def __init__(self, args, mode='train'):

        # init model and optimizer
        self.model = ProtoNetAlignQGPASR(args)
        print(self.model)
        if torch.cuda.is_available():
            self.model.cuda()

        if mode=='train':
           
            self.optimizer = torch.optim.AdamW(
                [{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                 {'params': self.model.base_learner.parameters()},
                 {'params': self.model.GraphNN.parameters(), 'lr': 0.0001},
                 {'params': self.model.GraphNN1.parameters()},
                 #{'params': self.model.GraphNN2.parameters()},
                 {'params': self.model.att_learner.parameters()}], lr=args.lr)

            #set learning rate scheduler
            #.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size,
            #                                              gamma=args.gamma)
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-5)  # T_max为周期长度，eta_min为最小学习率
            if args.model_checkpoint_path is None:
                # load pretrained model for point cloud encoding
                self.model = load_pretrain_checkpoint(self.model, args.pretrain_checkpoint_path)
            else:
                # resume from model checkpoint
                self.model, self.optimizer = load_model_checkpoint(self.model, args.model_checkpoint_path,
                                                                   optimizer=self.optimizer, mode='train')
        elif mode=='test':
            # Load model checkpoint
            self.model = load_model_checkpoint(self.model, args.model_checkpoint_path, mode='test')
        else:
            raise ValueError('Wrong GraphLearner mode (%s)! Option:train/test' %mode)

    def train(self, data):
        """
        Args:
            data: a list of torch tensors wit the following entries.
            - support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            - support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            - query_x: query point clouds with shape (n_queries, in_channels, num_points)
            - query_y: query labels with shape (n_queries, num_points)
        """

        [support_x, support_y, query_x, query_y] = data
        self.model.train()

        query_logits, loss,_= self.model(support_x, support_y, query_x, query_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.lr_scheduler.step()

        query_pred = F.softmax(query_logits, dim=1).argmax(dim=1)
        correct = torch.eq(query_pred, query_y).sum().item()  # including background class
        accuracy = correct / (query_y.shape[0]*query_y.shape[1])

        return loss, accuracy


    def test(self, data):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points), each point \in {0,1}.
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        """
        [support_x, support_y, query_x, query_y] = data
        self.model.eval()

        with torch.no_grad():
            logits, loss,_,fg_prototypes,bg_prototype,support_feat= self.model(support_x, support_y, query_x, query_y)
            support_feat=support_feat.transpose(2,3).reshape(-1,320)
            print(support_feat.shape,fg_prototypes.shape,bg_prototype.shape)
            fg_prototypes=fg_prototypes.view(160,320)
            bg_prototype=bg_prototype.view(80,320)
            filename = f'/zhaoyong/PAP3D/fg.txt'
            filename0 = f'/zhaoyong/PAP3D/bg.txt'
            filename1 = f'/zhaoyong/PAP3D/sf.txt'
            filename2 = f'/zhaoyong/PAP3D/sy.txt'
            numpy_data = fg_prototypes.cpu().numpy()  
            numpy_data1 = bg_prototype.cpu().numpy()  
            # 将NumPy数组转换为字符串形式，每行一个数组元素  
            np.savetxt(filename, numpy_data, fmt='%f')
            np.savetxt(filename0, numpy_data1, fmt='%f')
            
            #data_tensor = torch.cat((query_feat.cuda(), groundtruth.reshape(2048,1)), dim=1)
            data_numpy = support_feat.cpu().numpy()
            np.savetxt(filename1, data_numpy, fmt='%f')
            support_y[1, 0, support_y[1, 0] == 1] = 2
            # 将修改后的张量与自身拼接，生成[4096]维的张量
            #new_tensor = torch.cat((tensor.view(-1), tensor.view(-1)))
            y = support_y.view(4096,1).cpu().numpy()
            np.savetxt(filename2, y, fmt='%d')
            print(support_feat.shape,fg_prototypes.shape,bg_prototype.shape,y.shape)
            pred = F.softmax(logits, dim=1).argmax(dim=1)
            correct = torch.eq(pred, query_y).sum().item()
            accuracy = correct / (query_y.shape[0]*query_y.shape[1])
            
        return pred, loss, accuracy

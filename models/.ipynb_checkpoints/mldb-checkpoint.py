import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from models.dgcnn_new import DGCNN_semseg
from models.attention import SelfAttention, QGPA
from torch_cluster import fps

import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GraphNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphNN, self).__init__()
    
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(p=0.1)  
        self.batch_norm = torch.nn.BatchNorm1d(hidden_channels)  
    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index.cuda())
        x = torch.relu(x)
        x = self.batch_norm(x)  # Apply batch normalization
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x, edge_index.cuda())
        x = torch.relu(x)
        return x


class GraphNN1(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphNN1, self).__init__()
    
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(p=0.1)  
        self.batch_norm = torch.nn.BatchNorm1d(hidden_channels)  
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index.cuda())
        x = torch.relu(x)
        x = self.batch_norm(x)  # Apply batch normalization
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x, edge_index.cuda())
        x = torch.relu(x)
        return x

class GraphNN2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphNN2, self).__init__()
    
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(p=0.1)  
        self.batch_norm = torch.nn.BatchNorm1d(hidden_channels)  

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index.cuda())
        x = torch.relu(x)
        x = self.batch_norm(x)  # Apply batch normalization
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x, edge_index.cuda())
        x = torch.relu(x)
        return x

def create_graph_data(features, num_nodes):
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make it undirected
    edge_index = edge_index.to(torch.long)
    
    return Data(x=features, edge_index=edge_index)

class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i-1]
            self.convs.append(nn.Sequential(
                              nn.Conv1d(in_dim, params[i], 1),
                              nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs-1:
                x = F.relu(x)
        return x


class ProtoNetAlignQGPASR(nn.Module):
    def __init__(self, args):
        super(ProtoNetAlignQGPASR, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = 'cosine'
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.use_align = args.use_align
        self.use_linear_proj = args.use_linear_proj
        self.use_supervise_prototype = args.use_supervise_prototype
  
        self.encoder = DGCNN_semseg(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        if self.use_linear_proj:
            self.conv_1 = nn.Sequential(nn.Conv1d(args.train_dim, args.train_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(args.train_dim),
                                   nn.LeakyReLU(negative_slope=0.2))
           # self.conv_2 = nn.Sequential(nn.Linear(args.train_dim, args.train_dim))
                                   #nn.BatchNorm1d(args.train_dim),
                                   #nn.LeakyReLU(negative_slope=0.2))
        self.use_transformer = args.use_transformer
        
        self.num_prototypes=2
        self.feat_dim=320
        self.sigma=1
        self.n_classes=self.n_way+1
        self.GraphNN=GraphNN(320,320,320)
        self.GraphNN1=GraphNN1(320,320,320)
        #self.GraphNN2=GraphNN2(320,320,320)
        self.k_connect=200
    def forward(self, support_x, support_y, query_x, query_y):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points) [2, 9, 2048]
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points) [2, 1, 2048]
            query_x: query point clouds with shape (n_queries, in_channels, num_points) [2, 9, 2048]
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way} [2, 2048]
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points)
        support_feat, _ = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        query_feat, xyz = self.getFeatures(query_x) #(n_queries, feat_dim, num_points
        query_feat = query_feat.transpose(1,2).contiguous().view(-1, self.feat_dim) #(n_queries*num_points, feat_dim)
        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        k=100
        loss=0
        query_pred=torch.zeros(2,3,2048).cuda()
     
        ##############常规通过多原型来预测
        fg_prototypes, fg_labels = self.getForegroundPrototypes(support_feat, fg_mask, k)
        bg_prototype, bg_labels = self.getBackgroundPrototypes(support_feat, bg_mask, k)

        # prototype learning
        if bg_prototype is not None and bg_labels is not None:
            #bg_prototype1, bg_labels1 = self.getBackgroundPrototypes1(bg_prototype.view(1,bg_prototype.shape[0],320).unsqueeze(-1).transpose(2,3), torch.argmax(bg_labels,dim=1).view(1,-1).unsqueeze(1), k=30)
            prototypes = torch.cat((bg_prototype, fg_prototypes), dim=0) #(*, feat_dim)
            prototype_labels = torch.cat((bg_labels, fg_labels), dim=0) #(*,n_classes)
        else:
            prototypes = torch.cat((fg_prototypes), dim=0)
            prototype_labels = torch.cat((fg_labels), dim=0)
        
        loss,query_pred=self.predict(prototypes,prototype_labels,query_feat,query_y)
        loss += self.computeCrossEntropyLoss(query_pred, query_y)


        #查询超点
        pre_label = torch.argmax(query_pred, dim=1)
        
        query_pred11=torch.cat((query_pred[0],query_pred[1]),dim=-1)
        predictions = torch.argmax(query_pred11, dim=0)   
        confidences = torch.max(query_pred11, dim=0)[0]  
        final_labels = -1 * torch.ones_like(predictions, dtype=torch.int64)  
        for i in range(confidences.shape[0]):  
            if confidences[i] > torch.mean(confidences):    
                final_labels[i] = predictions[i]  # 使用预测的类别索引作为最终标签  
                #s=s+1
       
        
        first_half = final_labels[:2048]  # 取前2048个元素  
        second_half = final_labels[2048:]  # 取后2048个元素  

        # 拼接张量  
        pre_label = torch.stack([first_half, second_half], dim=0)  # 沿着第0维拼接  
   
        query_fg_mask = pre_label.unsqueeze(1)
        query_bg_mask = torch.logical_not(pre_label).unsqueeze(1)
        query_feat1=query_feat.view(2,2048,320).transpose(1,2).unsqueeze(1)
        #生成查询超点
        query_fg_prototypes, query_fg_labels = self.getForegroundPrototypes(query_feat1, query_fg_mask, k=100)
        query_bg_prototype, query_bg_labels = self.getBackgroundPrototypes(query_feat1, query_bg_mask, k=100)



        #################全局超点
        #拼接原型
        fg_proto=torch.cat((fg_prototypes,query_fg_prototypes),dim=0)
        fg_label=torch.cat((torch.argmax(fg_labels,dim=1),torch.argmax(query_fg_labels,dim=1)),dim=0)
        if bg_prototype is not None and query_bg_prototype is not None:
            bg_proto=torch.cat((bg_prototype,query_bg_prototype),dim=0)
            bg_label=torch.cat((torch.argmax(bg_labels,dim=1),torch.argmax(query_bg_labels,dim=1)),dim=0)
        elif bg_prototype is not None:
            bg_proto=bg_prototype
            bg_label=torch.argmax(bg_labels,dim=1)
        elif query_bg_prototype is not None:
            bg_proto=query_bg_prototype
            bg_label=torch.argmax(query_bg_labels,dim=1)
        if fg_proto.shape[0]%2==1:
            fg_proto=torch.cat((fg_proto,fg_proto),dim=0)
            fg_label=torch.cat((fg_label,fg_label),dim=0)
        #生产全局超点
        query_fg_prototypes1, query_fg_labels1 = self.getForegroundPrototypes(fg_proto.view(2,-1,320).unsqueeze(-1).transpose(2,3), fg_label.view(2,-1).unsqueeze(1), k=2)
        
        # prototype learning
        if bg_proto is not None and bg_label is not None:
            query_bg_prototype1, query_bg_labels1 = self.getBackgroundPrototypes1(bg_proto.view(1,-1,320).unsqueeze(-1).transpose(2,3), bg_label.view(1,-1).unsqueeze(1), k=1)
            query_prototypes = torch.cat((query_bg_prototype1, query_fg_prototypes1), dim=0)
            query_prototype_labels = torch.cat((query_bg_labels1, query_fg_labels1), dim=0)
        else:
            query_prototypes = torch.cat((query_fg_prototypes1), dim=0)
            query_prototype_labels = torch.cat((query_fg_labels1), dim=0)
        # prototype learning
        ###全局超点预测支持特征
        loss_ref,_=self.predict(query_prototypes,query_prototype_labels,support_feat,support_y)

        ###全局超点预测查询特征（第一次）
        loss_1,query_pred_1=self.predict(query_prototypes,query_prototype_labels,query_feat,query_y)


        # 查询mask来生成查询原型
        pre_label_1 = torch.argmax(query_pred_1, dim=1)
        query_pred_1=torch.cat((query_pred_1[0],query_pred_1[1]),dim=-1)
        predictions_1 = torch.argmax(query_pred_1, dim=0)
        confidences_1 = torch.max(query_pred_1, dim=0)[0]
        final_labels_1 = -1 * torch.ones_like(predictions_1, dtype=torch.int64)
        for i in range(confidences_1.shape[0]):
            if confidences_1[i] > torch.mean(confidences_1):
                final_labels_1[i] = predictions_1[i]  # 使用预测的类别索引作为最终标签
        first_half_1 = final_labels_1[:2048]  # 取前2048个元素
        second_half_1 = final_labels_1[2048:]  # 取后2048个元素

        pre_label_1 = torch.stack([first_half_1, second_half_1], dim=0)  # 沿着第0维拼接
        query_fg_mask_1 = pre_label_1.unsqueeze(1)
        query_bg_mask_1 = torch.logical_not(pre_label_1).unsqueeze(1)
        query_feat1=query_feat.view(2,2048,320).transpose(1,2).unsqueeze(1)

        query_fg_prototypes, query_fg_labels = self.getForegroundPrototypes(query_feat1, query_fg_mask, k=100)
        query_bg_prototype, query_bg_labels = self.getBackgroundPrototypes(query_feat1, query_bg_mask, k=100)

        ########查询原型来生成全局原型
        fg_proto=torch.cat((fg_prototypes,query_fg_prototypes),dim=0)
        fg_label=torch.cat((torch.argmax(fg_labels,dim=1),torch.argmax(query_fg_labels,dim=1)),dim=0)
        if bg_prototype is not None and query_bg_prototype is not None:
            bg_proto=torch.cat((bg_prototype,query_bg_prototype),dim=0)
            bg_label=torch.cat((torch.argmax(bg_labels,dim=1),torch.argmax(query_bg_labels,dim=1)),dim=0)
        elif bg_prototype is not None:
            bg_proto=bg_prototype
            bg_label=torch.argmax(bg_labels,dim=1)
        elif query_bg_prototype is not None:
            bg_proto=query_bg_prototype
            bg_label=torch.argmax(query_bg_labels,dim=1)
        if fg_proto.shape[0]%2==1:
            fg_proto=torch.cat((fg_proto,fg_proto),dim=0)
            fg_label=torch.cat((fg_label,fg_label),dim=0)
        
        query_fg_prototypes1, query_fg_labels1 = self.getForegroundPrototypes(fg_proto.view(2,-1,320).unsqueeze(-1).transpose(2,3), fg_label.view(2,-1).unsqueeze(1), k=2)
        
        # prototype learning
        if bg_proto is not None and bg_label is not None:

            query_bg_prototype1, query_bg_labels1 = self.getBackgroundPrototypes1(bg_proto.view(1,-1,320).unsqueeze(-1).transpose(2,3), bg_label.view(1,-1).unsqueeze(1), k=1)
            query_prototypes = torch.cat((query_bg_prototype1, query_fg_prototypes1), dim=0)
            query_prototype_labels = torch.cat((query_bg_labels1, query_fg_labels1), dim=0)
        else:
            query_prototypes = torch.cat((query_fg_prototypes1), dim=0)
            query_prototype_labels = torch.cat((query_fg_labels1), dim=0)
        # prototype learning
        loss2,query_pred2=self.predict(query_prototypes,query_prototype_labels,query_feat,query_y)

##################################
        ###全局超点预测查询特征（第二次）
        pre_label2 = torch.argmax(query_pred2, dim=1)
        query_pred2=torch.cat((query_pred2[0],query_pred2[1]),dim=-1)
        predictions2 = torch.argmax(query_pred2, dim=0)
        confidences2 = torch.max(query_pred2, dim=0)[0]
        final_labels2 = -1 * torch.ones_like(predictions2, dtype=torch.int64)
        for i in range(confidences2.shape[0]):
            if confidences2[i] > torch.mean(confidences2):
                final_labels2[i] = predictions2[i]  # 使用预测的类别索引作为最终标签
        first_half2 = final_labels2[:2048]  # 取前2048个元素
        second_half2 = final_labels2[2048:]  # 取后2048个元素

        pre_label2 = torch.stack([first_half2, second_half2], dim=0)  # 沿着第0维拼接

        query_fg_mask2 = pre_label2.unsqueeze(1)
        query_bg_mask2 = torch.logical_not(pre_label2).unsqueeze(1)
        query_feat1 = query_feat.view(2, 2048, 320).transpose(1, 2).unsqueeze(1)

        query_fg_prototypes, query_fg_labels = self.getForegroundPrototypes(query_feat1, query_fg_mask, k=100)
        query_bg_prototype, query_bg_labels = self.getBackgroundPrototypes(query_feat1, query_bg_mask, k=100)
        fg_proto = torch.cat((fg_prototypes, query_fg_prototypes), dim=0)
        fg_label = torch.cat((torch.argmax(fg_labels, dim=1), torch.argmax(query_fg_labels, dim=1)), dim=0)
        if bg_prototype is not None and query_bg_prototype is not None:
            bg_proto = torch.cat((bg_prototype, query_bg_prototype), dim=0)
            bg_label = torch.cat((torch.argmax(bg_labels, dim=1), torch.argmax(query_bg_labels, dim=1)), dim=0)
        elif bg_prototype is not None:
            bg_proto = bg_prototype
            bg_label = torch.argmax(bg_labels, dim=1)
        elif query_bg_prototype is not None:
            bg_proto = query_bg_prototype
            bg_label = torch.argmax(query_bg_labels, dim=1)
        if fg_proto.shape[0] % 2 == 1:
            fg_proto = torch.cat((fg_proto, fg_proto), dim=0)
            fg_label = torch.cat((fg_label, fg_label), dim=0)

        query_fg_prototypes1, query_fg_labels1 = self.getForegroundPrototypes(
            fg_proto.view(2, -1, 320).unsqueeze(-1).transpose(2, 3), fg_label.view(2, -1).unsqueeze(1), k=2)

        # prototype learning
        if bg_proto is not None and bg_label is not None:

            query_bg_prototype1, query_bg_labels1 = self.getBackgroundPrototypes1(
                bg_proto.view(1, -1, 320).unsqueeze(-1).transpose(2, 3), bg_label.view(1, -1).unsqueeze(1), k=1)
            query_prototypes = torch.cat((query_bg_prototype1, query_fg_prototypes1), dim=0)
            query_prototype_labels = torch.cat((query_bg_labels1, query_fg_labels1), dim=0)
        else:
            query_prototypes = torch.cat((query_fg_prototypes1), dim=0)
            query_prototype_labels = torch.cat((query_fg_labels1), dim=0)
        # prototype learning
        ###全局超点预测查询特征（第三次）
        loss3, query_pred3 = self.predict(query_prototypes, query_prototype_labels, query_feat, query_y)

        loss+=loss_1
        loss+=loss2
        loss+=loss3
        return query_pred, loss,prototypes,fg_prototypes, bg_prototype,support_feat


    def iterrefine(self,all_prototypes,all_prototype_labels,query_feat,query_y):
        loss=0
        all_prototypes1=[]
        labels=[]
        for i in range(3):
            index=torch.where(all_prototype_labels[:, i] == 1) 
            feat =  all_prototypes[index]
            if feat.shape[0] != 0:
                class_prototypes = self.getMutiplePrototypes(feat, 10)
                all_prototypes1.append(class_prototypes[:10])
                class_labels = torch.zeros(class_prototypes.shape[0], self.n_classes)
                class_labels[:, i] = 1
                labels.append(class_labels[:10])

        all_prototypes2 = torch.cat(all_prototypes1, dim=0)
        all_labels2 = torch.cat(labels, dim=0)
        self.all_num_prototypes = all_prototypes2.shape[0]
        self.all_num_nodes = self.all_num_prototypes + query_feat.shape[0] # number of node of partial observed graph
        YA = torch.zeros(self.all_num_nodes, self.n_classes).cuda()
        YA[:self.all_num_prototypes] = all_labels2
        all_node_feat = torch.cat((all_prototypes2, query_feat), dim=0) #(num_nodes, feat_dim)
        
        AA = self.calculateLocalConstrainedAffinity2(all_node_feat, k=self.k_connect)
        ZA = self.label_propagate2(AA, YA) #(num_nodes, n_way+1)
        all_query_pred = ZA[self.all_num_prototypes:, :] #(n_queries*num_points, n_way+1)
        all_query_pred = all_query_pred.view(2, 2048, self.n_classes).transpose(1,2) #(n_queries, n_way+1, num_points)
        loss = self.computeCrossEntropyLoss(all_query_pred, query_y)
        return loss

    def predict(self,all_prototypes,all_prototype_labels,query_feat,query_y):
        loss=0
        self.all_num_prototypes = all_prototypes.shape[0]
        self.all_num_nodes = self.all_num_prototypes + query_feat.shape[0] # number of node of partial observed graph
        YA = torch.zeros(self.all_num_nodes, self.n_classes).cuda()
        YA[:self.all_num_prototypes] = all_prototype_labels
        all_node_feat = torch.cat((all_prototypes, query_feat), dim=0) #(num_nodes, feat_dim)
        
        AA = self.calculateLocalConstrainedAffinity2(all_node_feat, k=self.k_connect)
        ZA = self.label_propagate2(AA, YA) #(num_nodes, n_way+1)
        all_query_pred = ZA[self.all_num_prototypes:, :] #(n_queries*num_points, n_way+1)
        all_query_pred = all_query_pred.view(self.n_way, 2048, self.n_classes).transpose(1,2) #(n_queries, n_way+1, num_points)
        loss = self.computeCrossEntropyLoss(all_query_pred, query_y)
        return loss,all_query_pred
    

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        if self.use_attention:
            feat_level1, feat_level2, xyz = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            att_feat = self.att_learner(feat_level2)
            if self.use_linear_proj:
                return self.conv_1(torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1)), xyz
            else:
                return torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1), xyz
        else:
            # return self.base_learner(self.encoder(x))
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            map_feat = self.linear_mapper(feat_level2)
            return torch.cat((feat_level1, map_feat, feat_level3), dim=1)
    def getMutiplePrototypes(self, feat, k):
        """
        Extract multiple prototypes by points separation and assembly

        Args:
            feat: input point features, shape:(n_points, feat_dim)
        Return:
            prototypes: output prototypes, shape: (n_prototypes, feat_dim)
        """
        # sample k seeds as initial centers with Farthest Point Sampling (FPS)
        n = feat.shape[0]
        
        #assert n > 0
        if n<=0:
            n=2048
        ratio = k / n
        if ratio < 1:
            fps_index = fps(feat, None, ratio=ratio, random_start=False).unique()##FPS选择K个种子
            num_prototypes = len(fps_index)#原型个数
            farthest_seeds = feat[fps_index]#K个种子的特征【K, feat_dim】
            
            # compute the point-to-seed distance
            distances = F.pairwise_distance(feat[..., None], farthest_seeds.transpose(0, 1)[None, ...],
                                            p=2)  # (n_points, n_prototypes)
            
            # hard assignment for each point
            #根据种子点划分区域
            assignments = torch.argmin(distances, dim=1)  # (n_points,)
            
            # aggregating each cluster to form prototype
            prototypes = torch.zeros((num_prototypes, self.feat_dim)).cuda()
            for i in range(num_prototypes):
                selected = torch.nonzero(assignments == i).squeeze(1)
                selected = feat[selected, :]
                num_nodes = selected.shape[0]
                if num_nodes<1000:
                    prototypes[i] = selected.mean(0)#特征取平均
                #####################
                #利用GNN进行聚合
                else:
                # 使用 torch.combinations 生成所有可能的节点对索引  
                    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
                    graph_data = create_graph_data(selected, num_nodes)
                    graph_data.edge_index = edge_index
                    #构造图结构
                    if k>5:
                        f = self.GraphNN(graph_data.x, graph_data.edge_index, None)
                    else:
                        f = self.GraphNN1(graph_data.x, graph_data.edge_index, None)
                    prototypes[i]=f.mean(0)
                    #聚类成原型
            return prototypes
        else:
            return feat

    def getForegroundPrototypes(self, feats, masks, k=100):
        """
        Extract foreground prototypes for each class via clustering point features within that class

        Args:
            feats: input support features, shape: (n_way, k_shot, feat_dim, num_points)
            masks: foreground binary masks, shape: (n_way, k_shot, num_points)
        Return:
            prototypes: foreground prototypes, shape: (n_way*k, feat_dim)
            labels: foreground prototype labels (one-hot), shape: (n_way*k, n_way+1)
        """
        prototypes = []
        labels = []
        for i in range(self.n_way):
            # extract point features belonging to current foreground class
            feat = feats[i, ...].transpose(1,2).contiguous().view(-1, self.feat_dim) #(k_shot*num_points, feat_dim)
            index = torch.nonzero(masks[i, ...].view(-1)).squeeze(1) #(k_shot*num_points,)
            feat = feat[index]
            ##不同类别的特征数量【num_class i, feat_dim】
            if feat.shape[0] != 0:
                class_prototypes = self.getMutiplePrototypes(feat, k)
                prototypes.append(class_prototypes[:k])

            # construct label matrix
                class_labels = torch.zeros(class_prototypes.shape[0], self.n_classes)
                class_labels[:, i+1] = 1
                labels.append(class_labels[:k])

        prototypes = torch.cat(prototypes, dim=0)
        labels = torch.cat(labels, dim=0)

        return prototypes, labels
    def getForegroundPrototypes1(self, feats, masks, k=100):
        """
        Extract foreground prototypes for each class via clustering point features within that class
        feat:[200,320]
        mask:[2,1]
        Args:
            feats: input support features, shape: (n_way, k_shot, feat_dim, num_points)
            masks: foreground binary masks, shape: (n_way, k_shot, num_points)
        Return:
            prototypes: foreground prototypes, shape: (n_way*k, feat_dim)
            labels: foreground prototype labels (one-hot), shape: (n_way*k, n_way+1)
        """
        prototypes = []
        labels = []
        for i in range(self.n_way):
            # extract point features belonging to current foreground class
            #feat = feats[i, ...].transpose(1,2).contiguous().view(-1, self.feat_dim) #(k_shot*num_points, feat_dim)
            feat=feats.view(2,100,320)[i,...]
            index=torch.nonzero(masks[i,...])
            #index = torch.nonzero(masks[i, ...].view(-1)).squeeze(1) #(k_shot*num_points,)
            feat = feat[index]
            ##不同类别的特征数量【num_class i, feat_dim】
            if feat.shape[0] != 0:
                class_prototypes = self.getMutiplePrototypes(feat, k)
                prototypes.append(class_prototypes)

            # construct label matrix
                class_labels = torch.zeros(class_prototypes.shape[0], self.n_classes)
                class_labels[:, i+1] = 1
                labels.append(class_labels)

        prototypes = torch.cat(prototypes, dim=0)
        labels = torch.cat(labels, dim=0)

        return prototypes, labels
    def getBackgroundPrototypes1(self, feats, masks, k=100):
        """
        Extract background prototypes via clustering point features within background class

        Args:
            feats: input support features, shape: (n_way, k_shot, feat_dim, num_points)
            masks: background binary masks, shape: (n_way, k_shot, num_points)
        Return:
            prototypes: background prototypes, shape: (k, feat_dim)
            labels: background prototype labels (one-hot), shape: (k, n_way+1)
        """
        
        feats = feats.transpose(2,3).contiguous().view(-1, self.feat_dim)
        #index = torch.zero(masks.view(-1)).squeeze(1)
        index=torch.arange(0,masks.shape[-1])
        feat = feats[index]
        #print(111,feats.shape,feat.shape,masks.shape)
        # in case this support set does not contain background points..
        if feat.shape[0] != 0:
            prototypes = self.getMutiplePrototypes(feat, k)
            labels = torch.zeros(prototypes.shape[0], self.n_classes)
            labels[:, 0] = 1

            return prototypes[:k], labels[:k]
        else:
            return None, None
    def getBackgroundPrototypes(self, feats, masks, k=100):
        """
        Extract background prototypes via clustering point features within background class

        Args:
            feats: input support features, shape: (n_way, k_shot, feat_dim, num_points)
            masks: background binary masks, shape: (n_way, k_shot, num_points)
        Return:
            prototypes: background prototypes, shape: (k, feat_dim)
            labels: background prototype labels (one-hot), shape: (k, n_way+1)
        """
        feats = feats.transpose(2,3).contiguous().view(-1, self.feat_dim)
        index = torch.nonzero(masks.view(-1)).squeeze(1)
        feat = feats[index]
        #print(100,feats.shape,feat.shape,index.shape)
        # in case this support set does not contain background points..
        if feat.shape[0] != 0:
            prototypes = self.getMutiplePrototypes(feat, k)
            labels = torch.zeros(prototypes.shape[0], self.n_classes)
            labels[:, 0] = 1

            return prototypes[:k], labels[:k]
        else:
            return None, None
    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2)
        masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-5)
        return masked_feat

    def getPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype = bg_feat.sum(dim=(0,1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype

    def calculateLocalConstrainedAffinity(self, node_feat, k=200, method='gaussian'):
        """
        Calculate the Affinity matrix of the nearest neighbor graph constructed by prototypes and query points,
        It is a efficient way when the number of nodes in the graph is too large.

        Args:
            node_feat: input node features
                  shape: (num_nodes, feat_dim)
            k: the number of nearest neighbors for each node to compute the similarity
            method: 'cosine' or 'gaussian', different similarity function
        Return:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
        """
        # kNN search for the graph
        X = node_feat.detach().cpu().numpy()
        # build the index with cpu version
        index = faiss.IndexFlatL2(self.feat_dim)
        index.add(X)
        _, I = index.search(X, k + 1)
        I = torch.from_numpy(I[:, 1:]).cuda() #(num_nodes, k)

        # create the affinity matrix
        knn_idx = I.unsqueeze(2).expand(-1, -1, self.feat_dim).contiguous().view(-1, self.feat_dim)
        knn_feat = torch.gather(node_feat, dim=0, index=knn_idx).contiguous().view(self.num_nodes, k, self.feat_dim)
        
        if method == 'cosine':
            knn_similarity = F.cosine_similarity(node_feat[:,None,:], knn_feat, dim=0)
        elif method == 'gaussian':
            dist = F.pairwise_distance(node_feat[:,:,None], knn_feat.transpose(1,2), p=2)
            knn_similarity = torch.exp(-0.5*(dist/self.sigma)**2)
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        
        A = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float).cuda()
        A = A.scatter_(1, I, knn_similarity)
        ####I:[4096,k]
        ####knn_similarity:[4096,feat_dim]

        A = A + A.transpose(0,1)

        identity_matrix = torch.eye(self.num_nodes, requires_grad=False).cuda()
        A = A * (1 - identity_matrix)
        return A
    def calculateLocalConstrainedAffinity1(self, node_feat, k=200, method='gaussian'):
        """
        Calculate the Affinity matrix of the nearest neighbor graph constructed by prototypes and query points,
        It is a efficient way when the number of nodes in the graph is too large.

        Args:
            node_feat: input node features
                  shape: (num_nodes, feat_dim)
            k: the number of nearest neighbors for each node to compute the similarity
            method: 'cosine' or 'gaussian', different similarity function
        Return:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
        """
        # kNN search for the graph
        X = node_feat.detach().cpu().numpy()
        # build the index with cpu version
        index = faiss.IndexFlatL2(self.feat_dim)
        index.add(X)
        _, I = index.search(X, k + 1)
        I = torch.from_numpy(I[:, 1:]).cuda() #(num_nodes, k)

        # create the affinity matrix
        knn_idx = I.unsqueeze(2).expand(-1, -1, self.feat_dim).contiguous().view(-1, self.feat_dim)
        knn_feat = torch.gather(node_feat, dim=0, index=knn_idx).contiguous().view(self.query_num_nodes, k, self.feat_dim)
        
        if method == 'cosine':
            knn_similarity = F.cosine_similarity(node_feat[:,None,:], knn_feat, dim=0)
        elif method == 'gaussian':
            dist = F.pairwise_distance(node_feat[:,:,None], knn_feat.transpose(1,2), p=2)
            knn_similarity = torch.exp(-0.5*(dist/self.sigma)**2)
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        
        A = torch.zeros(self.query_num_nodes, self.query_num_nodes, dtype=torch.float).cuda()
        #print(knn_feat.shape,I.shape,A.shape,knn_similarity.shape)
        A = A.scatter_(1, I, knn_similarity)
        
        ####I:[4096,k]
        ####knn_similarity:[4096,feat_dim]

        A = A + A.transpose(0,1)

        identity_matrix = torch.eye(self.query_num_nodes, requires_grad=False).cuda()
        A = A * (1 - identity_matrix)
        return A
    def calculateLocalConstrainedAffinity2(self, node_feat, k=200, method='gaussian'):
        """
        Calculate the Affinity matrix of the nearest neighbor graph constructed by prototypes and query points,
        It is a efficient way when the number of nodes in the graph is too large.

        Args:
            node_feat: input node features
                  shape: (num_nodes, feat_dim)
            k: the number of nearest neighbors for each node to compute the similarity
            method: 'cosine' or 'gaussian', different similarity function
        Return:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
        """
        # kNN search for the graph
        X = node_feat.detach().cpu().numpy()
        # build the index with cpu version
        index = faiss.IndexFlatL2(self.feat_dim)
        index.add(X)
        _, I = index.search(X, k + 1)
        I = torch.from_numpy(I[:, 1:]).cuda() #(num_nodes, k)

        # create the affinity matrix
        knn_idx = I.unsqueeze(2).expand(-1, -1, self.feat_dim).contiguous().view(-1, self.feat_dim)
        knn_feat = torch.gather(node_feat, dim=0, index=knn_idx).contiguous().view(self.all_num_nodes, k, self.feat_dim)
        
        if method == 'cosine':
            knn_similarity = F.cosine_similarity(node_feat[:,None,:], knn_feat, dim=0)
        elif method == 'gaussian':
            dist = F.pairwise_distance(node_feat[:,:,None], knn_feat.transpose(1,2), p=2)
            knn_similarity = torch.exp(-0.5*(dist/self.sigma)**2)
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        
        A = torch.zeros(self.all_num_nodes, self.all_num_nodes, dtype=torch.float).cuda()
        #print(knn_feat.shape,I.shape,A.shape,knn_similarity.shape)
        A = A.scatter_(1, I, knn_similarity)
        
        ####I:[4096,k]
        ####knn_similarity:[4096,feat_dim]

        A = A + A.transpose(0,1)

        identity_matrix = torch.eye(self.all_num_nodes, requires_grad=False).cuda()
        A = A * (1 - identity_matrix)
        return A
    def label_propagate(self, A, Y, alpha=0.99):
        """ Label Propagation, refer to "Learning with Local and Global Consistency" NeurIPs 2003
        Args:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
            Y: initial label matrix, shape: (num_nodes, n_way+1)
            alpha: a parameter to control the amount of propagated info.
        Return:
            Z: label predictions, shape: (num_nodes, n_way+1)
        """
        #compute symmetrically normalized matrix S
        eps = np.finfo(float).eps
        D = A.sum(1) #(num_nodes,)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D_sqrt_inv = torch.diag_embed(D_sqrt_inv).cuda()
        S = D_sqrt_inv @ A @ D_sqrt_inv

        #close form solution
        Z = torch.inverse(torch.eye(self.num_nodes).cuda() - alpha*S + eps) @ Y
        return Z
    def label_propagate1(self, A, Y, alpha=0.99):
        """ Label Propagation, refer to "Learning with Local and Global Consistency" NeurIPs 2003
        Args:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
            Y: initial label matrix, shape: (num_nodes, n_way+1)
            alpha: a parameter to control the amount of propagated info.
        Return:
            Z: label predictions, shape: (num_nodes, n_way+1)
        """
        #compute symmetrically normalized matrix S
        eps = np.finfo(float).eps
        D = A.sum(1) #(num_nodes,)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D_sqrt_inv = torch.diag_embed(D_sqrt_inv).cuda()
        S = D_sqrt_inv @ A @ D_sqrt_inv

        #close form solution
        Z = torch.inverse(torch.eye(self.query_num_nodes).cuda() - alpha*S + eps) @ Y
        return Z
    def label_propagate2(self, A, Y, alpha=0.99):
        """ Label Propagation, refer to "Learning with Local and Global Consistency" NeurIPs 2003
        Args:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
            Y: initial label matrix, shape: (num_nodes, n_way+1)
            alpha: a parameter to control the amount of propagated info.
        Return:
            Z: label predictions, shape: (num_nodes, n_way+1)
        """
        #compute symmetrically normalized matrix S
        eps = np.finfo(float).eps
        D = A.sum(1) #(num_nodes,)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D_sqrt_inv = torch.diag_embed(D_sqrt_inv).cuda()
        S = D_sqrt_inv @ A @ D_sqrt_inv

        #close form solution
        Z = torch.inverse(torch.eye(self.all_num_nodes).cuda() - alpha*S + eps) @ Y
        return Z

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)

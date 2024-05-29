
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from .model_prototype import  VectorFeature
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling.make_layers import make_fc
from .model_prototype import MemoryBanks, PQLayer, FeedForward, MultiHeadAttention
from .model_prototype import PQLayer
from .model_motifs import LSTMContext, FrequencyBias, LSTMContext_mp
from .utils_prototype import *
from .utils_relation import layer_init
from .utils_motifs import obj_edge_vectors, rel_vectors, encode_box_info, nms_overlaps, to_onehot
from .utils_prototype import ExponentialMovingAverage, NetworkExponentialMovingAverage
import torch.nn.functional as F
from .model_transformer import TransformerContext

@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor_sub_prototype_Memory")
class MotifPredictor_sub_prototype_Memory(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor_sub_prototype_Memory, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.cfg = config

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = LSTMContext_mp(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.mlp_dim = 4096
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.mlp_dim)
        layer_init(self.post_emb, xavier=True)

        self.embed_dim = 300
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)
        self.project_head = MLP(self.mlp_dim, self.mlp_dim, int(self.mlp_dim * 2), 2)

        dropout_p = 0.2
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)

        # self.softmax_mask = nn.Parameter(torch.ones([]) * np.log(100000000000),requires_grad=False)
        self.logits_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)

        # sub-prototype initilization
        self.pred_diversity_list = get_semantic_diversity()
        self.num_sub_proto = int(sum(self.pred_diversity_list))
        self.sub_proto_list = range(self.num_sub_proto)
        self.proto_dim = 300
        self.codebook = PQLayer(feat_dim=self.proto_dim, K=self.num_sub_proto)
        #self.codebook._C = nn.Parameter(torch.randn((self.num_sub_proto, 300))).cuda()
        self.predicate_proto = []
        pred_proto_tuple = self.codebook._C.split(self.pred_diversity_list)
        for pred_proto in pred_proto_tuple:
            self.predicate_proto.append(pred_proto.mean(dim=0).view(1, -1))
        self.predicate_proto = cat(self.predicate_proto, dim=0).cuda() # average [51, 300]

        self.negative_mask_sub = torch.ones((self.num_sub_proto, self.num_sub_proto), requires_grad=False).cuda()
        for i in range(307):
            self.negative_mask_sub[i, i] = 0
        # start = 0
        # for k in range(len(self.pred_diversity_list)):
        #     length = self.pred_diversity_list[k]
        #     self.negative_mask_sub[start:start + length, start:start + length] = 0
        #     start = start + length
        self.negative_mask = torch.ones((self.num_rel_cls, self.num_rel_cls), requires_grad=False).cuda()
        for i in range(self.num_rel_cls):
            self.negative_mask[i, i] = 0
        self.softmax_mask = nn.Parameter(torch.ones([]) * np.log(100000000000), requires_grad=False)
        self.max_temp = 1
        self.curr_temp = self.max_temp
        self.min_temp = 0.1
        self.temp_decay = 0.9998

        # memory banks constructionsss
        self.use_memory = True
        self.is_timing = False
        self.bank_size = 64
        self.memory_bank = MemoryBanks(config, max_size=self.bank_size, feature_dim=self.mlp_dim * 2, sub_proto_list=self.sub_proto_list)

        # memory fusion unit
        self.attn = MultiHeadAttention(n_head=8, d_model=self.mlp_dim, d_k=int(self.mlp_dim/8), d_v=int(self.mlp_dim/8))

        self.iters = 0

        self.mask_general_sub_matching = torch.ones((self.num_sub_proto, self.num_rel_cls), requires_grad=False).cuda()
        start = 0
        for k in range(len(self.pred_diversity_list)):
            length = self.pred_diversity_list[k]
            self.mask_general_sub_matching[start:start + length, k] = 0
            start = start+length
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,
                sub_rel_labels=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            rel_labels (list[Tensor]): (batch_num, num_rel) relation labels
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        # load memory
        self.iters += 1
        if self.iters >= 10000 and self.use_memory:
            self.is_timing = True
        #self.is_timing = True
        if self.is_timing:
            with torch.no_grad():
                self.memory_bank = self.memory_bank.cpu()
                updated_memory = self.memory_bank.read()
                memo_roi_feature, memo_proposals, memo_rel_pair, memo_union_feat = updated_memory

                mem_rel_rep_list = []

                for m_roi, m_proposal, m_rel_pair, m_union in zip(memo_roi_feature, memo_proposals, memo_rel_pair, memo_union_feat):
                    if isinstance(m_roi, list):
                        mem_rel_rep_list.append(torch.zeros((4096,)).cuda())
                        continue

                    m_roi = m_roi.to("cuda")
                    m_proposal_list = [m_pro.to("cuda") for m_pro in m_proposal]
                    m_pair = m_rel_pair.to("cuda")
                    m_union = m_union.to("cuda")
                    _, _, edge_ctx, _ = self.context_layer(m_roi, m_proposal_list, m_pair, logger)

                    mem_edge_rep = self.post_emb(edge_ctx) # [N, 2048]
                    mem_edge_rep = mem_edge_rep.view(mem_edge_rep.size(0), 2, self.mlp_dim // 2)
                    mem_head_rep = mem_edge_rep[:, 0].contiguous().view(-1, self.mlp_dim // 2) # [N_obj, 1024] 都表示 obj 的 repr
                    mem_tail_rep = mem_edge_rep[:, 1].contiguous().view(-1, self.mlp_dim // 2) # [N_obj, 1024] 都表示 obj 的 repr

                    prod_reps = torch.cat((mem_head_rep[m_pair[:, 0]], mem_tail_rep[m_pair[:, 1]]), dim=-1)
                    if len(prod_reps.shape) == 1:
                        prod_reps = prod_reps.unsqueeze(0)
                    # relation representation
                    mem_rel_rep = prod_reps * self.down_samp(m_union)  # [N_rel_all, 2048]
                    mem_rel_rep_list.append(mem_rel_rep.mean(dim=0))

                mem_rel_reps = torch.stack(mem_rel_rep_list, dim=0).cuda().detach()
                del mem_rel_rep_list

        # extract feature ------ encode context infomation (extract relation feature from object)
        obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)  # edge_ctx 代表混合了 visual 和 embedding 特征的 predicate 特征

        # post decode
        edge_rep = self.post_emb(edge_ctx)  # [N_obj, 2, 1024]
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.mlp_dim // 2)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.mlp_dim // 2)  # [N_obj, 1024] 都表示 obj 的 repr
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.mlp_dim // 2)  # [N_obj, 1024] 都表示 obj 的 repr

        num_rels = [r.shape[0] for r in rel_pair_idxs]  # 关系数量 [N, 2]
        num_objs = [len(b) for b in proposals]  # 物体框数量 [N, ]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  # (N,): [N_obj, ]
        tail_reps = tail_rep.split(num_objs, dim=0)  # (N,): [N_obj, ]
        obj_preds = obj_preds.split(num_objs, dim=0)  # (N,): [N_obj, ] # object label or prediction from faster-rcnn

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))  # 2048
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)  # 所有样本的表征 [N_rel_all, 2048]
        rel_rep = prod_rep * self.down_samp(union_features)  # [N_rel_all, 2048]

        # feature update
        # query: rel_rep, key: mem_rel_reps, value: mem_rel_reps
        if self.is_timing:
            attn_mem_rep, _ = self.attn(rel_rep.unsqueeze(0), mem_rel_reps.unsqueeze(0), mem_rel_reps.unsqueeze(0))
            fusion_mem_rep = F.normalize(attn_mem_rep.squeeze(0),dim=0)
            rel_rep = 0.95 * rel_rep + 0.05 * fusion_mem_rep

        predicate_proto = self.W_pred(self.predicate_proto)  # [307, 2048]
        predicate_proto_sub = self.W_pred(self.codebook._C)

        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        predicate_proto_sub = self.project_head(self.dropout_pred(torch.relu(predicate_proto_sub)))

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm [N, 4096]

        rel_dists_sub = rel_rep_norm @ predicate_proto_sub.t() * self.logits_scale.exp()  # [N,102]
        rel_dists_tuple = rel_dists_sub.split(self.pred_diversity_list,dim=1)  #[N,51,*]
        self.temp_decay_func(1)

        rel_dists = []
        for rel_dist in rel_dists_tuple:
            rel_dist_mask = F.softmax(rel_dist.detach() * self.softmax_mask.exp(), dim=1)  # [N,1/2]
            #rel_dist_mask = F.softmax(rel_dist.detach()/self.curr_temp, dim=1)
            rel_dist = torch.sum(rel_dist * rel_dist_mask, dim=1) # [N,1/2]
            rel_dists.append(rel_dist.unsqueeze(dim=1) )# [N,1/2]
        rel_dists = cat(rel_dists, dim=1)  # [N,51]

        obj_dists = obj_dists.split(num_objs, dim=0)  # (N, ): [N_obj, 151]
        rel_dists = rel_dists.split(num_rels, dim=0)  # (N, ): [N_rel, 102]
        rel_dists_sub = rel_dists_sub.split(num_rels, dim=0)

        add_losses = {}
        if self.training:
            # write memory
            if self.is_timing:
                with torch.no_grad():
                    roi_features_objs = roi_features.split(num_objs, dim=0)
                    union_features_rels = union_features.split(num_rels, dim=0)
                    self.memory_bank.write(roi_features_objs, proposals, rel_pair_idxs, rel_dists_sub, union_features_rels)
            target_predicate_proto = predicate_proto.clone().detach()# [51, 2048]
            orthogonalities_sub_general = predicate_proto_sub @ target_predicate_proto.t()# [307, 51]
            orthogonalities_sub_general_negative = orthogonalities_sub_general.mul(self.mask_general_sub_matching)# [102, 51]
            orthogonalities_sub_general_loss = torch.norm(torch.norm(orthogonalities_sub_general_negative, p=2, dim=1), p=1) / (self.num_rel_cls*self.num_sub_proto)
            add_losses.update({"sub_general_loss": orthogonalities_sub_general_loss})

            # target_predicate_proto = predicate_proto.clone().detach()# [102, 4096]
            # orthogonalities = predicate_proto @ target_predicate_proto.t()# [102, 102]
            # orthogonalities_negative = orthogonalities.mul(self.negative_mask)# [102, 51]
            # orthogonalities_loss_negative = torch.norm(torch.norm(orthogonalities_negative, p=2, dim=1), p=1) / \
            #                                 (self.num_rel_cls * self.num_rel_cls)
            # add_losses.update({"p_loss_cs": orthogonalities_loss_negative})

            target_predicate_proto_sub = predicate_proto_sub.clone().detach()# [102, 4096]
            orthogonalities_sub = predicate_proto_sub @ target_predicate_proto_sub.t()# [102, 102]
            orthogonalities_sub_negative = orthogonalities_sub.mul(self.negative_mask_sub)# [102, 51]
            orthogonalities_sub_loss_negative = torch.norm(torch.norm(orthogonalities_sub_negative, p=2, dim=1), p=1) / \
                                                (self.num_sub_proto * self.num_sub_proto)
            # add_losses.update({"p_sub_loss_cs": orthogonalities_sub_loss_negative})

            rel_labels_tensor = cat(rel_labels, dim=0)
            rel_rep_list = []
            rel_labels_list = []
            for i in range(len(rel_labels_tensor)):
                if rel_labels_tensor[i] != 0:
                    rel_rep_list.append(rel_rep[i])
                    rel_labels_list.append(rel_labels_tensor[i])
            rel_rep_tensor = torch.stack(rel_rep_list, dim=0)
            rel_labels_tensors = torch.stack(rel_labels_list, dim=0)

            positive_mask = torch.ones((self.num_rel_cls, self.num_rel_cls), requires_grad=False).cuda()
            for i in range(self.num_rel_cls):
                positive_mask[i,i] = 0
            positive_mask[:, 0] = 0
            positive_mask[0, :] = 0
            positive_mask_batch = torch.ones((len(rel_labels_tensors), len(rel_labels_tensors)), requires_grad=False).cuda()
            for i in range(len(rel_labels_tensors)):
                positive_mask_batch[i,i] = 0
            target_rel_rep = rel_rep_tensor.clone().detach()
            rep_matrix = (rel_rep_tensor @ target_rel_rep.t()).mul(positive_mask_batch)
            rep_matrix_neg = torch.zeros((self.num_rel_cls, self.num_rel_cls), requires_grad=False).cuda()
            count_matrix_neg = torch.zeros((self.num_rel_cls, self.num_rel_cls), requires_grad=False).cuda()
            for i in range(len(rel_labels_tensors)):
                for j in range(len(rel_labels_tensors)):
                    rep_matrix_neg[rel_labels_tensors[i],rel_labels_tensors[j]] += rep_matrix[i,j]
                    count_matrix_neg[rel_labels_tensors[i],rel_labels_tensors[j]] += 1
            rep_matrix_neg /= (count_matrix_neg + 0.001)
            rep_matrix_neg = rep_matrix_neg.mul(positive_mask)
            orthogonalities_rep_loss = torch.norm(torch.norm(rep_matrix_neg, p=2, dim=1), p=1) / \
                                       (len(rel_labels_tensors) * len(rel_labels_tensors))
            add_losses.update({"rep_loss_neg": 0.1 * orthogonalities_rep_loss})

        return obj_dists, rel_dists, add_losses

    def temp_decay_func(self, num_updates):
        self.curr_temp = max(
            self.curr_temp * self.temp_decay ** num_updates, self.min_temp
        )

@registry.ROI_RELATION_PREDICTOR.register("VectorPredictor_mine_Memory")
class VectorPredictor_mine_Memory(nn.Module):
    def __init__(self, config, in_channels):
        super(VectorPredictor_mine_Memory, self).__init__()

        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.logits_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.mlp_dim = 2048
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)

        self.embed_dim = 300
        dropout_p = 0.2

        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR,
                                          wv_dim=self.embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR,
                                     wv_dim=self.embed_dim)  # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        # self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            # self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        # sub-prototype initilization
        self.pred_diversity_list = get_semantic_diversity()
        self.num_sub_proto = int(sum(self.pred_diversity_list))
        self.sub_proto_list = range(self.num_sub_proto)
        self.proto_dim = 300
        # self.codebook = PQLayer(feat_dim=self.proto_dim, K=self.num_sub_proto)
        self.codebook = PQLayer(feat_dim=self.proto_dim, K=self.num_sub_proto)  # sub-prototype space

        # self.codebook._C = nn.Parameter(torch.randn((self.num_sub_proto, 300)))
        self.predicate_proto = []
        pred_proto_tuple = self.codebook._C.split(self.pred_diversity_list)
        for pred_proto in pred_proto_tuple:
            self.predicate_proto.append(pred_proto.mean(dim=0).view(1, -1))
        self.predicate_proto = cat(self.predicate_proto, dim=0).cuda()  # average [51, 300]

        self.negative_mask_sub = torch.ones((self.num_sub_proto, self.num_sub_proto), requires_grad=False).cuda()
        start = 0
        for k in range(len(self.pred_diversity_list)):
            length = self.pred_diversity_list[k]
            self.negative_mask_sub[start:start + length, start:start + length] = 0
            start = start + length
        self.negative_mask = torch.ones((self.num_rel_cls, self.num_rel_cls), requires_grad=False).cuda()
        for i in range(self.num_rel_cls):
            self.negative_mask[i, i] = 0
        self.softmax_mask = nn.Parameter(torch.ones([]) * np.log(100000000000), requires_grad=False)
        self.max_temp = 1
        self.curr_temp = self.max_temp
        self.min_temp = 0.1
        self.temp_decay = 0.9998

        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)

        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)

        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)

        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes)
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        self.mask_general_sub_matching = torch.ones((self.num_sub_proto, self.num_rel_cls), requires_grad=False).cuda()
        start = 0
        for k in range(len(self.pred_diversity_list)):
            length = self.pred_diversity_list[k]
            self.mask_general_sub_matching[start:start + length, k] = 0
            start = start + length

        # memory banks construction
        self.use_memory = config.MODEL.ROI_RELATION_HEAD.MEMORY.USE_MEMORY
        self.is_timing = False
        self.bank_size = config.MODEL.ROI_RELATION_HEAD.MEMORY.BANK_SIZE
        self.memory_bank = MemoryBanks(config, max_size=self.bank_size, feature_dim=self.mlp_dim * 2,
                                       sub_proto_list=self.sub_proto_list)

        # ema update
        self.ema_decay = config.MODEL.ROI_RELATION_HEAD.MEMORY.EMA_DECAY
        self.ema_codebook = ExponentialMovingAverage(self.codebook._C,
                                                     self.ema_decay)  # update sub-prototype init_value: [460, 300]
        self.ema_weights = NetworkExponentialMovingAverage(self.W_pred, self.ema_decay)  # update W_p

        # memory fusion unit
        self.ffn = FeedForward(self.mlp_dim, self.mlp_dim)
        self.attn = MultiHeadAttention(n_head=1, d_model=2048, d_k=2048, d_v=2048)

        self.iters = 0

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,
                sub_rel_labels=None):

        # load memory
        self.iters += 1
        if self.iters >= self.cfg.SOLVER.MAX_ITER / 2 and self.use_memory:
            self.is_timing = True

        if self.is_timing:
            with torch.no_grad():
                self.memory_bank = self.memory_bank.cpu()
                updated_memory = self.memory_bank.read()
                memo_roi_feature, memo_proposals, memo_rel_pair, memo_union_feat = updated_memory

                mem_rel_rep_list = []

                for m_roi, m_proposal, m_rel_pair, m_union in zip(memo_roi_feature, memo_proposals, memo_rel_pair,
                                                                  memo_union_feat):
                    if isinstance(m_roi, list):
                        mem_rel_rep_list.append(torch.zeros((2048,)).cuda())
                        continue

                    m_roi = m_roi.to("cuda")
                    m_proposal_list = [m_pro.to("cuda") for m_pro in m_proposal]
                    m_pair = m_rel_pair.to("cuda")
                    m_union = m_union.to("cuda")

                    _, mem_entity_preds = self.refine_obj_labels(m_roi, m_proposal_list)
                    mem_entity_rep = self.post_emb(m_roi)
                    mem_entity_rep = mem_entity_rep.view(mem_entity_rep.size(0), 2, self.mlp_dim)

                    mem_sub_rep = mem_entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)  # xs
                    mem_obj_rep = mem_entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)  # xo

                    mem_entity_embeds = self.obj_embed(mem_entity_preds)

                    # num_objs = [len(b) for b in m_proposal_list]
                    # num_rels = [r.shape[0] for r in m_pair]

                    # mem_sub_reps = mem_sub_rep.split(num_objs, dim=0)
                    # mem_obj_reps = mem_obj_rep.split(num_objs, dim=0)
                    # mem_entity_preds = mem_entity_preds.split(num_objs, dim=0)
                    # mem_entity_embeds = mem_entity_embeds.split(num_objs, dim=0)

                    mem_s_embed = self.W_sub(mem_entity_embeds[m_pair[:, 0]])  # Ws x ts
                    mem_o_embed = self.W_obj(mem_entity_embeds[m_pair[:, 1]])  # Wo x to

                    mem_sem_sub = self.vis2sem(mem_sub_rep[m_pair[:, 0]])  # h(xs)
                    mem_sem_obj = self.vis2sem(mem_obj_rep[m_pair[:, 1]])  # h(xo)

                    gate_sem_sub = torch.sigmoid(self.gate_sub(cat((mem_s_embed, mem_sem_sub), dim=-1)))  # gs
                    gate_sem_obj = torch.sigmoid(self.gate_obj(cat((mem_o_embed, mem_sem_obj), dim=-1)))  # go

                    sub = mem_s_embed + mem_sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
                    obj = mem_o_embed + mem_sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

                    ##### for the model convergence
                    sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
                    obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
                    #####
                    fusion_so = fusion_func(sub, obj)

                    sem_pred = self.vis2sem(self.down_samp(m_union))  # h(xu)
                    gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

                    mem_rel_rep = fusion_so - sem_pred * gate_sem_pred  # F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
                    mem_rel_rep_list.append(mem_rel_rep.mean(dim=0))

                mem_rel_reps = torch.stack(mem_rel_rep_list, dim=0).cuda()
                del mem_rel_rep_list

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        #####

        entity_rep = self.post_emb(roi_features)  # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)  # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)  # xo

        entity_embeds = self.obj_embed(entity_preds)  # obtaining the word embedding of entities with GloVe

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed in zip(rel_pair_idxs, sub_reps, obj_reps,
                                                                         entity_preds, entity_embeds):
            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  # Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  # Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)

            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj))  # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  # F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up

        # feature update
        # query: rel_rep, key: mem_rel_reps, value: mem_rel_reps
        if self.is_timing:
            project_rel_rep = self.ffn(rel_rep).unsqueeze(0)
            project_mem_rel_rep = self.ffn(mem_rel_reps).unsqueeze(0)
            attn_mem_rep, _ = self.attn(project_rel_rep, project_mem_rel_rep, project_mem_rel_rep)
            fusion_mem_rep = F.normalize(self.ffn(attn_mem_rep)).squeeze(0)
            rel_rep = 0.5 * rel_rep + 0.5 * fusion_mem_rep

        predicate_proto = self.W_pred(self.predicate_proto)
        predicate_proto_sub = self.W_pred(self.codebook._C)  # [307, 2048]
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        predicate_proto_sub = self.project_head(self.dropout_pred(torch.relu(predicate_proto_sub)))

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm [N, 4096]

        rel_dists_sub = rel_rep_norm @ predicate_proto_sub.t() * self.logits_scale.exp()  # [N,102]
        rel_dists_sub_gumbel = F.gumbel_softmax(rel_dists_sub, dim=1)
        rel_dists_tuple = rel_dists_sub.split(self.pred_diversity_list, dim=1)  # [N,51,*]

        rel_dists = []
        for rel_dist in rel_dists_tuple:
            # rel_dist_mask = F.softmax(rel_dist.detach() * self.softmax_mask.exp(), dim=1)  # [N,1/2]
            rel_dist_mask = F.softmax(rel_dist.detach() / self.curr_temp, dim=1)
            rel_dist = torch.sum(rel_dist * rel_dist_mask, dim=1)  # [N,1/2]
            rel_dists.append(rel_dist.unsqueeze(dim=1))  # [N,1/2]
        rel_dists = cat(rel_dists, dim=1)  # [N,51]

        entity_dists = entity_dists.split(num_objs, dim=0)  # (N, ): [N_obj, 151]
        rel_dists = rel_dists.split(num_rels, dim=0)  # (N, ): [N_rel, 102]
        rel_dists_sub = rel_dists_sub.split(num_rels, dim=0)

        add_losses = {}
        if self.training:
            # write memory
            if self.is_timing:
                with torch.no_grad():
                    roi_features_objs = roi_features.split(num_objs, dim=0)
                    union_features_rels = union_features.split(num_rels, dim=0)
                    self.memory_bank.write(roi_features_objs, proposals, rel_pair_idxs, rel_dists_sub,
                                           union_features_rels, sub_rel_labels)

            target_predicate_proto = predicate_proto.clone().detach()  # [51, 2048]
            orthogonalities_sub_general = predicate_proto_sub @ target_predicate_proto.t()  # [307, 51]
            orthogonalities_sub_general_negative = orthogonalities_sub_general.mul(
                self.mask_general_sub_matching)  # [102, 51]
            orthogonalities_sub_general_loss = torch.norm(torch.norm(orthogonalities_sub_general_negative, p=2, dim=1),
                                                          p=1) / (self.num_rel_cls * self.num_sub_proto)
            add_losses.update({"sub_general_loss": orthogonalities_sub_general_loss})

            target_predicate_proto_sub = predicate_proto_sub.clone().detach()  # [102, 4096]
            orthogonalities_sub = predicate_proto_sub @ target_predicate_proto_sub.t()  # [102, 102]
            orthogonalities_sub_negative = orthogonalities_sub.mul(self.negative_mask_sub)  # [102, 51]
            orthogonalities_sub_loss_negative = torch.norm(torch.norm(orthogonalities_sub_negative, p=2, dim=1), p=1) / \
                                                (self.num_sub_proto * self.num_sub_proto)
            add_losses.update({"p_sub_loss_cs": orthogonalities_sub_loss_negative})

            rel_labels_tensor = cat(rel_labels, dim=0)
            rel_rep_list = []
            rel_labels_list = []
            for i in range(len(rel_labels_tensor)):
                if rel_labels_tensor[i] != 0:
                    rel_rep_list.append(rel_rep[i])
                    rel_labels_list.append(rel_labels_tensor[i])
            rel_rep_tensor = torch.stack(rel_rep_list, dim=0)
            rel_labels_tensors = torch.stack(rel_labels_list, dim=0)
            positive_mask = torch.ones((self.num_rel_cls, self.num_rel_cls), requires_grad=False).cuda()
            for i in range(self.num_rel_cls):
                positive_mask[i, i] = 0
            positive_mask[:, 0] = 0
            positive_mask[0, :] = 0
            positive_mask_batch = torch.ones((len(rel_labels_tensors), len(rel_labels_tensors)),
                                             requires_grad=False).cuda()
            for i in range(len(rel_labels_tensors)):
                positive_mask_batch[i, i] = 0
            target_rel_rep = rel_rep_tensor.clone().detach()
            rep_matrix = (rel_rep_tensor @ target_rel_rep.t()).mul(positive_mask_batch)
            rep_matrix_neg = torch.zeros((self.num_rel_cls, self.num_rel_cls), requires_grad=False).cuda()
            count_matrix_neg = torch.zeros((self.num_rel_cls, self.num_rel_cls), requires_grad=False).cuda()
            for i in range(len(rel_labels_tensors)):
                for j in range(len(rel_labels_tensors)):
                    rep_matrix_neg[rel_labels_tensors[i], rel_labels_tensors[j]] += rep_matrix[i, j]
                    count_matrix_neg[rel_labels_tensors[i], rel_labels_tensors[j]] += 1
            rep_matrix_neg /= (count_matrix_neg + 0.001)
            rep_matrix_neg = rep_matrix_neg.mul(positive_mask)
            orthogonalities_rep_loss = torch.norm(torch.norm(rep_matrix_neg, p=2, dim=1), p=1) / \
                                       (len(rel_labels_tensors) * len(rel_labels_tensors))
            add_losses.update({"rep_loss_neg": 0.1 * orthogonalities_rep_loss})
        return entity_dists, rel_dists, add_losses

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()

        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from .utils_prototype import get_semantic_diversity,cal_kl
def predicate_statistics(predicate_proportion, predicate_count, pred_weight_beta,id2pred):
    if id2pred == None:
        mean_pred = predicate_count[0] * 2
        min_value = 0.01
        predicate_proportion.append((1.0 - pred_weight_beta) / (1.0 - pred_weight_beta ** mean_pred))
        for i in range(1,len(predicate_count),1):
            if predicate_count[i]==0:
                predicate_count[i] = min_value
            predicate_proportion.append((1.0 - pred_weight_beta) / (1.0 - pred_weight_beta ** predicate_count[i]))
        predicate_proportion = predicate_proportion / np.sum(predicate_proportion) * len(predicate_proportion)
        pred_weight = (torch.FloatTensor(predicate_proportion)).cuda()
        return pred_weight
    else:
        mean_pred = np.sum(np.array(list(predicate_count.values()))) * 2
        predicate_proportion.append((1.0 - pred_weight_beta) / (1.0 - pred_weight_beta ** mean_pred))
        for i in range(len(predicate_count)):
            predicate_proportion.append(
                (1.0 - pred_weight_beta) / (1.0 - pred_weight_beta ** predicate_count[id2pred[str(i + 1)]]))
        predicate_proportion = predicate_proportion / np.sum(predicate_proportion) * len(predicate_proportion)
        pred_weight = (torch.FloatTensor(predicate_proportion)).cuda()
        return pred_weight#, dict
class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        use_class_balance,
        predicate_proportion,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.use_class_balance = use_class_balance
        # class balanced loss
        if self.use_class_balance:
            self.pred_weight = torch.FloatTensor([0.0418, 0.1109, 2.1169, 1.3740, 1.1754, 1.2242, 0.4437, 0.4318, 0.1235,
            1.2934, 1.2475, 0.8236, 1.8305, 1.1202, 0.9135, 2.1414, 0.4787, 1.4416,
            2.2787, 0.4441, 0.0446, 0.1203, 0.0455, 0.3264, 1.1399, 1.3780, 2.2594,
            1.7788, 1.8779, 0.0675, 0.0544, 0.0419, 2.2067, 0.4703, 1.3727, 1.5585,
            2.0469, 1.1191, 0.4936, 1.8878, 0.2460, 0.3163, 1.6831, 0.2068, 2.1942,
            2.4253, 0.9280, 1.2198, 0.0563, 0.2921, 0.0862]).cuda()
        else:
            self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()

        self.pred_diversity_list = get_semantic_diversity()
        self.num_sub_proto = int(sum(self.pred_diversity_list))
        self.iter = 0
        self.max_temp = 1
        self.curr_temp = self.max_temp
        self.min_temp = 0.1
        self.temp_decay = 0.9998
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(self.max_temp),requires_grad=False) # 温度系数越小，越陡峭,方差越大
        self.cof_fg_matching = 0.00001
        self.cof_uniform_matching = 1.0
        self.uniform_decay = 0.999995
        self.max_cof_uniform = 1.0
        self.min_cof_uniform = 0.5
        self.curr_cof_uniform = 0
        self.softmax_mask = nn.Parameter(torch.ones([]) * np.log(100000000000),requires_grad=False)
        import json
        predicate_proportion = []
        vg_dict = json.load(open('/mnt/hdd1/zhanghaonan/code/code_sgg/vg//VG-SGG-dicts-with-attri.json', 'r'))
        id2pred = vg_dict['idx_to_predicate']
        predicate_count = vg_dict['predicate_count']
        self.pred_weight = predicate_statistics(predicate_proportion, predicate_count, 0.9999899, id2pred)
    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):
    # def __call__(self, proposals, rel_labels, relation_logits, refine_logits,sub_rel_labels):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        self.iter+=1

        self.uniform_cof_decay_func(1)

        refine_obj_logits = refine_logits
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)
        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)
        loss_relation = F.cross_entropy(relation_logits, rel_labels.long(), weight=self.pred_weight)
        #loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())
        return loss_relation, loss_refine_obj
    def temp_decay_func(self, num_updates):
        self.curr_temp = max(
            self.curr_temp * self.temp_decay ** num_updates, self.min_temp
        )
    def uniform_cof_decay_func(self, num_updates):
        self.curr_cof_uniform = max(
            self.cof_uniform_matching * self.uniform_decay ** num_updates, self.min_cof_uniform
        )
    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        attri_on=cfg.MODEL.ATTRIBUTE_ON,
        num_attri_cat=cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        max_num_attri=cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        attribute_sampling=cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        attribute_bgfg_ratio=cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        use_label_smoothing=cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        use_class_balance=cfg.MODEL.ROI_RELATION_HEAD.CLASS_BALANCE_LOSS,
        predicate_proportion=cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
    )

    return loss_evaluator

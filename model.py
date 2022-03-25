
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import copy
import math

from centroid import Centroid
from itertools import permutations
from itertools import combinations

# definition of Gradient Reversal Layer
class GradRevLayer(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


# definition of Adversarial Domain Classifier (base part)
class AdvDomainClsBase(nn.Module):
    def __init__(self, in_feat, hidden_size, type_adv, args):
        super(AdvDomainClsBase, self).__init__()
        # ====== collect arguments ====== #
        self.num_f_maps = args.num_f_maps
        self.DA_adv_video = args.DA_adv_video
        self.pair_ssl = args.pair_ssl
        self.type_adv = type_adv

        # ====== main architecture ====== #
        if self.type_adv == 'video' and self.DA_adv_video == 'rev_grad_ssl_2':
            self.fc_pair = nn.Linear(self.num_f_maps * 2, self.num_f_maps)

        self.fc1 = nn.Linear(in_feat, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, input_data, beta):
        feat = GradRevLayer.apply(input_data, beta)
        if self.type_adv == 'video' and self.DA_adv_video == 'rev_grad_ssl_2':
            num_seg = int(input_data.size(-1)/self.num_f_maps)
            feat = feat.reshape(-1, num_seg, self.num_f_maps)  # reshape --> (video#, seg#, dim)

            # get the pair indices
            id_pair = torch.tensor(list(combinations(range(num_seg), 2))).long()  # all possible indices
            if self.pair_ssl == 'adjacent':
                id_pair = torch.tensor([(i, i + 1) for i in range(num_seg-1)])
            if input_data.get_device() >= 0:
                id_pair = id_pair.to(input_data.get_device())

            # get the pairwise features
            feat = feat[:, id_pair, :]  # (video#, pair#, 2, dim)
            feat = feat.reshape(-1, self.num_f_maps*2)  # (video# x pair#, 2 x dim)
            feat = self.fc_pair(feat)  # (video# x pair#, dim)
            feat = feat.reshape(-1, id_pair.size(0)*self.num_f_maps)  # (video#, pair# x dim)

        feat = self.fc1(feat)
        feat = self.relu(feat)
        feat = self.dropout(feat)  

        return feat


# definition of MS-TCN
class MultiStageModel(nn.Module):
    def __init__(self, args, num_classes):
        super(MultiStageModel, self).__init__()
        # ====== collect arguments ====== #
        # this function only
        num_stages = args.num_stages
        num_layers = args.num_layers
        num_f_maps = args.num_f_maps
        dim_in = args.features_dim
        method_centroid = args.method_centroid

        # cross-function
        self.use_target = args.use_target
        self.multi_adv = args.multi_adv
        self.DA_adv_video = args.DA_adv_video
        self.ps_lb = args.ps_lb
        self.use_attn = args.use_attn
        self.num_seg = args.num_seg
        self.pair_ssl = args.pair_ssl
        self.DA_ens = args.DA_ens
        self.SS_video = args.SS_video
        self.fore_attn = args.fore_attn

        # ====== main architecture ====== #
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim_in, num_classes, self.DA_ens)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, self.DA_ens)) for s in range(num_stages-1)])
        self.stage1_attn = ForegroundAttentionModule(num_f_maps)

        # domain discriminators
        self.ad_net_base = nn.ModuleList()
        self.ad_net_base += [AdvDomainClsBase(num_f_maps, num_f_maps, 'frame', args)]
        self.ad_net_cls = nn.ModuleList()
        self.ad_net_cls += [nn.Linear(num_f_maps, 2)]

        # domain discriminators (video-level)
        if 'rev_grad' in self.DA_adv_video and self.use_target != 'none':
            num_domain_class = 2
            num_concat = 1
            if 'rev_grad_ssl' in self.DA_adv_video:
                num_domain_class = int(math.factorial(self.num_seg*2)/(math.factorial(self.num_seg)**2))

                num_concat = self.num_seg * 2
                if self.DA_adv_video == 'rev_grad_ssl_2':
                    if self.pair_ssl == 'all':
                        num_concat = int(math.factorial(self.num_seg * 2) / (2 * math.factorial(self.num_seg * 2 - 2)))
                    elif self.pair_ssl == 'adjacent':
                        num_concat = self.num_seg * 2 - 1

            self.ad_net_video_base = nn.ModuleList()
            self.ad_net_video_base += [AdvDomainClsBase(num_f_maps * num_concat, num_f_maps, 'video', args)]
            self.ad_net_video_cls = nn.ModuleList()
            self.ad_net_video_cls += [nn.Linear(num_f_maps, num_domain_class)]

        # video-order classifier
        if self.SS_video == 'VCOP':
            num_order_pair = int(self.num_seg * (self.num_seg - 1) / 2)
            num_order_class = math.factorial(self.num_seg)
            self.video_order_base = nn.Sequential(
                nn.Linear(num_f_maps*2, num_f_maps),
                nn.ReLU(),
                nn.Dropout()
            )
            self.video_order_cls = nn.Linear(num_f_maps * num_order_pair, num_order_class)

        # for class-based domain discriminators (frame-level only)
        if self.multi_adv[1] == 'Y':  # separate weights for domain classifiers
            for i in range(1, num_classes):
                self.ad_net_cls += [nn.Linear(num_f_maps, 2)]

            # if separating feature weights, classifier weights must be separate
            if self.multi_adv[0] == 'Y':  # separate weights for domain features
                for i in range(1, num_classes):
                    self.ad_net_base += [AdvDomainClsBase(num_f_maps, num_f_maps, 'frame', args)]

        # store the centroids
        if method_centroid != 'none':
            self.centroids = nn.ModuleList()
            for s in range(num_stages):
                self.centroids += [Centroid(num_f_maps, num_classes)]


    def forward(self, x_s, x_t, mask_s, mask_t, beta, reverse):
        # forward source & target data at the same time
        pred_source, a_pred_source, f_pred_source, prob_source, feat_source, feat_source_video, pred_d_source, pred_d_source_video, label_d_source, label_d_video_source, pred_source_2, prob_source_2 \
            = self.forward_domain(x_s, mask_s, 0, beta, reverse)
        pred_target, _, f_pred_target, prob_target, feat_target, feat_target_video, pred_d_target, pred_d_target_video, label_d_target, label_d_video_target, pred_target_2, prob_target_2  \
            = self.forward_domain(x_t, mask_t, 1, beta, reverse)

        # concatenate domain predictions & labels (frame-level)
        pred_d = torch.cat((pred_d_source, pred_d_target), 0)
        label_d = torch.cat((label_d_source, label_d_target), 0).long()

        # concatenate domain predictions & labels (video-level)
        pred_d_video = torch.cat((pred_d_source_video, pred_d_target_video), 0)
        label_d_video = torch.cat((label_d_video_source, label_d_video_target), 0).long()

        # self-supervised temporal domain adaptation 
        #  ------ Global SSTDA ------ #
        if 'rev_grad_ssl' in self.DA_adv_video and self.use_target != 'none':
            # get the list of permutation for source & target segments
            label_d_all = ([0] + [1]) * self.num_seg  # [0, 1, 0, 1]
            list_label_d = torch.tensor(list(set(permutations(label_d_all))))  # e.g. [[1,0,1,0],[1,1,0,0],...]  (domain_class#, seg#)
            if label_d_video.get_device() >= 0:
                list_label_d = list_label_d.to(label_d_video.get_device())

            # concatenate source & target
            feat_video = torch.cat((feat_source_video, feat_target_video), -1)  # (batch, stage#, dim, seg#x2)
            num_batch = feat_video.size(0)  # num_batch = batch x seg#x2
            pred_d_video_ssl_single, label_d_video_ssl_single = self.predict_domain_video_ssl(feat_video[0], list_label_d, beta[1])
            pred_d_video_ssl = pred_d_video_ssl_single.unsqueeze(0)  # (batch x seg#x2, stage#, domain_class#)
            label_d_video_ssl = label_d_video_ssl_single.unsqueeze(0)  # (batch x seg#x2, stage#)
            for i in range(1, num_batch):
                pred_d_video_ssl_single, label_d_video_ssl_single = self.predict_domain_video_ssl(feat_video[i], list_label_d, beta[1])
                pred_d_video_ssl = torch.cat((pred_d_video_ssl, pred_d_video_ssl_single.unsqueeze(0)), dim=0)
                label_d_video_ssl = torch.cat((label_d_video_ssl, label_d_video_ssl_single.unsqueeze(0)), dim=0)

            # replace the original pred_d_video & label_d_video
            pred_d_video = pred_d_video_ssl
            label_d_video = label_d_video_ssl

        # self-supervised learning for videos
        if self.SS_video == 'VCOP' and self.use_target != 'none':
            # get the list of order for video segments
            label_order_all = list(range(self.num_seg))
            list_label_order = torch.tensor(list(permutations(label_order_all)))  # e.g. [[0,1,2],[1,0,2],...]
            if label_d_video.get_device() >= 0:
                list_label_order = list_label_order.to(label_d_video.get_device())

            # collect inputs
            feat_video = torch.cat((feat_source_video, feat_target_video), 0)  # (batchx2, stage#, dim, seg#)
            num_batch = feat_video.size(0)
            pred_order_video_ssl_single, label_order_video_ssl_single = self.predict_order_video_ssl(feat_video[0], list_label_order)
            pred_order_video_ssl = pred_order_video_ssl_single.unsqueeze(0)
            label_order_video_ssl = label_order_video_ssl_single.unsqueeze(0)
            for i in range(1, num_batch):
                pred_order_video_ssl_single, label_order_video_ssl_single = self.predict_order_video_ssl(feat_video[i], list_label_order)
                pred_order_video_ssl = torch.cat((pred_order_video_ssl, pred_order_video_ssl_single.unsqueeze(0)), dim=0)
                label_order_video_ssl = torch.cat((label_order_video_ssl, label_order_video_ssl_single.unsqueeze(0)), dim=0)

            # replace the original pred_d_video & label_d_video
            pred_d_video = pred_order_video_ssl
            label_d_video = label_order_video_ssl

        return pred_source, a_pred_source, f_pred_source, f_pred_target, prob_source, feat_source, pred_target, prob_target, feat_target, \
               pred_d, pred_d_video, label_d, label_d_video, \
               pred_source_2, prob_source_2, pred_target_2, prob_target_2

    def forward_domain(self, x, mask, domain_GT, beta, reverse):
        out_feat = self.stage1(x)  # (batch, dim, frame#) 

        # ------ Contextual Foreground Attention ------ # 
        out_feat_f, score_f = self.stage1_attn(out_feat)  # out_feat_f: (batch, dim, frame#) / score_f: (batch, frame#)

        if reverse:  # reverse the gradient
            out_feat = GradRevLayer.apply(out_feat, beta[0])

        # predictions
        out = self.stage1.conv_out(out_feat)  # (batch, class#, frame#)
        # predictions from contextual foreground features
        out_a = self.stage1.conv_out(out_feat_f)  # (batch, class#, frame#)  

        out_2 = out.clone()
        if self.DA_ens != 'none':
            out_2 = self.stage1.conv_out_2(out_feat)  # out: (batch, class#, frame#)
        prob = F.softmax(out, dim=1)  # (batch, class#, frame#)
        prob_2 = F.softmax(out_2, dim=1)  # (batch, class#, frame#)

        # compute domain predictions for single stage
        out_d, out_d_video, lb_d, lb_d_video, out_feat_video \
            = self.forward_stage(out_feat, out_feat_f, prob, beta, mask, domain_GT)
            
        # store outputs
        outputs_feat = out_feat.unsqueeze(1)  # (batch, stage#, dim, frame#)
        outputs_feat_video = out_feat_video.unsqueeze(1)  # (batch, stage#, dim, seg#)
        outputs = out.unsqueeze(1)  # (batch, stage#, class#, frame#)
        outputs_a = out_a.unsqueeze(1)  # (batch, stage#, class#, frame#)
        outputs_f = score_f.unsqueeze(1)  # (batch, stage#, frame#)
        probs = prob.unsqueeze(1)  # prob: (batch, stage#, class#, frame#)
        outputs_2 = out_2.unsqueeze(1)  # (batch, stage#, class#, frame#)
        probs_2 = prob_2.unsqueeze(1)  # prob: (batch, stage#, class#, frame#)

        outputs_d = out_d.unsqueeze(1)  # (batch x frame#, stage#, class#, 2)
        outputs_d_video = out_d_video.unsqueeze(1)  # (batch x seg#, stage#, 2)
        labels_d = lb_d.unsqueeze(1)  # (batch x frame#, stage#, class#)
        labels_d_video = lb_d_video.unsqueeze(1)  # (batch x seg#, stage#)

        for s in self.stages:
            out_feat = s(prob)  

            # ------ Contextual Foreground Attention ------ #
            out_feat_f, score_f, = self.stage1_attn(out_feat)  # out_feat_f: (batch, dim, frame#) / score_f: (batch, frame#)

            if reverse:  # reverse the gradient
                out_feat = GradRevLayer.apply(out_feat, beta[0])

            # prediction
            out = s.conv_out(out_feat)  # (batch, class#, frame#) 
            # prediction from contextual foreground features
            out_a = s.conv_out(out_feat_f)  # (batch, class#, frame#) 

            out_2 = out.clone()  
            if self.DA_ens != 'none':
                out_2 = s.conv_out_2(out_feat)  # (batch, class#, frame#)
            prob = F.softmax(out, dim=1)  # (batch, class#, frame#)
            prob_2 = F.softmax(out_2, dim=1)  # (batch, class#, frame#)

            # compute domain predictions for single stage
            out_d, out_d_video, lb_d, lb_d_video, out_feat_video \
                = self.forward_stage(out_feat, out_feat_f, prob, beta, mask, domain_GT)
               
            # store outputs
            outputs_feat = torch.cat((outputs_feat, out_feat.unsqueeze(1)), dim=1)
            outputs_feat_video = torch.cat((outputs_feat_video, out_feat_video.unsqueeze(1)), dim=1)
            outputs = torch.cat((outputs, out.unsqueeze(1)), dim=1)
            outputs_a = torch.cat((outputs_a, out_a.unsqueeze(1)), dim=1)
            outputs_f = torch.cat((outputs_f, score_f.unsqueeze(1)), dim=1)
            probs = torch.cat((probs, prob.unsqueeze(1)), dim=1)
            outputs_2 = torch.cat((outputs_2, out_2.unsqueeze(1)), dim=1)
            probs_2 = torch.cat((probs_2, prob_2.unsqueeze(1)), dim=1)

            outputs_d = torch.cat((outputs_d, out_d.unsqueeze(1)), dim=1)
            outputs_d_video = torch.cat((outputs_d_video, out_d_video.unsqueeze(1)), dim=1)
            labels_d = torch.cat((labels_d, lb_d.unsqueeze(1)), dim=1)
            labels_d_video = torch.cat((labels_d_video, lb_d_video.unsqueeze(1)), dim=1)

        return outputs, outputs_a, outputs_f, probs, outputs_feat, outputs_feat_video, outputs_d, outputs_d_video, labels_d, labels_d_video, outputs_2, probs_2

    def forward_stage(self, out_feat, out_feat_f, prob, beta, mask, domain_GT):
        # === Produce domain predictions === #
        # ------ Local SSTDA ------ #
        if self.fore_attn == 'Y':
            out_d = self.predict_domain_frame(out_feat_f, beta[0])  # (batch, class#, 2, frame#) 
        else:
            out_d = self.predict_domain_frame(out_feat, beta[0])  # (batch, class#, 2, frame#) 

        ## Domain Attentive Temporal Pooling
        # apply attention to frame-level features if domain_attn
        out_feat_video = out_feat
        if self.use_attn == 'domain_attn' and self.use_target != 'none':
            out_feat_video = self.apply_attn_feat_frame(out_feat_video, out_d, prob, 'ver2')

        # --- video-level --- #
        # video-level feature (temporal pooling) ==> need to consider the frame mask
        out_feat_video = self.aggregate_frames(out_feat_video, mask)  # (batch, dim, seg#)

        # video-wisely apply ad_net_video
        # 1. naive initialization (only the shape is correct)
        out_d_video = out_d[:, :, :, :self.num_seg].mean(1)  # (batch, 2, seg#)

        # 2. video-level DA (binary domain prediction)
        if self.DA_adv_video == 'rev_grad' and self.use_target != 'none':  # no multi-adv
            out_d_video = self.predict_domain_video(out_feat_video, beta[0])  # (batch, 2, seg#)

        # === Select valid frames + Generate domain labels === #
        out_d, out_d_video, lb_d, lb_d_video = self.select_masked(out_d, mask, out_d_video, domain_GT)

        return out_d, out_d_video, lb_d, lb_d_video, out_feat_video

    def predict_domain_frame(self, feat, beta_value):
        # feat: (batch, dim, frame#)
        dim_feat = feat.size(1)
        num_frame = feat.size(2)
        feat = feat.transpose(1, 2).reshape(-1, dim_feat)    # reshape to (batch x frame#, dim)
        out = self.ad_net_cls[0](self.ad_net_base[0](feat, beta_value))  # (batch x frame#, 2)
        out = nn.Softmax(dim=-1)(out)
        out = out.reshape(-1, num_frame, 2).transpose(1, 2)  # reshape back to (batch, 2, frame#)
        out = out.unsqueeze(1)  # (batch, 1, 2, frame#)

        if self.multi_adv[1] == 'Y':  # class-based domain discriminators w/ pseudo-labels
            for i in range(1, len(self.ad_net_cls)):
                id_base = i if self.multi_adv[0] == 'Y' else 0  # decide whether to separate weights for discriminator base
                out_single_class = self.ad_net_cls[i](self.ad_net_base[id_base](feat, beta_value))
                out_single_class = out_single_class.reshape(-1, num_frame, 2).transpose(1, 2)  # (batch, 2, frame#)
                out = torch.cat((out, out_single_class.unsqueeze(1)), dim=1)  # (batch, class#, 2, frame#)

        return out

    def get_domain_attn(self, pred_domain, type_attn):
        # pred_domain: (batch, class#, 2, frame#)
        dim_pred = 2
        softmax = nn.Softmax(dim=dim_pred)
        logsoftmax = nn.LogSoftmax(dim=dim_pred)
        entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), dim_pred)
        if type_attn == 'ver1':
            weights = entropy
        elif type_attn == 'ver2':
            weights = 1 - entropy

        return weights

    def apply_attn_feat_frame(self, feat, pred_domain, prob, type_attn):
        # feat: (batch, dim, frame#) / pred_domain: (batch, class#, 2, frame#) / prob: (batch, class#, frame#)
        weights_attn = self.get_domain_attn(pred_domain, type_attn)  # (batch, class#, frame#)

        if self.multi_adv[1] == 'Y':  # class-based domain discriminators w/ pseudo-labels
            # get weighting from prob
            classweight = prob.detach()
            classweight_hard = classweight == classweight.max(dim=1, keepdim=True)[0]  # highest prob: 1, others: 0
            classweight_hard = classweight_hard.float()

            if self.ps_lb == 'soft':
                weights_attn *= classweight
            elif self.ps_lb == 'hard':
                weights_attn *= classweight_hard

        weights_attn = weights_attn.unsqueeze(2).repeat(1, 1, feat.size(1), 1)  # (batch, class#, dim, frame#)
        feat_expand = feat.unsqueeze(1).repeat(1, weights_attn.size(1), 1, 1)  # (batch, class#, dim, frame#)
        feat_attn = ((weights_attn + 1) * feat_expand).sum(1)  # (batch, dim, frame#)

        return feat_attn

    def aggregate_frames(self, out_feat, mask):
        dim_feat = out_feat.size(1)
        num_batch = out_feat.size(0)

        # calculate total frame# for each video
        num_total_frame = mask[:, 0, :].sum(-1)

        # make sure the total frame# can be divided by seg#
        num_frame_seg = (num_total_frame / self.num_seg).int()

        num_frame_new = self.num_seg * num_frame_seg

        # reshape frame-level features based on num_seg --> aggregate frames
        out_feat_video_batch = out_feat[0, :, :num_frame_new[0]].reshape(dim_feat, self.num_seg, num_frame_seg[0])  # (dim, seg#, seg_frame#)
        out_feat_video_batch = out_feat_video_batch.sum(-1)/num_frame_seg[0]  # average all the features in a segment ==> (dim, seg#)
        out_feat_video = out_feat_video_batch.unsqueeze(0)  # (1, dim, seg#)
        for b in range(1, num_batch):
            out_feat_video_batch = out_feat[b, :, :num_frame_new[b]].reshape(dim_feat, self.num_seg, num_frame_seg[b])
            out_feat_video_batch = out_feat_video_batch.sum(-1)/(num_frame_seg[b].float())
            out_feat_video = torch.cat((out_feat_video, out_feat_video_batch.unsqueeze(0)), dim=0)  # (batch, dim, seg#)

        return out_feat_video

    def predict_domain_video(self, feat, beta_value):
        dim_feat = feat.size(1)
        num_seg = feat.size(2)
        feat = feat.transpose(1, 2).reshape(-1, dim_feat)  # reshape to (batch x seg#, dim)
        out = self.ad_net_video_cls[0](self.ad_net_video_base[0](feat, beta_value))  # (batch x seg#, 2)
        out = out.reshape(-1, num_seg, 2).transpose(1, 2)  # reshape back to (batch, 2, seg#)

        return out

    def select_masked(self, out_d, mask, out_d_video, domain_GT):
        # --- frame-level --- #
        # reshape --> (batch x frame#, ...)
        num_class_domain = out_d.size(1)
        out_d = out_d.transpose(2, 3).transpose(1, 2).reshape(-1, num_class_domain, 2)  # (batch x frame#, class#, 2)

        # select frames w/ mask + generate frame-level domain labels
        mask_frame = mask[:, 0, :].reshape(-1)  # (batch x frame#)
        mask_frame = mask_frame > 0
        out_d = out_d[mask_frame]  # (batch x valid_frame#, class#, 2)
        lb_d = torch.full_like(out_d[:, :, 0], domain_GT)  # (batch x valid_frame#, class#)

        # --- video-level --- #
        # reshape --> (batch x seg#, ...)
        out_d_video = out_d_video.transpose(1, 2).reshape(-1, 2)  # (batch x seg#, 2)
        lb_d_video = torch.full_like(out_d_video[:, 0], domain_GT)  # (batch x seg#)

        return out_d, out_d_video, lb_d, lb_d_video

    def predict_domain_video_ssl(self, feat, list_label_d_seg, beta_value):
        # feat: (stage#, dim, seg#) / list_label_d: (domain_class#, seg#)
        num_stage = feat.size(0)
        dim_feat = feat.size(1)
        num_seg = feat.size(2)

        # random sorting + generate labels
        id_new = torch.randperm(num_seg)
        feat = feat[:, :, id_new]
        label_d_seg = (id_new >= (num_seg/2)).long()  # e.g. [1, 1, 0, 0]
        if feat.get_device() >= 0:
            label_d_seg = label_d_seg.to(feat.get_device())
        label_d_onehot = (list_label_d_seg == label_d_seg).sum(-1) == num_seg  # all elements need to be correct
        label_d = label_d_onehot.nonzero()  # e.g. [0, 0, 1, 0, 0, 0] --> [[2]]
        label_d = label_d.reshape(-1).repeat(num_stage)  # e.g. [[2]] --> [2] --> [2, 2, 2, 2]

        # directly concatenate source & target ==> long vector for each video
        feat = feat.transpose(1, 2).reshape(-1, num_seg*dim_feat)  # reshape to (stage#, seg# x dim)

        out = self.ad_net_video_cls[0](self.ad_net_video_base[0](feat, beta_value))  # out: (stage#, domain_class#)

        return out, label_d

    def predict_order_video_ssl(self, feat, list_label_order_seg):
        # feat: (stage#, dim, seg#) / list_label_d: (domain_class#, seg#)
        num_stage = feat.size(0)
        # dim_feat = feat.size(1)
        num_seg = feat.size(2)

        # shuffling + generate labels
        id_new = torch.randperm(num_seg)  # e.g. [0, 2, 1]
        feat = feat[:, :, id_new]
        label_order_seg = id_new.long()  # e.g. [0, 2, 1]
        if feat.get_device() >= 0:
            label_order_seg = label_order_seg.to(feat.get_device())
        label_order_onehot = (list_label_order_seg == label_order_seg).sum(-1) == num_seg  # all elements need to be correct
        label_order = label_order_onehot.nonzero()  # e.g. [0, 0, 1, 0, 0, 0] --> [[2]]
        label_order = label_order.reshape(-1).repeat(num_stage)  # e.g. [[2]] --> [2] --> [2, 2, 2, 2]

        # pairwise concatenation
        feat = feat.transpose(1, 2).transpose(0, 1)  # reshape to (seg#, stage#, dim)
        feat_pair = []
        for i in range(num_seg):
            for j in range(i+1, num_seg):
                feat_pair.append(torch.cat((feat[i], feat[j]), -1))

        # feed to fc separately --> concat
        feat_pair = [self.video_order_base(i) for i in feat_pair]  # [(stage#, dim)] x pair#
        feat_concat = torch.cat(feat_pair, dim=1)  # (stage#, dim x pair#)
        out = self.video_order_cls(feat_concat)  # out: (stage#, order_class#)

        return out, label_order


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim_in, num_classes, DA_ens):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim_in, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

        # for ensemble methods
        if DA_ens != 'none':
            self.conv_out_2 = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out_feat = self.conv_1x1(x)
        for layer in self.layers:
            out_feat = layer(out_feat)
        return out_feat 


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)  
        return x + out


class ForegroundAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ForegroundAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels

        self.query_enc = nn.Conv1d(self.inter_channels, self.inter_channels, kernel_size=1) 
        self.key_enc = nn.Conv1d(self.inter_channels, self.inter_channels, kernel_size=1) 
        self.value_enc = nn.Conv1d(self.inter_channels, self.inter_channels, kernel_size=1) 

        self.smooth = nn.Conv1d(self.inter_channels, self.inter_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.linear = nn.Linear(self.inter_channels, 1)

    def forward(self, feat):
        batch_size = feat.shape[0]
        d = feat.shape[1]
        T = feat.shape[2]

        value = self.value_enc(feat).view(batch_size, self.inter_channels, -1)  # (batch, dim, frame#)
        value = value.permute(0, 2, 1)  # (batch, frame#, dim)

        query = self.query_enc(feat).view(batch_size, self.inter_channels, -1)  # (batch, dim, frame#)
        query = query.permute(0, 2, 1)  # (batch, frame#, dim)

        key = self.key_enc(feat).view(batch_size, self.inter_channels, -1)  # (batch, dim, frame#)

        f = torch.matmul(query, key) / math.sqrt(d)  # (batch, frame#, frame#)
        f = self.softmax(f)
        y = torch.matmul(f, value)  # (batch, frame#, dim)
        f_score = self.linear(y).squeeze(2)  # (batch, frame#)
        f_score = torch.sigmoid(f_score)

        # ------ Foreground Background Separation ------ #
        b = 1-f
        z = torch.matmul(b, value)  # (batch, frame#, dim)
        b_score = self.linear(z).squeeze(2)  # (batch, frame#)
        b_score = 1-torch.sigmoid(b_score)

        f_score = (f_score + b_score)/2.

        y = y.permute(0, 2, 1).contiguous()
        y = self.smooth(y)  # (batch, dim, frame#)
        out = feat + y
        ln = nn.LayerNorm((self.inter_channels, T)).cuda()
        out = ln(out)

        return out, f_score



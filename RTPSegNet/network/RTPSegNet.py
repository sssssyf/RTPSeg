import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from network.basic_block import Lovasz_loss
from network.spvcnn import get_model as SPVCNN

from network.basic_block import ResNetFCN, ResNetFCN_infra
from network.torchsparse_utils.base_model import LightningBaseModel


class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 确保embed_dim可以被num_heads整除
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        # 定义线性投影层（Query来自模态A，Key/Value来自模态B）
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_a, x_b, mask=None):
        """
        Args:
            x_a: 模态A的特征 [batch_size, seq_len_a, embed_dim]
            x_b: 模态B的特征 [batch_size, seq_len_b, embed_dim]
            mask: 可选，用于遮挡无效位置 [batch_size, seq_len_a, seq_len_b]
        Returns:
            融合后的特征 [batch_size, seq_len_a, embed_dim]
        """
        x_a = x_a.reshape(1,x_a.shape[0],x_b.shape[1])
        batch_size =x_a.size(0)

        # 1. 投影到Query, Key, Value空间
        Q = self.query(x_a)  # [batch_size, seq_len_a, embed_dim]
        K = self.key(x_b)  # [batch_size, seq_len_b, embed_dim]
        V = self.value(x_b)  # [batch_size, seq_len_b, embed_dim]

        # 2. 分割多头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # [batch_size, num_heads, seq_len_a, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # [batch_size, num_heads, seq_len_b, head_dim]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # [batch_size, num_heads, seq_len_b, head_dim]

        # 3. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))  # [batch_size, num_heads, seq_len_a, seq_len_b]

        # 4. 应用mask（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 5. Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len_a, seq_len_b]

        # 6. 加权求和
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, seq_len_a, head_dim]

        # 7. 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                    self.embed_dim)  # [batch_size, seq_len_a, embed_dim]

        # 8. 输出投影
        output = self.out(attn_output)  # [batch_size, seq_len_a, embed_dim]

        output = output.reshape(output.shape[1], output.shape[2])
        return output


class xModalKD(nn.Module):
    def __init__(self,config):
        super(xModalKD, self).__init__()
        self.hiden_size_list = config['model_params']['hiden_size_list']
        self.hiden_size = config['model_params']['hiden_size']
        self.scale_list = config['model_params']['scale_list']
        self.num_classes = config['model_params']['num_class']
        self.lambda_xm = config['train_params']['lambda_xm']
        self.lambda_seg2d = config['train_params']['lambda_seg2d']
        self.num_scales = 3
        self.infra_only = config['infra_only']
        self.image_only = config['image_only']
        self.image_and_infra = config['image_and_infra']


        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size_list[i], 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )

        self.multihead_fuse_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        self.leaners = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()

        if self.image_and_infra:
            hiden_size_scale = self.hiden_size * 3
            self.infra_leaners = nn.ModuleList()
            for i in range(self.num_scales):
                self.infra_leaners.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
        else:
            hiden_size_scale = self.hiden_size * 2

        for i in range(self.num_scales):
            self.leaners.append(nn.Sequential(nn.Linear(self.hiden_size_list[i], self.hiden_size)))
            self.fcs1.append(nn.Sequential(nn.Linear(hiden_size_scale, self.hiden_size)))
            self.fcs2.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))

        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        #self.ce_loss = nn.CrossEntropyLoss(weight=seg_labelweights, ignore_index=config['dataset_params']['ignore_label'])
        #self.lovasz_loss = Lovasz_loss(ignore=config['dataset_params']['ignore_label'])

        self.ce_loss = nn.CrossEntropyLoss()
        self.lovasz_loss = Lovasz_loss()
        self.img_pool = nn.AdaptiveAvgPool1d(1)

    @staticmethod
    def p2img_mapping(pts_fea, p2img_idx, batch_idx):
        img_feat = []
        for b in range(batch_idx.max()+1):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
        return torch.cat(img_feat, 0)

    @staticmethod
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
        labels = unq_lbxyz[:, 0][label_ind]
        return labels

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss

    def fusion_to_single_KD(self, data_dict, idx):
        batch_idx = data_dict['batch_idx']
        batch_idx_2 = data_dict['batch_idx_2']
        point2img_index = data_dict['point2img_index']
        #last_scale = self.scale_list[idx - 1] if idx > 0 else 1
        if self.image_only:
            img_feat = data_dict['img_scale{}'.format(self.scale_list[idx])]
        elif self.infra_only:
            infra_feat = data_dict['infra_scale{}'.format(self.scale_list[idx])]
        elif self.image_and_infra:
            img_feat = data_dict['img_scale{}'.format(self.scale_list[idx])]
            infra_feat = data_dict['infra_scale{}'.format(self.scale_list[idx])]

        pts_feat = data_dict['layer_{}'.format(idx+1)]
        invs = data_dict['inverse_map']
        #coors_inv = data_dict['scale_{}'.format(last_scale)]['coors_inv']
        raw_pts_feat = pts_feat[invs.F]
        # 3D prediction
        pts_pred_full = self.multihead_3d_classifier[idx](raw_pts_feat)

        # correspondence
        #pts_label_full = self.voxelize_labels(data_dict['labels'], data_dict['layer_{}'.format(idx)]['full_coors'])
        #pts_label_full = data_dict['sparse_label']
        pts_label_full = data_dict['targets_mapped'].F.reshape(-1)
        pts_feat = self.p2img_mapping(raw_pts_feat, point2img_index, batch_idx_2)
        pts_pred = self.p2img_mapping(pts_pred_full, point2img_index, batch_idx_2)

        # modality fusion
        feat_learner = F.relu(self.leaners[idx](pts_feat))

        if self.image_only:
            feat_cat = torch.cat([img_feat, feat_learner], 1)
        elif self.infra_only:
            feat_cat = torch.cat([infra_feat, feat_learner], 1)
        elif self.image_and_infra:
            infra_feat_learner = F.relu(self.infra_leaners[idx](infra_feat))
           # cross_attn = MultiHeadCrossModalAttention(embed_dim=64, num_heads=8).cuda()
            #cross_features_RGB_TIR = cross_attn(img_feat, infra_feat_learner)
            #cross_features_RGB_PC = cross_attn(img_feat, feat_learner)
            #cross_features_TIR_PC = cross_attn(infra_feat_learner, feat_learner)
            #img_feat = img_feat + cross_features_RGB_TIR + cross_features_RGB_PC
            #infra_feat_learner = infra_feat_learner + cross_features_RGB_TIR + cross_features_TIR_PC
            #feat_learner = feat_learner + cross_features_RGB_PC + cross_features_TIR_PC
            feat_cat = torch.cat([img_feat, infra_feat_learner, feat_learner], 1)


        feat_cat = self.fcs1[idx](feat_cat)
        feat_weight = torch.sigmoid(self.fcs2[idx](feat_cat))
        fuse_feat = F.relu(feat_cat * feat_weight)

        # fusion prediction
        fuse_pred = self.multihead_fuse_classifier[idx](fuse_feat)

        # Segmentation Loss
        seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full)
        seg_loss_2d = self.seg_loss(fuse_pred, data_dict['img_label'])
        loss = seg_loss_3d + seg_loss_2d * self.lambda_seg2d / self.num_scales

        # KL divergence
        xm_loss = F.kl_div(
            F.log_softmax(pts_pred, dim=1),
            F.softmax(fuse_pred.detach(), dim=1),
        )
        loss += xm_loss * self.lambda_xm / self.num_scales

        return loss, fuse_feat

    def forward(self, data_dict):
        loss = 0
        img_seg_feat = []

        for idx in range(self.num_scales):
            singlescale_loss, fuse_feat = self.fusion_to_single_KD(data_dict, idx)
            img_seg_feat.append(fuse_feat)
            loss += singlescale_loss

        img_seg_logits = self.classifier(torch.cat(img_seg_feat, 1))
        loss += self.seg_loss(img_seg_logits, data_dict['img_label'])
        data_dict['loss'] += loss

        return data_dict


class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.baseline_only = config.baseline_only
        self.num_classes = config.model_params.num_class
        self.hiden_size = config.model_params.hiden_size
        self.lambda_seg2d = config.train_params.lambda_seg2d
        self.lambda_xm = config.train_params.lambda_xm
        self.scale_list = config.model_params.scale_list
        self.num_scales = len(self.scale_list)
        self.infra_only = config.infra_only
        self.image_only = config.image_only
        self.image_and_infra = config.image_and_infra




        if config.model_3d=='SPVCNN':

            self.model_3d = SPVCNN(config)
        #if config.model_3d=='HR_SPVCNN':
           # self.model_3d = HR_SPVCNN(config)

        if not self.baseline_only:
            if self.image_only:
                self.model_img = ResNetFCN(
                    backbone=config.model_params.backbone_2d,
                    pretrained=config.model_params.pretrained2d,
                    pretrained_load_path=config.model_params.pretrained_load_path,
                    config=config
                )
                self.fusion = xModalKD(config)
            elif self.infra_only:
                self.model_infra = ResNetFCN_infra(
                    backbone=config.model_params.backbone_2d,
                    pretrained=config.model_params.pretrained2d,
                    pretrained_load_path=config.model_params.pretrained_load_path,
                    config=config
                )
                self.fusion = xModalKD(config)
            elif self.image_and_infra:
                self.model_img = ResNetFCN(
                    backbone=config.model_params.backbone_2d,
                    pretrained=config.model_params.pretrained2d,
                    pretrained_load_path=config.model_params.pretrained_load_path,
                    config=config
                )
                self.model_infra = ResNetFCN_infra(
                    backbone=config.model_params.backbone_2d,
                    pretrained=config.model_params.pretrained2d,
                    pretrained_load_path=config.model_params.pretrained_load_path,
                    config=config
                )
                self.fusion = xModalKD(config)
        else:
            print('Start vanilla training!')

        for param in self.model_img.parameters():
            param.requires_grad = False

        for param in self.model_infra.parameters():
            param.requires_grad = False

    def forward(self, data_dict):
        # 3D network
        data_dict = self.model_3d(data_dict)

        # training with 2D network
        if self.training and not self.baseline_only:
            if self.image_only:
                data_dict = self.model_img(data_dict)
            elif self.infra_only:
                data_dict = self.model_infra(data_dict)
            elif self.image_and_infra:
                data_dict = self.model_img(data_dict)
                data_dict = self.model_infra(data_dict)

            data_dict = self.fusion(data_dict)

        return data_dict
# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torchvision
import torch.nn as nn
import torch
#from dependency import *
#from utils import get_parameter_number
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import random
import ot
from .DEQ_fusion import DEQFusion

# dataset_name = "SPC"


class OT_Attn_assem(nn.Module):
    def __init__(self, impl='pot-uot-l2', ot_reg=0.1, ot_tau=0.5):
        super().__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau

    def normalize_feature(self, x):
        # x: [N, D]
        x = x - x.mean(dim=-1, keepdim=True)
        x = F.normalize(x, p=2, dim=-1, eps=1e-8)  # 单位范数
        return x

    def OT(self, src, tgt):
        """
        src: (N, D)  -> shared feature
        tgt: (M, D)  -> semantic feature
        """
        cost_map = torch.cdist(src, tgt) ** 2  # (N, M)

        if self.impl == "pot-uot-l2":
            a = torch.from_numpy(ot.unif(src.size(0)).astype('float64')).to(src.device)
            b = torch.from_numpy(ot.unif(tgt.size(0)).astype('float64')).to(tgt.device)
            M_cost = cost_map.detach() / cost_map.detach().max()
            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M=M_cost.double(),
                                                           reg=self.ot_reg, reg_m=self.ot_tau)
            flow = flow.float()
        elif self.impl == "pot-sinkhorn-l2":
            a = src.sum(dim=1) / src.sum()
            b = tgt.sum(dim=1) / tgt.sum()
            M_cost = cost_map.detach() / cost_map.detach().max()
            flow = ot.sinkhorn(a.detach(), b.detach(), M=M_cost, reg=self.ot_reg)
        else:
            raise NotImplementedError

        dist = torch.sum(cost_map * flow)
        return flow, dist

    def forward(self, shared_feature, semantic_feature):
        shared_feature = self.normalize_feature(shared_feature)
        semantic_feature = self.normalize_feature(semantic_feature)

        pi, dist = self.OT(shared_feature, semantic_feature) # pi: (N, M)

        # 用 semantic_feature 生成与 shared_feature 对齐的语义增强特征
        semantic_aligned = torch.mm(pi, semantic_feature)  # (N, D)

        # 可以选择融合方式，这里简单相加
        fused_feature = shared_feature + semantic_aligned

        return fused_feature, dist

class TextFeatureExtractor(nn.Module):
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT'):
        super(TextFeatureExtractor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.language_model = AutoModel.from_pretrained(model_name).cuda()
        self.fc = nn.Linear(768, 2048).cuda() 

    def forward(self, x):
        tokens = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokens = {key: val.cuda() for key, val in tokens.items()}
        outputs = self.language_model(**tokens)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  
        features = self.fc(cls_embedding)
        return features

sigmoid = nn.Sigmoid()

class FeedForward_MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.3):
        super(FeedForward_MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.ffn(x))  # 残差 + 归一化

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

# class MINE(nn.Module):
#     def __init__(self, input_dim, hidden_dim=256):
#         super(MINE, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, f1, f2):
#         # f1: 比如 shared feature，f2: 比如 private / 单模态 feature
#         # joint: 正样本 (joint distribution)
#         joint = torch.cat([f1, f2], dim=1)  # [B, 2 * input_dim]

#         # 通过 shuffle f2 构造负样本 (product of marginals)
#         idx = torch.randperm(f2.size(0), device=f2.device)
#         f2_shuffled = f2[idx]
#         marginal = torch.cat([f1, f2_shuffled], dim=1)  # [B, 2 * input_dim]

#         # T(x, y)
#         T_joint = self.net(joint)          # [B, 1]
#         T_marginal = self.net(marginal)    # [B, 1]

#         # Donsker–Varadhan 表达式的互信息下界估计：
#         # I_hat = E_joint[T] - log E_marg[exp(T)]
#         T_joint_mean = torch.mean(T_joint)
#         # 数值稳定，避免 log(0)
#         T_marginal_exp_mean = torch.mean(torch.exp(T_marginal))
#         mi_hat = T_joint_mean - torch.log(T_marginal_exp_mean + 1e-8)

#         # 返回 “正的互信息估计值”（越大越相关）
#         return mi_hat

class OrthogonalDisentangle(nn.Module):
    """
    简单正交约束解耦：
    希望 shared_feature 和 private_feature 在表征空间中“尽量正交”，
    通过惩罚它们的内积（或余弦相似度）的平方来实现。
    """
    def __init__(self):
        super(OrthogonalDisentangle, self).__init__()

    def forward(self, f_shared, f_private):
        """
        f_shared: [B, D]  比如 shared_feature
        f_private: [B, D] 比如 x_clic 或 x_derm
        """
        # 1) 去掉 batch 方向上的均值（零均值，有点类似去除偏置）
        f_shared_centered = f_shared - f_shared.mean(dim=0, keepdim=True)
        f_private_centered = f_private - f_private.mean(dim=0, keepdim=True)

        # 2) L2 归一化，防止范数过大导致梯度爆炸
        f_shared_norm = F.normalize(f_shared_centered, dim=1, eps=1e-8)
        f_private_norm = F.normalize(f_private_centered, dim=1, eps=1e-8)

        # 3) 逐样本的内积，相当于余弦相似度
        #    如果完全正交，inner 应该接近 0
        inner = torch.sum(f_shared_norm * f_private_norm, dim=1)  # [B]

        # 4) 惩罚平方，保证非负且对正负相关都惩罚
        loss = torch.mean(inner ** 2)

        # loss >= 0，越小表示越正交
        return loss

class Interaction_Estimator(nn.Module):
    def __init__(self, feat_len=6, feat_dim=64):
        super().__init__()
        self.geno_fc = FeedForward_MLP(feat_dim, int(feat_dim*0.5))
        self.path_fc = FeedForward_MLP(feat_dim, int(feat_dim*0.5))
        self.geno_atten = nn.Linear(feat_dim, 1)
        self.path_atten = nn.Linear(feat_dim, 1)
        
    def forward(self, gfeat, pfeat):        
        g_align = self.geno_fc(gfeat)   # [B, D]
        p_align = self.path_fc(pfeat)   # [B, D]

        inter = g_align * p_align       # [B, D]

        # 注意力从交互中提取，而不是从各自模态单独提取
        geno_att = torch.sigmoid(self.geno_atten(inter))  # [B, 1]
        path_att = torch.sigmoid(self.path_atten(inter))  # [B, 1]

        interaction = geno_att * g_align + path_att * p_align
        return interaction

class LinearLayer(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        x = self.clf(x)
        return x

class Confidence_Classification_SubNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.feature_extractor = FeedForward_MLP(input_dim, hidden_dim, dropout)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)  # shape: [B, input_dim]
        logits = self.classifier(x)    # shape: [B, num_classes]
        return logits

def apply_missing_fixed(a, v, p_missing=0.8, device='cuda', mode='alternate', return_mask=False):
    """
    固定顺序的模态缺失模拟（固定前 p% 的样本发生缺失）

    Args:
        a: 模态1张量 [batch, ...] (fundus)
        v: 模态2张量 [batch, ...] (OCT)
        p_missing: 缺失比例 (0~1)，表示前 p% 样本将被置为缺失
        device: 设备
        mode: 缺失模式
              'fundus'    -> 固定前 p% 样本的 fundus 缺失
              'oct'       -> 固定前 p% 样本的 OCT 缺失
              'alternate' -> 固定前 p% 样本交替缺失两种模态
        return_mask: 是否返回mask

    Returns:
        a_new, v_new, (mask_a, mask_v)  # 如果 return_mask=True
        a_new, v_new                    # 如果 return_mask=False
    """
    batch_size = a.size(0)
    num_missing = int(batch_size * p_missing)

    # 初始化mask: 1=缺失, 0=保留
    missing_mask = torch.zeros(batch_size, 2, device=device)

    # 根据模式设定缺失
    if mode == 'fundus':
        missing_mask[:num_missing, 0] = 1  # 前p%缺fundus
    elif mode == 'oct':
        missing_mask[:num_missing, 1] = 1  # 前p%缺OCT
    elif mode == 'alternate':
        for idx in range(num_missing):
            if idx % 2 == 0:
                missing_mask[idx, 0] = 1  # fundus缺失
            else:
                missing_mask[idx, 1] = 1  # OCT缺失
    else:
        raise ValueError("Invalid mode. Choose from ['fundus', 'oct', 'alternate'].")

    # print(missing_mask)
    # 应用缺失
    a_new, v_new = a.clone(), v.clone()
    for i in range(batch_size):
        if missing_mask[i, 0] == 1:
            a_new[i] = torch.zeros_like(a_new[i])
        if missing_mask[i, 1] == 1:
            v_new[i] = torch.zeros_like(v_new[i])

    # 返回
    if return_mask:
        mask_a, mask_v = 1 - missing_mask[:, 0], 1 - missing_mask[:, 1]
        return a_new, v_new, mask_a, mask_v
    else:
        return a_new, v_new

class FusionNet(nn.Module):

    def __init__(self, num_classes,p_missing,dataset_name):
        super(FusionNet, self).__init__()
        self.num_label = num_classes #if dataset change, num_class should be change (7-point: 5, MMC-AMD: 4 )
        self.dropout = nn.Dropout(0.3)
        
        self.model_clinic = torchvision.models.resnet50(pretrained=True)
        self.model_derm   = torchvision.models.resnet50(pretrained=True)
        # self.fundus_branch = torchvision.models.resnet50(pretrained=False)
        # self.oct_branch = torchvision.models.resnet50(pretrained=False)
        
        self.model_clinic = nn.Sequential(*list(self.model_clinic.children())[:-1])  # 去掉 classifier 层
        self.model_derm = nn.Sequential(*list(self.model_derm.children())[:-1])  # 去掉 classifier 层
        
        self.temperature = 0.07
        self.p_missing = p_missing

        
        if dataset_name == "SPC":
            print(dataset_name)
            self.vocab = {
                0: "Nevi are skin lesions characterized by a regular global shape and structural symmetry, with clearly defined borders that distinguish them from surrounding tissue. They typically exhibit consistent coloration in brown tones and display organized internal patterns—either globular or reticular—that are symmetrically distributed. The surface texture is smooth or slightly raised, and the lesion tends to remain stable in size and structure over time.",
                1: "Basal Cell Carcinoma exhibits structural asymmetry with raised or rolled lesion borders. It typically displays heterogeneous color patterns with uneven pigmentation. Internally, it may show central erosion or ulceration and often contains branching, arborizing blood vessels. The surface may appear disrupted—shiny, crusted, or scaly. Over time, the lesion tends to grow slowly but progressively, indicating local invasion. These features reflect stable conceptual cores across both clinical and dermoscopic views and serve as robust semantic anchors for multimodal representation learning and disentanglement.",
                2: "Melanoma presents with structural asymmetry and irregular, poorly defined borders. It exhibits a heterogeneous mix of colors—often including black, brown, red, blue, or white—with abrupt transitions and uneven pigment distribution. Internally, the lesion lacks organized patterns and shows chaotic arrangements such as blotches, dots, or streaks. The surface is frequently disrupted, appearing crusted, ulcerated, or nodular. Atypical blood vessels may be present, reflecting disordered angiogenesis. Critically, the lesion evolves over time—changing in size, color, and shape—signifying its malignant progression. These characteristics represent core semantic anchors that are stable across modalities and well-suited for guiding multimodal feature disentanglement.",
                3: "Miscellaneous skin lesions typically show regular, symmetric shapes and sharply defined boundaries. Their pigmentation is often uniform and limited to one or two colors such as brown, red, or purple, depending on the lesion type. Internally, these lesions may exhibit benign structural features like cysts, scar-like centers, or blood-filled spaces, with no sign of architectural chaos. The surface tends to be stable—appearing waxy, flat, or slightly raised without signs of erosion or ulceration. If vascular structures are visible, they are orderly and non-atypical. Importantly, these lesions are biologically stable over time, with minimal or no change in shape, size, or color. These consistent and non-malignant characteristics serve as robust semantic anchors across both clinical and dermoscopic modalities.",
                4: "Seborrheic keratosis consistently presents as a sharply defined, slightly elevated lesion with a “stuck-on” appearance. The lesion's borders are clearly demarcated and non-infiltrative. Color is typically uniform within regions, ranging from light to dark brown or black. Internally, the lesion contains benign structural patterns such as milia-like cysts, comedo-like pores, and brain-like ridges. Its surface is keratinized—appearing waxy, scaly, or greasy—and blood vessels are usually absent unless irritated. Seborrheic keratosis lesions grow slowly and remain biologically stable over time. These conceptual characteristics form modality-invariant semantic anchors, making them well-suited for guiding multimodal feature disentanglement and cross-domain alignment."
            }

        if dataset_name == "MMC":
            print(dataset_name)
            self.vocab = {
                0: "Wet AMD is a macular disease driven by a neovascular complex that leaks, producing fluid pockets and sometimes hemorrhage and persistent subretinal deposits. This leakage deforms the retinal architecture—creating central elevations, cystic spaces, retinal thickening, and loss of the foveal contour—and is accompanied by disruption of the supporting deep layer (RPE/Bruch). Over time active leakage can subside into fibrovascular scarring and patchy atrophy. The lesion’s hallmark is this combination of neovascular tissue, exudation/hemorrhage, structural deformation, and temporal instability — a compact set of modality-invariant semantic anchors suitable for cross-modal feature disentanglement.",
                1: "Dry AMD is a central (macular) degenerative process characterized by accumulation of sub-RPE deposits (drusen) and often subretinal granular deposits (reticular pseudodrusen), producing RPE irregularity and focal RPE elevation (drusenoid PED). These changes are accompanied by focal hyperreflective deposits and progressive loss of outer-retinal/photoreceptor integrity (ellipsoid zone disruption). Over time confluent RPE and photoreceptor loss forms sharply demarcated geographic atrophy with increased choroidal visibility. Crucially, dry AMD lacks persistent exudative fluid or neovascular leakage and follows a slow, degenerative course.",
                2: "PCV is a macular/peripapillary choroidal vasculopathy characterized by a branching vascular network that produces aneurysmal polyp-like nodules beneath the RPE. It causes episodic serous and hemorrhagic exudation, often associated with sharp, peaked or notched PEDs that contain nodular material. The lesion deposits dense subretinal material (SHRM) that distorts outer retinal layers and creates cystic spaces; it commonly occurs on a thickened (“pachy”) choroid. Recurrent activity leads to fibrovascular scarring, RPE damage and focal atrophy. These core elements — BVN + polypoidal nodules, exudation/hemorrhage, characteristic PED morphology, SHRM with outer-retinal distortion, pachychoroid background, and episodic activity — form robust, modality-independent semantic anchors for multimodal feature disentanglement.",
                3: "A normal retina has a preserved foveal depression and symmetric, continuous retinal layers with a flat, uniform RPE; there is no intra- or subretinal fluid, hemorrhage, exudate, focal mass, or pigmentary disruption, retinal and choroidal vessels appear of normal caliber and orderly branching, the vitreoretinal interface is smooth, and the overall structure is stable over time.",
            }

        if dataset_name == "Harvard":
            print(dataset_name)
            self.vocab = {
                0: "A normal retina has a preserved foveal depression and symmetric, continuous retinal layers with a flat, uniform RPE; there is no intra- or subretinal fluid, hemorrhage, exudate, focal mass, or pigmentary disruption, retinal and choroidal vessels appear of normal caliber and orderly branching, the vitreoretinal interface is smooth, and the overall structure is stable over time.",
                1: "Glaucoma is a chronic optic neuropathy centered on the optic nerve head and peripapillary region, characterized by progressive optic-nerve excavation (increased cup-to-disc ratio) and focal neuroretinal-rim thinning (often superior and inferior). It produces peripapillary RNFL loss (localized notches or diffuse thinning) and macular ganglion-cell inner-layer thinning, frequently accompanied by peripapillary atrophy and episodic disc hemorrhages. Deep optic-head remodeling (lamina cribrosa displacement) and clear structural asymmetry between eyes or quadrants are common. These structural changes progress slowly and map to predictable visual-field deficits, making the combined set of excavation + rim/RNFL/GCIPL loss + asymmetry + chronic progression robust, modality-independent semantic anchors for multimodal feature disentanglement.",
            }
        
        # self.fc_fusion = nn.Linear(128, self.num_label)
        
        
        #######----Label_attentive_encoder---######
        #self.label_encoder = LabelEncoder(2048, nclass=5) #if dataset change, num_class should be change (7-point: 5, MMC-AMD: 4 )
        self.class_embedding = TextFeatureExtractor()

        self.mine_loss_fn = OrthogonalDisentangle() #解耦损失函数

        #self.know_decompose = Knowledge_Decomposition(feat_dim=2048)        

        dimension = 2048
        self.shared_encoder = Interaction_Estimator(feat_dim=dimension)
        # self.Regression = RegressionSubNetwork(dimension)
        # self.Classification = ClassificationSubNetwork(dimension, self.num_label)

        #Confidence estimation
        self.Confidence_fundus = Confidence_Classification_SubNetwork(dimension, int(dimension*0.5), num_classes=1)
        self.Confidence_oct = Confidence_Classification_SubNetwork(dimension, int(dimension*0.5), num_classes=1)
        self.Confidence_shared = Confidence_Classification_SubNetwork(dimension, int(dimension*0.5), num_classes=1)

        #Classification
        self.Classification_fundus = Confidence_Classification_SubNetwork(dimension, int(dimension*0.5), num_classes=self.num_label)
        self.Classification_oct = Confidence_Classification_SubNetwork(dimension, int(dimension*0.5), num_classes=self.num_label)
        self.Classification_shared = Confidence_Classification_SubNetwork(dimension, int(dimension*0.5), num_classes=self.num_label)
        self.Classification_fusion = Confidence_Classification_SubNetwork(dimension, int(dimension*0.5), num_classes=self.num_label)

        #Feature with confidence
        # self.FeedForward_fundus = FeedForward_MLP(dimension, dimension*0.5, dropout=0.5)
        # self.FeedForward_oct = FeedForward_MLP(dimension, dimension*0.5, dropout=0.5)
        # self.FeedForward_shared = FeedForward_MLP(dimension, dimension*0.5, dropout=0.5)

        self.deq_fusion = DEQFusion(dimension, 3).cuda()

        #loss weight
        self.weight1 = nn.Parameter(torch.tensor(1.0))  
        self.weight2 = nn.Parameter(torch.tensor(1.0))  

        #OT
        self.ot_module = OT_Attn_assem().cuda()
            
    def confidence_loss(self, TCPLogit, TCPConfidence, label):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        pred = F.softmax(TCPLogit, dim=1)
        #print(label)
        #label = torch.tensor(label)
        p_target = torch.gather(input=pred, dim=1, index=label.cuda().unsqueeze(dim=1).type(torch.int64)).view(-1)
        c_loss = torch.mean(F.mse_loss(TCPConfidence.view(-1), p_target, reduction='none')+criterion(TCPLogit, label))
        return c_loss

    def forward(self, x, label_guide, missing):
        (x_clic,x_derm) = x
        # 对权重取exp()避免负值（可选softplus）
        w1 = torch.exp(self.weight1)
        w2 = torch.exp(self.weight2)

        x_clic = torch.flatten(self.model_clinic(x_clic),1)
        x_derm = torch.flatten(self.model_derm(x_derm),1)

        if missing == True:
            #print("Missing is true")
            # 随机模态缺失：0 表示正常，1 表示缺失
            x_clic,x_derm, mask1, mask2 = apply_missing_fixed(x_clic,x_derm, p_missing=self.p_missing, mode='fundus', return_mask=True)   

        decoded_text = [self.vocab[idx.item()] for idx in label_guide]
        label_emb = self.class_embedding(decoded_text)
        #y = F.one_hot(label_guide, num_classes = 5)*1.0   #if dataset change, num_class should be change
        #label_emb = self.label_encoder(y)

        shared_feature = self.shared_encoder(x_clic, x_derm)
        # print(shared_feature.shape)
        
        clic_ct_loss = self.mine_loss_fn(x_clic, shared_feature)
        derm_ct_loss = self.mine_loss_fn(x_derm, shared_feature)
        #contr_loss   = self.contrastive_loss(shared_feature, label_emb)
        shared_feature, ot_dist = self.ot_module(shared_feature, label_emb)
        distangle_loss = clic_ct_loss+derm_ct_loss+ot_dist

        TCPConfidence_clic = self.Confidence_fundus(x_clic)
        TCPConfidence_clic = torch.sigmoid(TCPConfidence_clic)
        TCPConfidence_derm = self.Confidence_oct(x_derm)
        TCPConfidence_derm = torch.sigmoid(TCPConfidence_derm)
        TCPConfidence_shared = self.Confidence_shared(shared_feature)
        TCPConfidence_shared = torch.sigmoid(TCPConfidence_shared)

        TCPLogit_clic = self.Classification_fundus(x_clic)
        TCPLogit_derm = self.Classification_oct(x_derm)
        TCPLogit_shared = self.Classification_shared(shared_feature)
        
        x_clic = x_clic * TCPConfidence_clic
        x_derm = x_derm * TCPConfidence_derm
        shared_feature = shared_feature * TCPConfidence_shared

        c_loss_clic = self.confidence_loss(TCPLogit_clic, TCPConfidence_clic, label_guide)
        c_loss_derm = self.confidence_loss(TCPLogit_derm, TCPConfidence_derm, label_guide)
        c_loss_shared = self.confidence_loss(TCPLogit_shared, TCPConfidence_shared, label_guide)
        c_loss = c_loss_clic+c_loss_derm+c_loss_shared
        
        #x_fusion = torch.cat([x_clic, x_derm, shared_feature], dim=1)
        x_fusion,_,_ = self.deq_fusion([x_clic, x_derm, shared_feature])
        # x_fusion = x_clic + x_derm + shared_feature

        #mid_feature = feature
        logit_fusion = self.Classification_fusion(x_fusion)
        # x_fusion = self.dropout(x_fusion)
        # logit_fusion = self.fc_fusion(x_fusion)
        
        return logit_fusion, x_fusion, distangle_loss, c_loss, TCPConfidence_clic, TCPConfidence_derm, TCPConfidence_shared



    def criterion(self, logit, truth):

        loss = nn.CrossEntropyLoss()(logit, truth)

        return loss

    def criterion1(self, logit, truth):

        loss = nn.L1Loss()(logit, truth)

        return loss


    def metric(self, logit, truth):
        # prob = F.sigmoid(logit)
        _, prediction = torch.max(logit.data, 1)

        acc = torch.sum(prediction == truth)
        return acc

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError      
             
             



# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch          
import torch.nn as nn 
import torch.nn.functional as F  
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


class KoLeoLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, target):
        """
        x: shape (B, D) where B is batch size, D is feature dimension.
        target: shape (B,) the ID labels of the batch.
        """
        # Normalize the features to the unit hypersphere
        x = F.normalize(x, p=2, dim=-1)
        
        # Calculate pairwise cosine distances (1 - cosine_similarity)
        sim_matrix = torch.mm(x, x.t())
        
        # MASKING POSITIVES: Find all images that belong to the SAME person
        # (This also inherently masks the diagonal self-similarity)
        is_pos = target.expand(x.size(0), x.size(0)).eq(target.expand(x.size(0), x.size(0)).t())
        
        # Fill positive pairs with -inf so they are ignored in the max() operation
        # Now the model will only push away the nearest neighbor of a DIFFERENT person
        sim_matrix.masked_fill_(is_pos, float('-inf'))
        
        # Find the maximum similarity (nearest neighbor of a DIFFERENT class)
        max_sim, _ = torch.max(sim_matrix, dim=1)
        
        # Euclidean distance squared between normalized vectors = 2 - 2 * cos_sim
        distances_sq = 2.0 - 2.0 * max_sim
        distances = torch.sqrt(distances_sq + self.eps)
        
        # KoLeo loss is the negative log of the nearest neighbor distances
        loss = -torch.mean(torch.log(distances))
        
        return loss




class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (B, D) tensor of normalized or unnormalized embeddings
        labels: (B,) tensor of identity labels
        """
        device = features.device
        batch_size = features.shape[0]

        # 1. Normalize features to unit sphere
        features = F.normalize(features, p=2, dim=1)

        # 2. Compute all-to-all cosine similarity matrix: (B, B)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 3. Create masks
        # Mask for all positives (B, B) where mask[i, j] = 1 if label[i] == label[j]
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-similarity (we don't want an image to be its own positive)
        logits_mask = torch.scatter(
            torch.ones_like(pos_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        pos_mask = pos_mask * logits_mask # True positives excluding self

        # 4. Numerical stability for log-sum-exp
        row_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - row_max.detach()

        # 5. Compute Log-Sum-Exp for the denominator (all negatives + positives)
        # We only sum over elements where logits_mask == 1 (excluding self)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 6. Compute the mean log-probability for the positives
        # Average over the number of positives per anchor
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-12)

        # 7. Final InfoNCE/SupCon loss
        loss = -mean_log_prob_pos.mean()

        return loss













def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048 # Adjust if your feat_dim differs
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    
    use_koleo = getattr(cfg.MODEL, 'USE_KOLEO_LOSS', False)
    if use_koleo:
        koleo_criterion = KoLeoLoss()
        koleo_weight = getattr(cfg.MODEL, 'KOLEO_LOSS_WEIGHT', 0.1)
        print(f"KoLeo regularizer is enabled with weight: {koleo_weight}")
    use_gram_anchor = getattr(cfg.MODEL, 'USE_GRAM_ANCHOR_LOSS', False)
    if use_gram_anchor:
        from .gram_anchor_loss import GramAnchorLoss
        gram_criterion = GramAnchorLoss(
            student_dim=getattr(cfg.MODEL, 'GRAM_ANCHOR_STUDENT_DIM', 384),
            teacher_dim=getattr(cfg.MODEL, 'GRAM_ANCHOR_TEACHER_DIM', 384),
        )
        gram_weight = getattr(cfg.MODEL, 'GRAM_ANCHOR_LOSS_WEIGHT', 1.0)
        print(f"Gram Anchor Loss is enabled with weight: {gram_weight}")
    # ---- NEW: Initialize SupCon or Triplet ----
    if 'supcon' in cfg.MODEL.METRIC_LOSS_TYPE:
        # You can also pull temperature from cfg if you add it to defaults.py
        metric_criterion = SupervisedContrastiveLoss(temperature=0.05) 
        print("Using Supervised Contrastive (InfoNCE) loss for training")
    elif 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            metric_criterion = TripletLoss()
            print("using soft triplet loss for training")
        else:
            metric_criterion = TripletLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print(f"expected METRIC_LOSS_TYPE should be supcon or triplet but got {cfg.MODEL.METRIC_LOSS_TYPE}")

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target, target_cam, student_tokens=None, teacher_tokens=None):
            base_loss = F.cross_entropy(score, target)

            result = {}
            if use_koleo:
                koleo_reg = koleo_criterion(feat[0],target) if isinstance(feat, list) else koleo_criterion(feat,target)
                base_loss = base_loss + (koleo_weight * koleo_reg)
                result["koleo_loss"] = koleo_reg
            if use_gram_anchor and student_tokens is not None and teacher_tokens is not None:
                gram_loss = gram_criterion(student_tokens, teacher_tokens)
                base_loss = base_loss + (gram_weight * gram_loss)
                result["gram_loss"] = gram_loss
            if result:
                result["total_loss"] = base_loss
                return result
            return base_loss
           
    elif 'triplet' in sampler: # Keep the sampler name 'softmax_triplet' as it dictates the PxK batch sampling
        def loss_func(score, feat, target, target_cam, student_tokens=None, teacher_tokens=None):
            # 1. ID LOSS (Softmax / CrossEntropy)
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                if isinstance(score, list):
                    ID_LOSS = sum([xent(scor, target) for scor in score[1:]]) / len(score[1:])
                    ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                else:
                    ID_LOSS = xent(score, target)
            else:
                if isinstance(score, list):
                    ID_LOSS = sum([F.cross_entropy(scor, target) for scor in score[1:]]) / len(score[1:])
                    ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

            # 2. METRIC LOSS (SupCon or Triplet)
            if cfg.MODEL.METRIC_LOSS_TYPE == 'supcon':
                if isinstance(feat, list):
                    METRIC_LOSS = sum([metric_criterion(feats, target) for feats in feat[1:]]) / len(feat[1:])
                    METRIC_LOSS = 0.5 * METRIC_LOSS + 0.5 * metric_criterion(feat[0], target)
                else:
                    METRIC_LOSS = metric_criterion(feat, target)
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if isinstance(feat, list):
                    METRIC_LOSS = sum([metric_criterion(feats, target)[0] for feats in feat[1:]]) / len(feat[1:])
                    METRIC_LOSS = 0.5 * METRIC_LOSS + 0.5 * metric_criterion(feat[0], target)[0]
                else:
                    METRIC_LOSS = metric_criterion(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]

            # Combine losses
            base_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * METRIC_LOSS
            
            result = {}
            # 3. Optional KoLeo Regularization
            if use_koleo:
                koleo_reg = koleo_criterion(feat[0],target) if isinstance(feat, list) else koleo_criterion(feat,target)
                if koleo_reg is None:
                    raise ValueError("KoLeo loss is None")
                base_loss = base_loss + (koleo_weight * koleo_reg)
                result["koleo_loss"] = koleo_reg
                
            # 4. Optional Gram Anchor Loss
            if use_gram_anchor and student_tokens is not None and teacher_tokens is not None:
                gram_loss = gram_criterion(student_tokens, teacher_tokens)
                base_loss = base_loss + (gram_weight * gram_loss)
                result["gram_loss"] = gram_loss

            if result:
                result["total_loss"] = base_loss
                return result
                
            return base_loss
               
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
              
    return loss_func, center_criterion
# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch         # <--- ADDED START: Imported for KoLeoLoss
import torch.nn as nn 
import torch.nn.functional as F # <--- ADDED END
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss

# <--- ADDED START: Inserted the KoLeoLoss class definition
class KoLeoLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        x: shape (B, D) where B is batch size, D is feature dimension.
        """
        # Normalize the features to the unit hypersphere
        x = F.normalize(x, p=2, dim=-1)
        
        # Calculate pairwise cosine distances (1 - cosine_similarity)
        sim_matrix = torch.mm(x, x.t())
        
        # Fill diagonal with -inf so it's ignored in the max() operation
        sim_matrix.fill_diagonal_(float('-inf'))
        
        # Find the maximum similarity (nearest neighbor) for each feature
        max_sim, _ = torch.max(sim_matrix, dim=1)
        
        # Euclidean distance squared between normalized vectors = 2 - 2 * cos_sim
        distances_sq = 2.0 - 2.0 * max_sim
        distances = torch.sqrt(distances_sq + self.eps)
        
        # KoLeo loss is the negative log of the nearest neighbor distances
        loss = -torch.mean(torch.log(distances))
        
        return loss
# <--- ADDED END


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    
    # <--- ADDED START: Check flag and instantiate KoLeoLoss conditionally
    use_koleo = getattr(cfg.MODEL, 'USE_KOLEO_LOSS', False) # Safely checks for the boolean flag
    if use_koleo:
        koleo_criterion = KoLeoLoss()
        koleo_weight = getattr(cfg.MODEL, 'KOLEO_LOSS_WEIGHT', 0.1) # You can tune this hyperparameter
        print(f"KoLeo regularizer is enabled with weight: {koleo_weight}")
    # <--- ADDED END

    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler in ['softmax', 'id']:
        def loss_func(score, feat, target,target_cam):
            base_loss = F.cross_entropy(score, target)
            
            # <--- ADDED START: Conditionally add KoLeo loss
            if use_koleo:
                koleo_reg = koleo_criterion(feat[0]) if isinstance(feat, list) else koleo_criterion(feat)
                return base_loss + (koleo_weight * koleo_reg)
            return base_loss
            # <--- ADDED END

    #  elif cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'id_triplet', 'img_triplet']:
    elif 'triplet' in sampler:
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]

                    # <--- ADDED START: Compute base loss, then conditionally add KoLeo
                    base_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if use_koleo:
                        koleo_reg = koleo_criterion(feat[0]) if isinstance(feat, list) else koleo_criterion(feat)
                        return base_loss + (koleo_weight * koleo_reg)
                    return base_loss
                    # <--- ADDED END
                    
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target, normalize_feature=cfg.SOLVER.TRP_L2)[0]

                    # <--- ADDED START: Compute base loss, then conditionally add KoLeo
                    base_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if use_koleo:
                        koleo_reg = koleo_criterion(feat[0]) if isinstance(feat, list) else koleo_criterion(feat)
                        return base_loss + (koleo_weight * koleo_reg)
                    return base_loss
                    # <--- ADDED END
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
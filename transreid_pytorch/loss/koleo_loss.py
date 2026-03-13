import torch
import torch.nn as nn
import torch.nn.functional as F

class KoLeoLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        x: shape (B, D) where B is batch size, D is feature dimension.
           In your case, this would be the `global_feat` [CLS] token.
        """
        # Normalize the features to the unit hypersphere
        x = F.normalize(x, p=2, dim=-1)
        
        # Calculate pairwise cosine distances (1 - cosine_similarity)
        # Using matrix multiplication for cosine similarity
        sim_matrix = torch.mm(x, x.t())
        
        # We want the nearest neighbor, which means the HIGHEST similarity.
        # We must mask out the diagonal (a token's similarity to itself, which is 1.0)
        # Fill diagonal with -inf so it's ignored in the max() operation
        sim_matrix.fill_diagonal_(float('-inf'))
        
        # Find the maximum similarity (nearest neighbor) for each feature in the batch
        max_sim, _ = torch.max(sim_matrix, dim=1)
        
        # Convert similarity back to distance (L2 distance on unit sphere)
        # Euclidean distance squared between normalized vectors = 2 - 2 * cos_sim
        distances_sq = 2.0 - 2.0 * max_sim
        distances = torch.sqrt(distances_sq + self.eps)
        
        # KoLeo loss is the negative log of the nearest neighbor distances
        # Minimizing this loss maximizes the distance to the nearest neighbor
        loss = -torch.mean(torch.log(distances))
        
        return loss
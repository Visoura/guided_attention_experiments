import torch
import torch.nn as nn
import torch.nn.functional as F


class GramAnchorLoss(nn.Module):
    """
    Gram Anchor Loss: compares structural correlations between student (TransReID)
    and teacher (DINOv3) token sequences via Gram matrices.

    Inputs are full token sequences (CLS + patches). If dimensions differ,
    a learned linear projection aligns the student to the teacher dim.
    """

    def __init__(self, student_dim=384, teacher_dim=384):
        super().__init__()
        self.proj = None
        if student_dim != teacher_dim:
            self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, student_tokens, teacher_tokens):
        """
        Args:
            student_tokens: (B, N, D_s) — CLS + patch tokens from TransReID
            teacher_tokens: (B, N, D_t) — CLS + patch tokens from DINOv3
        Returns:
            Scalar MSE loss between Gram matrices.
        """
        # Project student dim to match teacher if needed
        if self.proj is not None:
            student_tokens = self.proj(student_tokens)

        # L2-normalize along feature dim
        s = F.normalize(student_tokens, p=2, dim=-1)
        t = F.normalize(teacher_tokens, p=2, dim=-1)

        # Gram matrices: (B, N, N)
        gram_s = torch.bmm(s, s.transpose(1, 2))
        gram_t = torch.bmm(t, t.transpose(1, 2))

        return F.mse_loss(gram_s, gram_t)

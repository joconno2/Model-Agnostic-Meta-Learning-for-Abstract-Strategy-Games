import torch
import torch.nn.functional as F


def combined_loss(logits, v_pred, y_action, z, lambda_value: float = 1.0, legal_mask=None):
    """
    Policy: CrossEntropy (optionally masked to legal moves only)
    Value:  MSE

    legal_mask: optional [B, n_actions] bool/uint8 tensor. If provided, illegal
    moves are set to -inf before cross-entropy, so the policy only competes
    over legal moves. This is critical: with n_actions=20480 and ~30 legal
    moves/position, unmasked CE wastes nearly all gradient signal learning
    which outputs are illegal.
    """
    if legal_mask is not None:
        mask = legal_mask.bool()
        logits = logits.masked_fill(~mask, float("-inf"))

    lp = F.cross_entropy(logits, y_action)
    lv = F.mse_loss(v_pred, z)
    return lp + lambda_value * lv, lp.detach(), lv.detach()

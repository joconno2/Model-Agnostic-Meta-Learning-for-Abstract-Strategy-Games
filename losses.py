# losses.py
import torch
import torch.nn.functional as F

#Calculates the loss for both the policy and value heads
def combined_loss(logits, v_pred, y_action, z, lambda_value: float = 1.0):
    """
    - Policy: CrossEntropy
    - Value:  MSE
    """
    lp = F.cross_entropy(logits, y_action) #--> Policy
    lv = F.mse_loss(v_pred, z) #--> Value
    return lp + lambda_value * lv, lp.detach(), lv.detach()

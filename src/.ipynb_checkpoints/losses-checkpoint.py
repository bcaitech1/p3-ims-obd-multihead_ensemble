import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
"""
    https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
        - DiceLoss
        - DiceBCELoss
        - IoULoss
        - FocalLoss
        - TverskyLoss
        - FocalTverskyLoss
        - ComboLoss
"""


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        num_classes = inputs.size(1)
        true_1_hot = torch.eye(num_classes)[targets]

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(inputs, dim=1)

        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice = ((2. * intersection + smooth) / (cardinality + smooth)).mean()

        return 1 - dice


class DiceCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCELoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        num_classes = inputs.size(1)
        true_1_hot = torch.eye(num_classes)[targets]

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(inputs, dim=1)

        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = ((2. * intersection + smooth) / (cardinality + smooth)).mean()
        dice_loss = (1 - dice_loss)

        ce = F.cross_entropy(inputs, targets, reduction='mean', weight=self.weight)
        dice_bce = ce * 0.7 + dice_loss * 0.3
        return dice_bce

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.eps = 1e-8

    def forward(self, inputs , targets):
        """Computes the Jaccard loss, a.k.a the IoU loss.
        Notes: [Batch size,Num classes,Height,Width]
        Args:
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            inputs: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model. (prediction)
            eps: added to the denominator for numerical stability.
        Returns:
            iou: the average class intersection over union value
                for multi-class image segmentation
        """
        num_classes = inputs.shape[1]
        # Single class segmentation?
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[targets.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(inputs)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        # Multi-class segmentation
        else:
            # Convert target to one-hot encoding
            # true_1_hot = torch.eye(num_classes)[torch.squeeze(targets,1)]
            true_1_hot = torch.eye(num_classes)[targets.squeeze(1)]
            # Permute [B,H,W,C] to [B,C,H,W]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            # Take softmax along class dimension; all class probs add to 1 (per pixel)
            probas = F.softmax(inputs, dim=1)
        true_1_hot = true_1_hot.type(inputs.type())
        # Sum probabilities by class and across batch images
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims) # [class0,class1,class2,...]
        cardinality = torch.sum(probas + true_1_hot, dims)  # [class0,class1,class2,...]
        union = cardinality - intersection
        iou = (intersection / (union + self.eps)).mean()   # find mean of class IoU values
        return 1 - iou



class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=.25, eps=1e-7, weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.weight = weights

    def forward(self, inp, tar):
        logp = F.log_softmax(inp, dim=1)
        ce_loss = F.nll_loss(logp, tar, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)

        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky


class ComboLoss(nn.Module):
    def __init__(self, ce_ratio=0.5, alpha=0.5, beta=0.5, weight=None, size_average=True):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        inputs = torch.clamp(inputs, e, 1.0 - e)
        out = - (self.alpha * ((targets * torch.log(inputs)) + ((1 - self.alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)

        return combo
    
class OnlineHardExampleMiningLoss(nn.Module):
    def __init__(self, top_k=0.7, weight=None, size_average=None,
                    ignore_index=0, reduce=None, reduction='none'):
        super(OnlineHardExampleMiningLoss, self).__init__()
        
        self.ignore_index = ignore_index
        self.top_k = top_k
        self.loss = nn.NLLLoss(weight=weight, 
                    ignore_index=ignore_index, reduction='none')

    def forward(self, input, target):
        loss = self.loss(F.log_softmax(input, dim=1), target)
        
        if self.top_k == 1:
            return torch.mean(loss)
        else:
            valid_loss, idxs = torch.topk(loss, int(self.top_k * loss.size()[0]))    
            return torch.mean(valid_loss)

class TripleLoss(nn.Module):
    def __init__(self):
        super(TripleLoss , self).__init__()
        self.f1 = FocalLoss()
        self.iou = IoULoss()
        self.msssim = MS_SSIM(data_range=1, size_average=True, channel = 12)
    
    def forward(self, inputs, targets):
        target = targets.unsqueeze(1)
        target = target.expand(-1,12,-1,-1).float()
        return self.f1(inputs, targets) + self.iou(inputs,targets) + (1 - self.msssim(inputs,target))

class DoubleLoss(nn.Module):
    def __init__(self):
        super(DoubleLoss , self).__init__()
        self.f1 = FocalLoss()
        self.iou = IoULoss()
    
    def forward(self, inputs, targets):
        return self.f1(inputs, targets) + self.iou(inputs,targets)
    
    
_criterion_entrypoints = {
    'CELoss': nn.CrossEntropyLoss,
    'DiceLoss': DiceLoss,
    'DiceCELoss': DiceCELoss,
    'IoULoss': IoULoss,
    'FocalLoss' : FocalLoss,
    'TverskyLoss' : TverskyLoss,
    'FocalTverskyLoss' : FocalTverskyLoss,
    'ComboLoss' : ComboLoss,
    'TripleLoss' : TripleLoss,
    'DoubleLoss' : DoubleLoss,
    'OnlineHardExampleMiningLoss' : OnlineHardExampleMiningLoss
}

def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints

def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion



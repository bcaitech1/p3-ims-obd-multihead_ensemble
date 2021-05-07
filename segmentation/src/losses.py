import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        num_classes = inputs.size(1)
        true_1_hot = torch.eye(num_classes)[targets]

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(inputs, dim=1)

        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection

        IoU = ((intersection + smooth) / (union + smooth)).mean()
        return 1 - IoU


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


class FocalCELoss(nn.Module):
    def __init__(self, gamma=2, alpha=.25, eps=1e-7, weight=None):
        super(FocalCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.weight = weight

    def forward(self, inp, tar):
        logp = F.log_softmax(inp, dim=1)
        ce_loss = F.nll_loss(logp, tar, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)

        fc_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        fc_loss = fc_loss.mean()

        ce = F.cross_entropy(inp, tar, reduction='mean', weight=self.weight)
        fc_ce = ce * 0.8 + fc_loss * 0.2
        return fc_ce


class AmazingCELoss(nn.Module):
    def __init__(self, gamma=2, alpha=.25, eps=1e-7, weight=None):
        super(AmazingCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.weight = weight

    def forward(self, inp, tar, smooth=1.0):
        num_classes = inp.size(1)
        true_1_hot = torch.eye(num_classes)[tar]

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(inp, dim=1)

        true_1_hot = true_1_hot.type(inp.type())
        dims = (0,) + tuple(range(2, tar.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = ((2. * intersection + smooth) / (cardinality + smooth)).mean()
        dice_loss = (1 - dice_loss)

        logp = F.log_softmax(inp, dim=1)
        ce_loss = F.nll_loss(logp, tar, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)

        fc_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        fc_loss = fc_loss.mean()

        ce_loss = F.cross_entropy(inp, tar, reduction='mean', weight=self.weight)
        amazing_loss = ce_loss * 0.5 + dice_loss * 0.3 + fc_loss * 0.2
        return amazing_loss


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



from torchgeometry.losses import SSIM
class OhMyLoss(nn.Module):
    def __init__(self,alpha = 0.75, beta = 0.25 , gamma = 0.25, weight=None):
        super(OhMyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dice = DiceLoss()
        self.weight = weight
        self.ssim = SSIM(window_size=11, reduction='mean')

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs , targets)
        ce_loss = F.cross_entropy(inputs, targets, reduction='mean', weight=self.weight)

        ss_input = F.softmax(inputs, dim=1)
        ss_input = torch.argmax(ss_input, dim=1, keepdim=True).float()
        h , w = targets.shape[1] , targets.shape[2]
        targets = targets.view(-1 , 1, h, w).float()
        ssim_loss = self.ssim(ss_input, targets)

        triple_loss = self.alpha * ce_loss + self.beta * dice_loss + self.gamma * ssim_loss
        return triple_loss
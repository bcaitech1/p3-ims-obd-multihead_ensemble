import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomLoss, self).__init__()
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
        dice_loss = ((2. * intersection + smooth) /
                     (cardinality + smooth)).mean()
        dice_loss = (1 - dice_loss)

        ce = F.cross_entropy(inputs, targets, reduction='mean', weight=self.weight)
        focal_loss=FocalLoss(inputs,targets)
        dice_bce = ce* 0.3 + dice_loss * 0.7
        return dice_bce

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection+smooth)/(inputs.sum()+targets.sum()+smooth)

        return 1-dice
        
class DiceCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCELoss, self).__init__()

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
        dice_loss = (2. * intersection / (cardinality + 1e-7)).mean()
        dice_loss = (1 - dice_loss)

        ce = F.cross_entropy(inputs, targets, reduction='mean')
        dice_bce = ce * 0.75 + dice_loss * 0.25
        return dice_bce

class IOULOss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IOULOss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs*targets).sum()
        total = (inputs+targets).sum()
        union = total-intersection

        IOU = (intersection+smooth)/(union+smooth)
        return 1-IOU


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BEC_EXP = torch.exp(-BCE)
        focal_loss = self.alpha*(1-BEC_EXP)**self.gamma*BCE
        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs*targets).sum()
        FP = ((1-targtes)*inputs).sum()
        FN = (targets*(1-inputs)).sum()

        Tversky = (TP+smooth)/(TP+self.alpha*FP + self.beta*FN + smooth)
        return 1-Tversky


class ComboLoss(nn.Module):
    def __init__(self, ce_ratio=0.5, alpha=0.5, beta=0.5, weight=None, size_average=True):
        suepr(ComboLoss, self).__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs*targets).sum()
        dice = (2.*intersection+smooth)/(inputs.sum()+targets.sum()+smooth)
        inputs = torch.clamp(inputs, e, 1.0-e)
        out = - (self.alpha * ((targets * torch.log(inputs)) +
                               ((1 - self.alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)
        return combo

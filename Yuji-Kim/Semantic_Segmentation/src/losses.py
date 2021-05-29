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
pos_weight = [
    0.3040,
    0.9994,
    0.9778,
    0.9097,
    0.9930,
    0.9911,
    0.9924,
    0.9713,
    0.9851,
    0.8821,
    0.9995,
    0.9947
]    

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pos_weight = torch.tensor(pos_weight).float().to(device)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """Computes the Dice loss
        Notes: [Batch size,Num classes,Height,Width]
        Args:
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            inputs: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model. (prediction)
            eps: added to the denominator for numerical stability.
        Returns:
            dice coefficient: the average 2 * class intersection over cardinality value
            for multi-class image segmentation
        """
        num_classes = inputs.size(1)                          # channel 수 
        true_1_hot = torch.eye(num_classes)[targets]          # target을 one_hot vector로 만들어준다. 

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()   # [B,H,W,C] -> [B,C,H,W]
        probas = F.softmax(inputs, dim=1)                     # preds를 softmax 취해주어 0~1사이 값으로 변환

        true_1_hot = true_1_hot.type(inputs.type())           # input과 type 맞춰주기
        dims = (0,) + tuple(range(2, targets.ndimension()))   # ?
        intersection = torch.sum(probas * true_1_hot, dims)   # TP
        cardinality = torch.sum(probas + true_1_hot, dims)    # TP + FP + FN + TN
        dice = ((2. * intersection + smooth) / (cardinality + smooth)).mean() 

        return 1 - dice


# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

#         return 1 - dice


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.5, gamma=2, weight=None, size_average=True):
#         super(FocalLoss, self).__init__()

#         self.alpha= alpha
#         self.gamma = gamma

#     def forward(self, inputs, targets, smooth=1):
#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         # first compute binary cross-entropy
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         BCE_EXP = torch.exp(-BCE)
#         focal_loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE

#         return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=.25, eps=1e-7, weights=None):   #weight = pos_weight 주면 됨 
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.weight = weights

    def forward(self, inputs, targets):
        """Computes the FocalLoss
        Notes: [Batch size,Num classes,Height,Width]
        Args:
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            inputs: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model. (prediction)
        Returns:
            focal loss : -alpha *(1-p)*gamma * log(p)
        """
        logp = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(logp, targets, weight=self.weight, reduction='none')
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

class DiceCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCELoss, self).__init__()

    def forward(self, inputs, targets, gamma=0.8, smooth=1):
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
        dice_bce = ce * gamma + dice_loss * (1 - gamma)
        return dice_bce

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha = 0.75):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, inputs, targets, eps = 1e-8):
        """Computes the DiceFocalLoss
        Notes: [Batch size,Num classes,Height,Width]
        Args:
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            inputs: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model. (prediction)
            eps: added to the denominator for numerical stability.
        Returns:
            DiceFocalLoss = alpha * Diceloss + (1 - alpha) * Focalloss
        """
        dice_loss = self.dice(inputs , targets)
        focal_loss = self.focal(inputs , targets)
        dice_focal_loss = focal_loss * self.alpha + dice_loss * (1-self.alpha)
        return dice_focal_loss
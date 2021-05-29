import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=12, smoothing=0.1, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



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
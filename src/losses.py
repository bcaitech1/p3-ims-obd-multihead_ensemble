import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchgeometry.losses import SSIM
import numpy as np

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


"""
    https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
        - DiceLoss
        - DiceBCELoss
        - IoULoss
        - FocalLoss
        - TverskyLoss
        - FocalTverskyLoss
        - ComboLoss
   # https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/4247f6a9a974cda65b8f16ceccf90937a726c978/LovaszSoftmax/lovasz_loss.py#L51
   # https://github.com/JunMa11/SegLoss/blob/a5732a9e0a552ac5699e137152e6f879480756c9/losses_pytorch/boundary_loss.py#L9
   # https://arxiv.org/pdf/2006.14822.pdf
"""

################################################################################
#                              pixel level                                     #
################################################################################
"""
  - Cross Entropy -> nn.CrossEntropy
  - FocalLoss 
"""
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=.25, weights=None):   
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
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


################################################################################
#                             map level                                     #
################################################################################
"""
  - DiceLoss
  - FocalLoss 
  - TverskyLoss
  - FocalTverskyLoss
  - SSLLoss
  - LogCoshDiceLoss
  TP | Targets : True , Preds : True
  TN | Targets : False , Preds : False
  FP | Targets : False , Preds : True
  FN | Targets : True , Preds : False
"""

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


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs , targets, eps = 1e-8):
        """Computes the IoULoss.
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
        num_classes = inputs.size(1)
        true_1_hot = torch.eye(num_classes)[targets]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(inputs, dim=1)

        true_1_hot = true_1_hot.type(inputs.type())

        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims) # [class0,class1,class2,...]
        cardinality = torch.sum(probas + true_1_hot, dims)  # [class0,class1,class2,...]
        union = cardinality - intersection
        iou = (intersection / (union + eps)).mean()   # find mean of class IoU values
        return 1 - iou


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):    # 0.5면 dice와 동일 
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        """Computes the TverskyLoss
        Notes: [Batch size,Num classes,Height,Width]
        Args:
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            inputs: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model. (prediction)
            smooth: added to the denominator for numerical stability.
        Returns:
            Tversky: TP / (TP + a * FP + b * FN)
        """
        num_classes = inputs.size(1)                          
        true_1_hot = torch.eye(num_classes)[targets]          
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()  
        probas = F.softmax(inputs, dim=1)                     
        true_1_hot = true_1_hot.type(inputs.type())           
        dims = (0,) + tuple(range(2, targets.ndimension()))   

        TP = torch.sum(probas * true_1_hot , dims)
        FP = torch.sum((1 - true_1_hot) * probas, dims)
        FN = torch.sum(true_1_hot * (1 - probas) , dims)

        Tversky = ((TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)).mean()

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha = .25 , beta=0.6, gamma=2):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


    def forward(self, inputs, targets, smooth=1):
        """Computes the FocalTverskyLoss
        Notes: [Batch size,Num classes,Height,Width]
        Args:
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            inputs: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model. (prediction)
            smooth: added to the denominator for numerical stability.
        Returns:
            FocalTversky: (1 -Tversky)**gamma
        """
        num_classes = inputs.size(1)                          
        true_1_hot = torch.eye(num_classes)[targets]          

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()   
        probas = F.softmax(inputs, dim=1)                     

        true_1_hot = true_1_hot.type(inputs.type())           
        dims = (0,) + tuple(range(2, targets.ndimension()))   

        TP = torch.sum(probas * true_1_hot , dims)
        FP = torch.sum((1 - true_1_hot) * probas, dims)
        FN = torch.sum(true_1_hot * (1 - inputs) , dims)

        Tversky = (TP + smooth) / (TP + self.beta * FP +(1-self.beta) * FN + smooth).mean()
        FocalTversky = (self.alpha * (1 - Tversky) ** self.gamma).mean()

        return FocalTversky


class SSLLoss(nn.Module):
    def __init__(self, alpha = 0.5, beta = 0.5):
        super(SSLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self,inputs, targets, smooth=1):
        """Computes the SSLLoss
        Notes: [Batch size,Num classes,Height,Width]
        Args:
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            inputs: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model. (prediction)
            smooth: added to the denominator for numerical stability.
        Returns:
            sensitivity : TP / (TP + FN)
            specificity = TN / (TN + FP)
        """
        num_classes = inputs.size(1)                          
        true_1_hot = torch.eye(num_classes)[targets]         

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()   
        probas = F.softmax(inputs, dim=1)                     

        true_1_hot = true_1_hot.type(inputs.type())           
        dims = (0,) + tuple(range(2, targets.ndimension()))  

        TP = torch.sum(probas * true_1_hot , dims)
        TN = torch.sum((1-probas) * (1-true_1_hot), dims)
        FP = torch.sum((1 - true_1_hot) * probas, dims)
        FN = torch.sum(true_1_hot * (1 - inputs) , dims)     

        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        
        loss = self.alpha * sensitivity + self.beta * specificity

        return loss.mean()


class LogCoshDiceLoss(nn.Module):
    def __init__(self):
        super(LogCoshDiceLoss, self).__init__()

    def forward(self, inputs, targets, eps=1e-8, smooth = 1):
        """Computes the LogCoshDiceLoss
        Notes: [Batch size,Num classes,Height,Width]
        Args:
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            inputs: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model. (prediction)
            eps: added to the denominator for numerical stability.
        Returns:
            log_cosh_dice: log(cosh(DiceLoss))
        """
        num_classes = inputs.size(1)                          
        true_1_hot = torch.eye(num_classes)[targets]          

        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()   
        probas = F.softmax(inputs, dim=1)                     

        true_1_hot = true_1_hot.type(inputs.type())           
        dims = (0,) + tuple(range(2, targets.ndimension()))   
        intersection = torch.sum(probas * true_1_hot, dims)   
        cardinality = torch.sum(probas + true_1_hot, dims)    
        dice = ((2. * intersection + smooth) / (cardinality + smooth)).mean() 
      
        log_cosh_dice_loss = torch.log(torch.cosh(1 - dice + eps))

        return log_cosh_dice_loss.mean()

################################################################################
#                           compound loss                                     #
################################################################################

class DiceCELoss(nn.Module):
    def __init__(self, weight=pos_weight, alpha = 0.75):
        super(DiceCELoss, self).__init__()
        self.alpha = alpha
        self.dice = DiceLoss()
        self.weight = weight

    def forward(self, inputs, targets, eps = 1e-8):
        """Computes the DiceCELoss
        Notes: [Batch size,Num classes,Height,Width]
        Args:
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            inputs: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model. (prediction)
            eps: added to the denominator for numerical stability.
        Returns:
            DiceCELoss = alpha * Diceloss + (1 - alpha) * CEloss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='mean',weight = self.weight)
        dice_loss = self.dice(inputs, targets)
        dice_ce = ce_loss * self.alpha + dice_loss * (1-self.alpha)
        return dice_ce


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


class TripleLoss(nn.Module):
    def __init__(self,alpha = 0.75, beta = 0.25 , gamma = 0.25):
        super(TripleLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma       
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.ssim = SSIM(window_size = 11  ,reduction = 'mean')

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs , targets)
        focal_loss = self.focal(inputs , targets)
        ss_input = F.softmax(inputs, dim=1)
        ss_input = torch.argmax(ss_input, dim=1, keepdim=True).float()
        h , w = targets.shape[1] , targets.shape[2]
        targets = targets.view(-1 , 1, h, w).float()
        ssim_loss = self.ssim(ss_input, targets)  
        triple_loss = self.alpha * focal_loss + self.beta * dice_loss + self.gamma * ssim_loss
        return triple_loss  

#######################################################################################

def lovasz_grad(gt_sorted):
  # https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/lovasz_loss.py
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class LovaszSoftmax(nn.Module):
    def __init__(self):
        super(LovaszSoftmax, self).__init__()

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)
        return losses

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs = inputs.view(-1, num_classes)
        targets = targets.view(-1)
        targets = targets.type(inputs.type())
        losses = self.lovasz_softmax_flat(inputs, targets).mean()
        return losses
      


#######################################################################################

_criterion_entrypoints = {
    'CELoss': nn.CrossEntropyLoss(weight = pos_weight),
    'DiceLoss': DiceLoss,
    'DiceCELoss': DiceCELoss,
    'IoULoss': IoULoss,
    'FocalLoss' : FocalLoss,
    'TverskyLoss' : TverskyLoss,
    'FocalTverskyLoss' : FocalTverskyLoss,
    'SSLLoss' : SSLLoss,
    'LogCoshDiceLoss': LogCoshDiceLoss,
    'LovaszSoftmax' : LovaszSoftmax,
    'DiceCELoss' : DiceCELoss,
    'DiceFocalLoss' : DiceFocalLoss,
    'TripleLoss' : TripleLoss,
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

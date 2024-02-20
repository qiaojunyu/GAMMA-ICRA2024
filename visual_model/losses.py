import pdb
from typing import Optional

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np

@torch.no_grad()
def pixel_accuracy(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    Compute pixel accuracy.
    """
    if gt_mask.numel() > 0:
        accuracy = (pred_mask == gt_mask).sum() / gt_mask.numel()
        accuracy = accuracy.item()
    else:
        accuracy = 0.
    return accuracy

@torch.no_grad()
def iou(y_pred, y_true):
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    current = confusion_matrix(y_true, y_pred)
    intersection = np.diag(current)
    gt = current.sum(axis=1)
    pred = current.sum(axis=0)
    union = gt + pred - intersection
    ioU = intersection / union.astype(np.float32) + 1e-8
    mean_iou = np.mean(ioU)
    return mean_iou, ioU

@torch.no_grad()
def iou_evel(y_pred, y_true):
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    current = confusion_matrix(y_true, y_pred)
    intersection = np.diag(current)
    gt = current.sum(axis=1)
    pred = current.sum(axis=0)
    union = gt + pred - intersection
    ioU = intersection / union.astype(np.float32) + 1e-8
    mean_iou = np.mean(ioU)
    gt_label = np.unique(y_true)
    if gt_label.shape[0] != ioU.shape[0]:
        # print("error iou:{} {}".format(gt_label, ioU))
        real_iou = ioU[gt_label]
    else:
        real_iou = ioU
    return mean_iou, real_iou, gt_label

def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        targets = targets[valid_mask]

        if targets.shape[0] == 0:
            return torch.tensor(0.0).to(dtype=inputs.dtype, device=inputs.device)

        inputs = inputs[valid_mask]

    log_p = F.log_softmax(inputs, dim=-1)
    ce_loss = F.nll_loss(
        log_p, targets, weight=alpha, ignore_index=ignore_index, reduction="none"
    )
    log_p_t = log_p.gather(1, targets[:, None]).squeeze(-1)
    loss = ce_loss * ((1 - log_p_t.exp()) ** gamma)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

    if not labels.dtype == torch.int64:
        raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {input.shape}")

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    dice_score = 2.0 * intersection / (cardinality + eps)

    return torch.mean(-dice_score + 1.0)

def compute_miou_loss(pred_seg_per_point, gt_seg_onehot):
    dot = torch.sum(pred_seg_per_point * gt_seg_onehot, axis=1)
    denominator = torch.sum(pred_seg_per_point, axis=1) + torch.sum(gt_seg_onehot, axis=1) - dot
    mIoU = dot / (denominator + 1e-10)
    return torch.mean(1.0 - mIoU)

def compute_coorindate_loss(pred_coordinate_per_point, gt_coordinate_per_point, num_parts, gt_seg_onehot):
    loss_coordinate = 0.0
    coordinate_splits = torch.split(pred_coordinate_per_point, split_size_or_sections=3, dim=2)
    mask_splits = torch.split(gt_seg_onehot, split_size_or_sections=1, dim=2)
    for i in range(num_parts):
        diff_l2 = torch.norm(coordinate_splits[i] - gt_coordinate_per_point, dim=2)
        loss_coordinate += torch.mean(mask_splits[i][:, :, 0] * diff_l2, axis = 1)
    return torch.mean(loss_coordinate, axis=0)

def vec_loss(pre_offset, gt_offsets, mask):
    mask = mask.reshape(-1)
    gt_offsets = gt_offsets.reshape(-1, 3)
    pre_offset = pre_offset.reshape(-1, 3)
    pt_diff = pre_offset - gt_offsets
    pt_dist = torch.sum(pt_diff.abs(), dim=-1)
    loss_pt_offset_dist = pt_dist[mask > 0].mean()
    gt_offsets_norm = torch.norm(gt_offsets, dim=1).reshape(-1, 1)
    gt_offsets = gt_offsets / (gt_offsets_norm + 1e-8)
    pre_offsets_norm = torch.norm(pre_offset, dim=1).reshape(-1, 1)
    pre_offset = pre_offset / (pre_offsets_norm + 1e-8)
    dir_diff = -(gt_offsets * pre_offset).sum(dim=-1)
    loss_offset_dir = dir_diff[mask > 0].mean()
    loss_offset = loss_offset_dir + loss_pt_offset_dist
    return loss_offset

def heatmap_loss_f(pre_heatmap, gt_heat_map, mask):
    mask = mask.reshape(-1)
    pre_heatmap = pre_heatmap.reshape(-1)
    gt_heat_map = gt_heat_map.reshape(-1)
    heatmap_loss = torch.abs(pre_heatmap - gt_heat_map)
    heatmap_loss = heatmap_loss[mask > 0].mean()
    return heatmap_loss

import torch
import numpy as np
from utils import *
from torch.nn import functional as F

def validation_swin(model, data_loader, criterion, device,meta, n_class=12):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        hist = np.zeros((n_class, n_class))
        all_iou = []
        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.long().to(device)            
            imgs = [images]
            outputs = model(imgs,meta,return_loss=False)
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()

            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs,n_class)

            mIoU = label_accuracy_score(masks.detach().cpu().numpy(), outputs)
            mIoU_list.append(mIoU)
            
            batch_iou = batch_iou_score(masks.detach().cpu().numpy(), outputs, len(outputs))
            all_iou.append(batch_iou)
            
        avrg_loss = total_loss / cnt
        miou2 = mIoU_score(hist)
        miou3 = np.mean(all_iou)
    model.train()
    return avrg_loss, np.mean(mIoU_list), miou2, miou3


def mIoU_score(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    return mean_iu


def add_hist(hist, label_trues, label_preds, n_class):
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class=12):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    return mean_iu


def batch_iou_score(label_trues, label_preds, batch_size, n_class=12):
    hist = np.zeros((n_class, n_class))
    batch_iou = 0
    for lt, lp in zip(label_trues, label_preds):
        hist = _fast_hist(lt.flatten(), lp.flatten(), n_class)
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
            )
            batch_iou += np.nanmean(iu) / batch_size
    return batch_iou
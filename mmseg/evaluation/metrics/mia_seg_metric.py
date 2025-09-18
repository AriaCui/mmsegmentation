# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
import copy
import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable
import cv2
from mmseg.registry import METRICS
import pdb
import torch.nn as nn


def cal_intersection(gt, pred):
    gt = gt.flatten()
    pred = pred.flatten()
    inter = (gt * pred).sum()
    gt_area = gt.sum()
    eps = 1e-5

    return (inter + eps) / (gt_area + eps)


def cal_intersection_pred(gt, pred):
    gt = gt.flatten()
    pred = pred.flatten()
    inter = (gt * pred).sum()
    pred_area = pred.sum()
    eps = 1e-5

    return (inter + eps) / (pred_area + eps)


def dice_score(pred, gt):
    smooth = 1e-5

    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    output_ = pred_flat > 0.5
    target_ = gt_flat > 0.5
    # u = output_ & target_
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (2 * intersection + smooth) / (union + intersection + smooth)


class ED_torch(nn.Module):
    def __init__(self, k_size=5):
        super(ED_torch, self).__init__()
        self.max_pool = nn.MaxPool2d(k_size, 1, (k_size - 1) // 2, ceil_mode=False)

    def forward(self, x):
        # 先腐蚀，后膨胀
        x = -self.max_pool(-x)
        x = self.max_pool(x)

        # 先膨胀，后腐蚀
        x = self.max_pool(x)
        x = -self.max_pool(-x)
        return x


@METRICS.register_module()
class MIASegMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 k_size=5,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.k_size = k_size
        self.ed = ED_torch(k_size=k_size).cuda()
        self.ed.eval()


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        data_batch = data_batch['inputs']
        for img, data_sample in zip(data_batch, data_samples):
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            label = data_sample['gt_sem_seg']['data'].squeeze()
            area_intersect, area_union, area_pred_label, area_label = self.intersect_and_union(pred_label, label,
                                                                                               num_classes,
                                                                                               self.ignore_index)

            dice, gt_num, pred_num, pos_num_gt, pos_num_pred = self.recall_precision_f1(pred_label, label)
            self.results.append((
                area_intersect, area_union, area_pred_label, area_label, dice, gt_num, pred_num, pos_num_gt,
                pos_num_pred
            ))
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext('##'.join(
                    data_sample['img_path'].split('/')[-3:]))[0]
                filename = osp.abspath(osp.join(self.output_dir, f'{basename}.png'))
                pred_label = pred_label.cpu().numpy()
                label = label.cpu().numpy()
                output = np.hstack((label, pred_label))
                cv2.imencode('.png', output)[1].tofile(filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        idx = len(results)
        results = tuple(zip(*results))
        # assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)


        mean_dice_txt = sum(results[4])
        txt_gt_num = sum(results[5])
        txt_pred_num = sum(results[6])
        txt_pos_gt_num = sum(results[7])
        txt_pos_pred_num = sum(results[8])

        recall_txt = txt_pos_gt_num / txt_gt_num
        precision_txt = txt_pos_pred_num / (txt_pred_num + 1e-5)
        f1_score = 2 * recall_txt * precision_txt / (recall_txt + precision_txt + 1e-5)
        dice_txt = mean_dice_txt / idx

        recall_txt = round(recall_txt, 4)
        precision_txt = round(precision_txt, 4)
        f1_score = round(f1_score, 4)
        dice_txt = round(dice_txt, 4)

        # print_log('per class results:', logger)
        # print_log('\n' + class_table_data.get_string(), logger=logger)
        # print_log(f"f1-score: {f1_score}\nrecall: {recall_txt}\nprecision: {precision_txt}\ndice: {dice_txt}")

        metrics['f1-score'] = f1_score
        metrics['recall'] = recall_txt
        metrics['precision'] = precision_txt
        metrics['dice_txt'] = dice_txt

        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    def recall_precision_f1(self, pred_label, label):
        pred = pred_label.cpu().numpy()
        mask = label.cpu().numpy()

        iou_thresh1 = 0.1
        iou_thresh2 = 0.1

        if self.k_size > 0:
            pred = torch.from_numpy(pred[None, None, :, :].astype(np.float32)).cuda()
            perd = self.ed(pred)
            pred = perd.cpu().detach().numpy().squeeze().astype(np.uint8)
        dice = dice_score(pred, mask)
        mask = mask.astype(np.uint8)
        pred = pred.astype(np.uint8)
        gt_num, labels_gt, stats_gt, centroids_gt = cv2.connectedComponentsWithStats(mask)
        pred_num, labels_pred, stats_pred, centroids_pred = cv2.connectedComponentsWithStats(pred)

        gt_count = 0
        pos_num_gt = 0
        for i in range(gt_num):
            if i == 0:  # 跳过背景
                continue
            x = stats_gt[i, 0]
            y = stats_gt[i, 1]
            w = stats_gt[i, 2]
            h = stats_gt[i, 3]
            area = stats_gt[i, 4]
            if area <= 25:
                gt_count += 1
                continue
            gt_sub = mask[y:y + h, x:x + w]
            pred_sub = pred[y:y + h, x:x + w]

            iou = cal_intersection(gt_sub, pred_sub)

            if iou >= iou_thresh1:
                pos_num_gt += 1

        pred_count = 0
        pos_num_pred = 0
        for i in range(pred_num):
            if i == 0:  # 跳过背景
                continue
            x = stats_pred[i, 0]
            y = stats_pred[i, 1]
            w = stats_pred[i, 2]
            h = stats_pred[i, 3]
            area = stats_pred[i, 4]
            if area <= 20:
                pred_count += 1
                continue
            gt_sub = mask[y:y + h, x:x + w]
            pred_sub = pred[y:y + h, x:x + w]
            iou = cal_intersection_pred(gt_sub, pred_sub)
            # print("iou",iou)
            if iou >= iou_thresh2:
                pos_num_pred += 1

        gt_num = max(0, gt_num - 1 - gt_count)
        pred_num = max(0, pred_num - 1 - pred_count)  # 应该max(1, pred_num - 1 - pred_count)？


        return dice, gt_num, pred_num, pos_num_gt, pos_num_pred

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta ** 2) * (precision * recall) / (
                    (beta ** 2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        # pdb.set_trace()
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                        total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics

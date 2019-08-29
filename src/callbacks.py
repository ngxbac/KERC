from catalyst.dl.core import Callback, RunnerState
from catalyst.dl.utils.criterion import accuracy
from catalyst.dl.callbacks.logging import TxtMetricsFormatter
from catalyst.contrib.criterion import IoULoss, BCEIoULoss
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import AttentionMiningLoss
import numpy as np
import cv2
from typing import List


class LabelSmoothCriterionCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        loss = criterion(
            state.output[self.output_key],
            state.input[self.input_key]
        )
        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("train"):
            criterion = state.get_key(
                key="criterion", inner_key=self.criterion_key
            )
        else:
            criterion = nn.CrossEntropyLoss()

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class SmoothMixupCallback(LabelSmoothCriterionCallback):
    """
    Callback to do mixup augmentation.
    Paper: https://arxiv.org/abs/1710.09412
    Note:
        MixupCallback is inherited from CriterionCallback and
        does its work.
        You may not use them together.
    """

    def __init__(
        self,
        fields: List[str] = ("images",),
        alpha=0.5,
        on_train_only=True,
        **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, \
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
            state.loader_name.startswith("train")

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)

        for f in self.fields:
            state.input[f] = self.lam * state.input[f] + \
                (1 - self.lam) * state.input[f][self.index]

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed:
            return super()._compute_loss(state, criterion)

        pred = state.output[self.output_key]
        y_a = state.input[self.input_key]
        y_b = state.input[self.input_key][self.index]

        loss = self.lam * criterion(pred, y_a) + \
            (1 - self.lam) * criterion(pred, y_b)
        return loss


class DSCriterionCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0,
        loss_weights: List[float] = None,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier
        self.loss_weights = loss_weights

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        outputs = state.output[self.output_key]
        input = state.input[self.input_key]
        assert len(self.loss_weights) == len(outputs)
        loss = 0
        for i, output in enumerate(outputs):
            loss += criterion(output, input) * self.loss_weights[i]
        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("train"):
            criterion = state.get_key(
                key="criterion", inner_key=self.criterion_key
            )
        else:
            criterion = nn.CrossEntropyLoss()

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class DSAccuracyCallback(Callback):
    """
    Accuracy metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "acc",
        logit_names: List[str] = None,
    ):
        self.prefix = prefix
        self.metric_fn = accuracy
        self.input_key = input_key
        self.output_key = output_key
        self.logit_names = logit_names

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        assert len(outputs) == len(self.logit_names)

        batch_metrics = {}

        for logit_name, output in zip(self.logit_names, outputs):
            metric = self.metric_fn(output, targets)
            key = f"{self.prefix}_{logit_name}"
            batch_metrics[key] = metric[0]

        state.metrics.add_batch_value(metrics_dict=batch_metrics)


class GAINCriterionCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_cls_key: str = "logits",
        output_am_key: str = "logits_am",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0,
    ):
        self.input_key = input_key
        self.output_cls_key = output_cls_key
        self.output_am_key = output_am_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        outputs_cls = state.output[self.output_cls_key]
        outputs_am = state.output[self.output_am_key]
        input = state.input[self.input_key]
        loss = criterion(outputs_cls, input) * 0.9
        loss_am = F.softmax(outputs_am)
        loss_am, _ = loss_am.max(dim=1)
        loss_am = loss_am.sum() / loss_am.size(0)
        loss += loss_am * 0.1
        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("train"):
            criterion = state.get_key(
                key="criterion", inner_key=self.criterion_key
            )
        else:
            criterion = nn.CrossEntropyLoss()

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class GAINMaskCriterionCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        input_mask: str = "masks",
        output_cls_key: str = "logits",
        output_am_key: str = "logits_am",
        output_soft_mask_key: str = "soft_mask",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0,
    ):
        self.input_key = input_key
        self.input_mask = input_mask
        self.output_cls_key = output_cls_key
        self.output_am_key = output_am_key
        self.output_soft_mask_key = output_soft_mask_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier
        self.soft_mask_criterion = nn.BCELoss()

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        outputs_cls = state.output[self.output_cls_key]
        outputs_am = state.output[self.output_am_key]
        output_soft_mask = state.output[self.output_soft_mask_key]
        input = state.input[self.input_key]
        input_mask = state.input[self.input_mask]
        loss = criterion(outputs_cls, input) * 0.8
        loss_am = F.softmax(outputs_am)
        loss_am, _ = loss_am.max(dim=1)
        loss_am = loss_am.sum() / loss_am.size(0)
        loss_mask = self.soft_mask_criterion(output_soft_mask, input_mask)
        loss += loss_am * 0.1
        loss += loss_mask * 0.1
        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("train"):
            criterion = state.get_key(
                key="criterion", inner_key=self.criterion_key
            )
        else:
            criterion = nn.CrossEntropyLoss()

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class GAINSaveHeatmapCallback(Callback):
    def __init__(
        self,
        heatmap_key: str = 'heatmap',
        image_name_key: str = 'image_names',
        image_key: str = 'images',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        outdir: str = './heatmaps/'
    ):
        self.heatmap_key = heatmap_key
        self.image_name_key = image_name_key
        self.image_key = image_key
        self.mean = mean
        self.std = std
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("valid"):
            outdir = os.path.join(self.outdir, f"epoch{state.epoch}")
            os.makedirs(outdir, exist_ok=True)
            images = state.input[self.image_key]
            heatmaps = state.output[self.heatmap_key]
            image_names = state.input[self.image_name_key]

            for image, ac, image_name in zip(images, heatmaps, image_names):
                ac = ac.data.cpu().numpy()[0]
                heat_map = self._combine_heatmap_with_image(
                    image=image,
                    heatmap=ac
                )
                cv2.imwrite(f"{outdir}/{image_name}", heat_map)
                # mask = mask.detach().cpu().numpy() * 255
                # mask = mask[0]
                # cv2.imwrite(f"{outdir}/{image_name}_mask.jpg", mask)

    def denorm(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    def _combine_heatmap_with_image(self, image, heatmap):
        heatmap = heatmap - np.min(heatmap)
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
        heatmap = np.float32(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET))

        scaled_image = self.denorm(image) * 255
        scaled_image = scaled_image.cpu().numpy().astype(np.uint8).transpose((1, 2, 0))

        cam = heatmap + np.float32(scaled_image)
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        heat_map = cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)
        return heat_map


class GCAMSaveHeatmapCallback(Callback):
    def __init__(
        self,
        # feedforward_key: str = 'feedforward',
        # backward_key: str = 'backward',
        head_map_key: str = 'heatmap',
        image_name_key: str = 'image_names',
        image_key: str = 'images',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        outdir: str = './heatmaps/'
    ):
        # self.feedforward_key = feedforward_key
        # self.backward_key = backward_key
        self.head_map_key = head_map_key
        self.image_name_key = image_name_key
        self.image_key = image_key
        self.mean = mean
        self.std = std
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("valid"):
            outdir = os.path.join(self.outdir, f"epoch{state.epoch}")
            os.makedirs(outdir, exist_ok=True)
            images = state.input[self.image_key]
            heatmaps = state.output[self.head_map_key]
            image_names = state.input[self.image_name_key]

            for image, heatmap, image_name in zip(images, heatmaps, image_names):
                # backward = backward.unsqueeze(0)
                # weight = backward.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
                # heatmap = F.relu((weight * forward).sum(dim=1)).squeeze(0)
                # heatmap = cv2.resize(heatmap.data.cpu().numpy(), images.size()[2:])
                heatmap = heatmap.data.cpu().numpy()[0]

                heat_map = self._combine_heatmap_with_image(
                    image=image,
                    heatmap=heatmap
                )
                cv2.imwrite(f"{outdir}/{image_name}", heat_map)

    def denorm(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    def _combine_heatmap_with_image(self, image, heatmap):
        # import pdb
        # pdb.set_trace()
        heatmap = heatmap - np.min(heatmap)
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
        heatmap = np.float32(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET))

        scaled_image = self.denorm(image) * 255
        scaled_image = scaled_image.cpu().numpy().astype(np.uint8).transpose((1, 2, 0))

        cam = heatmap + np.float32(scaled_image)
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        heat_map = cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)
        return heat_map


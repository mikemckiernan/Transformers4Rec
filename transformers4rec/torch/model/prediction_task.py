#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import logging
from typing import Dict, Iterable, Optional

import torch
import torchmetrics as tm

from ..block.base import Block, BuildableBlock, SequentialBlock
from ..block.mlp import MLPBlock
from ..masking import MaskedLanguageModeling
from ..ranking_metric import AvgPrecisionAt, NDCGAt, RecallAt
from ..utils.torch_utils import LambdaModule
from .base import BlockType, PredictionTask

LOG = logging.getLogger("transformers4rec")


class BinaryClassificationPrepareBlock(BuildableBlock):
    def build(self, input_size) -> SequentialBlock:
        return SequentialBlock(
            torch.nn.Linear(input_size[-1], 1, bias=False),
            torch.nn.Sigmoid(),
            LambdaModule(lambda x: x.view(-1)),
            output_size=[
                None,
            ],
        )


class BinaryClassificationTask(PredictionTask):
    DEFAULT_LOSS = torch.nn.BCELoss()
    DEFAULT_METRICS = (
        tm.Precision(num_classes=2, task="binary"),
        tm.Recall(num_classes=2, task="binary"),
        tm.Accuracy(task="binary"),
        # TODO: Fix this: tm.AUC()
    )

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[BlockType] = None,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
        summary_type="first",
    ):
        self.target_dim = 1
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            summary_type=summary_type,
            task_block=task_block,
            pre=BinaryClassificationPrepareBlock(),
            forward_to_prediction_fn=lambda x: torch.round(x).int(),
        )


class RegressionPrepareBlock(BuildableBlock):
    def build(self, input_size) -> SequentialBlock:
        return SequentialBlock(
            torch.nn.Linear(input_size[-1], 1),
            LambdaModule(lambda x: x.view(-1)),
            output_size=[
                None,
            ],
        )


class RegressionTask(PredictionTask):
    DEFAULT_LOSS = torch.nn.MSELoss()
    DEFAULT_METRICS = (tm.regression.MeanSquaredError(),)

    def __init__(
        self,
        target_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_block: Optional[BlockType] = None,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
        summary_type="first",
    ):
        self.target_dim = 1
        super().__init__(
            loss=loss,
            metrics=metrics,
            target_name=target_name,
            task_name=task_name,
            summary_type=summary_type,
            task_block=task_block,
            pre=RegressionPrepareBlock(),
        )


class NextItemPredictionTask(PredictionTask):
    """This block performs item prediction task for session and sequential-based models.
    It requires a body containing a masking schema to use for training and target generation.
    For the supported masking schemes, please refers to:
    https://nvidia-merlin.github.io/Transformers4Rec/main/model_definition.html#sequence-masking

    Parameters
    ----------
    loss: torch.nn.Module
        Loss function to use. Defaults to NLLLos.
    metrics: Iterable[torchmetrics.Metric]
        List of ranking metrics to use for evaluation.
    task_block:
        Module to transform input tensor before computing predictions.
    task_name: str, optional
        Name of the prediction task, if not provided a name will be automatically constructed based
        on the target-name & class-name.
    weight_tying: bool
        The item id embedding table weights are shared with the prediction network layer.
    softmax_temperature: float
        Softmax temperature, used to reduce model overconfidence, so that softmax(logits / T).
        Value 1.0 reduces to regular softmax.
    padding_idx: int
        pad token id.
    target_dim: int
        vocabulary size of item ids
    """

    DEFAULT_METRICS = (
        # default metrics suppose labels are int encoded
        NDCGAt(top_ks=[10, 20], labels_onehot=True),
        AvgPrecisionAt(top_ks=[10, 20], labels_onehot=True),
        RecallAt(top_ks=[10, 20], labels_onehot=True),
    )

    def __init__(
        self,
        loss: torch.nn.Module = torch.nn.NLLLoss(ignore_index=0),
        metrics: Iterable[tm.Metric] = DEFAULT_METRICS,
        task_block: Optional[BlockType] = None,
        task_name: str = "next-item",
        weight_tying: bool = False,
        softmax_temperature: float = 1,
        padding_idx: int = 0,
        target_dim: int = None,
    ):
        super().__init__(loss=loss, metrics=metrics, task_block=task_block, task_name=task_name)
        self.softmax_temperature = softmax_temperature
        self.weight_tying = weight_tying
        self.padding_idx = padding_idx
        self.target_dim = target_dim

        self.item_embedding_table = None
        self.masking = None

    def build(self, body, input_size, device=None, inputs=None, task_block=None, pre=None):
        """Build method, this is called by the `Head`."""
        if not len(input_size) == 3 or isinstance(input_size, dict):
            raise ValueError(
                "NextItemPredictionTask needs a 3-dim vector as input, found:" f"{input_size}"
            )

        # Retrieve the embedding module to get the name of itemid col and its related table
        if not inputs:
            inputs = body.inputs
        if not getattr(inputs, "item_id", None):
            raise ValueError(
                "For Item Prediction task a categorical_module "
                "including an item_id column is required."
            )
        self.embeddings = inputs.categorical_module
        if not self.target_dim:
            self.target_dim = self.embeddings.item_embedding_table.num_embeddings
        if self.weight_tying:
            self.item_embedding_table = self.embeddings.item_embedding_table
            item_dim = self.item_embedding_table.weight.shape[1]
            if input_size[-1] != item_dim and not task_block:
                LOG.warning(
                    f"Projecting inputs of NextItemPredictionTask to'{item_dim}' "
                    f"As weight tying requires the input dimension '{input_size[-1]}' "
                    f"to be equal to the item-id embedding dimension '{item_dim}'"
                )
                # project input tensors to same dimension as item-id embeddings
                task_block = MLPBlock([item_dim])

        # Retrieve the masking from the input block
        self.masking = inputs.masking
        if not self.masking:
            raise ValueError(
                "The input block should contain a masking schema for training and evaluation"
            )
        self.padding_idx = self.masking.padding_idx
        pre = NextItemPredictionPrepareBlock(
            target_dim=self.target_dim,
            weight_tying=self.weight_tying,
            item_embedding_table=self.item_embedding_table,
            softmax_temperature=self.softmax_temperature,
        )
        super().build(
            body, input_size, device=device, inputs=inputs, task_block=task_block, pre=pre
        )

    def forward(self, inputs: torch.Tensor, targets=None, training=False, testing=False, **kwargs):
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]
        x = inputs.float()

        if self.task_block:
            x = self.task_block(x)  # type: ignore

        # Retrieve labels from masking
        if training or testing:
            labels = self.masking.masked_targets  # type: ignore
            trg_flat = labels.flatten()
            non_pad_mask = trg_flat != self.padding_idx
            labels_all = torch.masked_select(trg_flat, non_pad_mask)
            # remove padded items, keep only masked positions
            x = self.remove_pad_3d(x, non_pad_mask)
            x = self.pre(x)  # type: ignore
            loss = self.loss(x, labels_all)
            return {
                "loss": loss,
                "labels": labels_all,
                "predictions": x,
                # "pred_metadata": {},
                # "model_outputs": [],
            }
        else:
            # Get the hidden position to use for predicting the next item
            labels = self.embeddings.item_seq
            non_pad_mask = labels != self.padding_idx
            rows_ids = torch.arange(labels.size(0), dtype=torch.long, device=labels.device)
            if isinstance(self.masking, MaskedLanguageModeling):
                last_item_sessions = non_pad_mask.sum(dim=1)
            else:
                last_item_sessions = non_pad_mask.sum(dim=1) - 1
            x = x[rows_ids, last_item_sessions]

        # Compute predictions probs
        x = self.pre(x)  # type: ignore

        return x

    def remove_pad_3d(self, inp_tensor, non_pad_mask):
        # inp_tensor: (n_batch x seqlen x emb_dim)
        inp_tensor = inp_tensor.flatten(end_dim=1)
        inp_tensor_fl = torch.masked_select(
            inp_tensor, non_pad_mask.unsqueeze(1).expand_as(inp_tensor)
        )
        out_tensor = inp_tensor_fl.view(-1, inp_tensor.size(1))
        return out_tensor

    def calculate_metrics(self, predictions, targets) -> Dict[str, torch.Tensor]:  # type: ignore
        if isinstance(targets, dict) and self.target_name:
            targets = targets[self.target_name]

        outputs = {}
        predictions = self.forward_to_prediction_fn(predictions)

        for metric in self.metrics:
            outputs[self.metric_name(metric)] = metric(predictions, targets)

        return outputs

    def compute_metrics(self):
        metrics = {
            self.metric_name(metric): metric.compute()
            for metric in self.metrics
            if getattr(metric, "top_ks", None)
        }
        # Explode metrics for each cut-off
        # TODO make result generic:
        # To accept a mix of ranking metrics and others not requiring top_ks ?
        topks = {self.metric_name(metric): metric.top_ks for metric in self.metrics}
        results = {}
        for name, metric in metrics.items():
            for measure, k in zip(metric, topks[name]):
                results[f"{name}_{k}"] = measure
        return results


class NextItemPredictionPrepareBlock(BuildableBlock):
    def __init__(
        self,
        target_dim: int,
        weight_tying: bool = False,
        item_embedding_table: Optional[torch.nn.Module] = None,
        softmax_temperature: float = 0,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.weight_tying = weight_tying
        self.item_embedding_table = item_embedding_table
        self.softmax_temperature = softmax_temperature

    def build(self, input_size) -> Block:
        return Block(
            _NextItemPredictionTask(
                input_size,
                self.target_dim,
                self.weight_tying,
                self.item_embedding_table,
                self.softmax_temperature,
            ),
            [-1, self.target_dim],
        )


class _NextItemPredictionTask(torch.nn.Module):
    """Predict the interacted item-id probabilities.

    - During inference, the task consists of predicting the next item.
    - During training, the class supports the following Language modeling tasks:
        Causal LM, Masked LM, Permutation LM and Replacement Token Detection

    Parameters:
    -----------
        input_size: int
            Input size of this module.
        target_dim: int
            Dimension of the target.
        weight_tying: bool
            The item id embedding table weights are shared with the prediction network layer.
        item_embedding_table: torch.nn.Module
            Module that's used to store the embedding table for the item.
        softmax_temperature: float
            Softmax temperature, used to reduce model overconfidence, so that softmax(logits / T).
            Value 1.0 reduces to regular softmax.
    """

    def __init__(
        self,
        input_size: int,
        target_dim: int,
        weight_tying: bool = False,
        item_embedding_table: Optional[torch.nn.Module] = None,
        softmax_temperature: float = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.target_dim = target_dim
        self.weight_tying = weight_tying
        self.item_embedding_table = item_embedding_table
        self.softmax_temperature = softmax_temperature
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        if self.weight_tying:
            self.output_layer_bias = torch.nn.Parameter(torch.Tensor(self.target_dim))
            torch.nn.init.zeros_(self.output_layer_bias)
        else:
            self.output_layer = torch.nn.Linear(
                self.input_size[-1], self.target_dim  # type: ignore
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.weight_tying:
            logits = torch.nn.functional.linear(
                inputs,
                weight=self.item_embedding_table.weight,  # type: ignore
                bias=self.output_layer_bias,
            )
        else:
            logits = self.output_layer(inputs)

        if self.softmax_temperature:
            # Softmax temperature to reduce model overconfidence
            # and better calibrate probs and accuracy
            logits = torch.div(logits, self.softmax_temperature)

        predictions = self.log_softmax(logits)

        return predictions

    def _get_name(self) -> str:
        return "NextItemPredictionTask"

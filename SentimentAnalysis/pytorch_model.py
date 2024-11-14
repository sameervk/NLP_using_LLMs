import sys
from typing import Any

import lightning
import torch
import lightning as L
import torchmetrics
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification

class PytorchLogReg(torch.nn.Module):

    def __init__(self, num_features: int, num_classes: int):

        super().__init__()

        self.linear = torch.nn.Linear(num_features, num_classes, bias=True, dtype=torch.float32)

    def forward(self, X):

        logits = self.linear(X)
        return logits


class LightningModel(L.LightningModule):

    def __init__(self, pytorch_model: torch.nn.Module, learning_rate: float):

        super().__init__()

        self.model = pytorch_model
        self.learning_rate = learning_rate

        # accuracy trackers
        self.train_acc = torchmetrics.Accuracy(task="multiclass", threshold=0.5, num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", threshold=0.5, num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", threshold=0.5, num_classes=2)

    def forward(self, x):

        return self.model(x)

    def _shared_step(self, batch):

        features, true_labels = batch

        logits = self(features)
        # forward method is called in this manner

        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        # using cross_entropy, labels are automatically converted to one-hot encoding

        predicted_labels = torch.argmax(logits, dim=1)
        # dim= 1 is along the columns, in this case the num of classes

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)
        # on_step=False, on_epoch=True
        # using these parameters is resulting in progress bar for every batch item in the validation data
        # and an error message: IOPub message rate exceeded.

        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("test_loss", loss, prog_bar=True)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, prog_bar=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:

        optimiser = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        return optimiser


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


class LightningModelDistilBERT(lightning.LightningModule):

    def __init__(self, model: DistilBertForSequenceClassification, learning_rate=5e-5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def training_step(self, batch, batch_idx):

        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["label"]
                       )

        self.log("train_loss", outputs["loss"])
        # unlike in the above LightningModel where loss had to be calculated, here the calculation of loss is implicit.

        return outputs["loss"]

    def validation_step(self, batch, batch_idx):

        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["label"]
                       )

        logits = outputs["logits"]
        # before softmax is applied

        predicted_labels = torch.argmax(logits, dim=1)
        self.val_acc(predicted_labels, batch["label"])

        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_loss", outputs["loss"], prog_bar=True)

    def test_step(self, batch, batch_idx):

        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["label"]
                       )

        logits = outputs["logits"]
        # before softmax is applied

        predicted_labels = torch.argmax(logits, dim=1)
        self.test_acc(predicted_labels, batch["label"])

        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_loss", outputs["loss"], prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(),
                                     lr=self.learning_rate
                                     )
        return optimizer

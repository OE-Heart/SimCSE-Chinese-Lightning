from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, SpearmanCorrCoef

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simcse_unsup_loss(y_pred):
    """无监督的损失函数.

    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    # FIXME: /opt/conda/conda-bld/pytorch_1656352660876/work/aten/src/ATen/native/cuda/Loss.cu:271: nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [16,0,0] Assertion `t >= 0 && t < n_classes` failed.
    loss = F.cross_entropy(sim, y_true)
    return loss


def simcse_sup_loss(y_pred):
    """有监督的损失函数.

    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss


class SimcseModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        supervise: bool,
        lr: float = 0.001,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = simcse_sup_loss if self.hparams.supervise else simcse_unsup_loss

        self.val_corrcoef = SpearmanCorrCoef()
        self.test_corrcoef = SpearmanCorrCoef()

        # for logging best so far validation corrcoef
        self.val_corrcoef_best = MaxMetric()

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.net(input_ids, attention_mask, token_type_ids)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_corrcoef_best doesn't store corrcoef from these checks
        self.val_corrcoef_best.reset()

    def training_step(self, batch: Any, batch_idx: int):
        dim = 3 if self.hparams.supervise else 2
        real_batch_num = batch.get("input_ids").shape[0]
        input_ids = batch.get("input_ids").view(real_batch_num * dim, -1).to(DEVICE)
        attention_mask = batch.get("attention_mask").view(real_batch_num * dim, -1).to(DEVICE)
        token_type_ids = batch.get("token_type_ids").view(real_batch_num * dim, -1).to(DEVICE)

        output = self.forward(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(output)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        source, target, label = batch

        # source        [batch, 1, seq_len] -> [batch, seq_len]
        source_input_ids = source.get("input_ids").squeeze(1).to(DEVICE)
        source_attention_mask = source.get("attention_mask").squeeze(1).to(DEVICE)
        source_token_type_ids = source.get("token_type_ids").squeeze(1).to(DEVICE)
        source_pred = self.forward(source_input_ids, source_attention_mask, source_token_type_ids)
        loss = self.criterion(source_pred)

        # target        [batch, 1, seq_len] -> [batch, seq_len]
        target_input_ids = target.get("input_ids").squeeze(1).to(DEVICE)
        target_attention_mask = target.get("attention_mask").squeeze(1).to(DEVICE)
        target_token_type_ids = target.get("token_type_ids").squeeze(1).to(DEVICE)
        target_pred = self.forward(target_input_ids, target_attention_mask, target_token_type_ids)

        sim = F.cosine_similarity(source_pred, target_pred, dim=-1)

        # log val metrics
        corrcoef = self.val_corrcoef(sim, label.float())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/corrcoef", corrcoef, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "corrcoef": corrcoef}

    def validation_epoch_end(self, outputs: List[Any]):
        corrcoef = self.val_corrcoef.compute()  # get val corrcoef from current epoch
        self.val_corrcoef_best.update(corrcoef)
        self.log(
            "val/corrcoef_best", self.val_corrcoef_best.compute(), on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        source, target, label = batch

        source_input_ids = source.get("input_ids").squeeze(1).to(DEVICE)
        source_attention_mask = source.get("attention_mask").squeeze(1).to(DEVICE)
        source_token_type_ids = source.get("token_type_ids").squeeze(1).to(DEVICE)
        source_pred = self.net(source_input_ids, source_attention_mask, source_token_type_ids)

        target_input_ids = target.get("input_ids").squeeze(1).to(DEVICE)
        target_attention_mask = target.get("attention_mask").squeeze(1).to(DEVICE)
        target_token_type_ids = target.get("token_type_ids").squeeze(1).to(DEVICE)

        output = self.forward(target_input_ids, target_attention_mask, target_token_type_ids)
        loss = self.criterion(output)

        sim = F.cosine_similarity(source_pred, output, dim=-1)

        # log test metrics
        corrcoef = self.val_corrcoef(sim, label.float())
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/corrcoef", corrcoef, on_step=False, on_epoch=True)

        return {"loss": loss, "corrcoef": corrcoef}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.test_corrcoef.reset()
        self.val_corrcoef.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.AdamW(params=self.parameters(), lr=self.hparams.lr)

import random
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from src.datamodules.components.simcse_dataset import SimcseDataset, load_data


class SimcseDataModule(LightningDataModule):
    def __init__(
        self,
        PLM_path: str,
        snli_train_dir: str,
        sts_train_dir: str,
        sts_dev_dir: str,
        sts_test_dir: str,
        supervise: bool,
        train_size: int = 10000,
        max_length: int = 64,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.PLM_path)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if self.hparams.supervise:
                train_data = load_data("snli", self.hparams.snli_train_dir, self.hparams.supervise)
            else:
                train_data_snli = load_data(
                    "snli", self.hparams.snli_train_dir, self.hparams.supervise
                )
                train_data_sts = load_data(
                    "sts", self.hparams.sts_train_dir, self.hparams.supervise
                )
                train_data = train_data_snli + [_[0] for _ in train_data_sts]  # 两个数据集组合
                train_data = random.sample(train_data, self.hparams.train_size)
            self.data_train = SimcseDataset(
                data=train_data,
                split="train",
                supervise=self.hparams.supervise,
                tokenizer=self.tokenizer,
                max_length=self.hparams.max_length,
            )

            dev_data = load_data("sts", self.hparams.sts_dev_dir, self.hparams.supervise)
            self.data_val = SimcseDataset(
                data=dev_data,
                split="test",
                supervise=self.hparams.supervise,
                tokenizer=self.tokenizer,
                max_length=self.hparams.max_length,
            )

            test_data = load_data("sts", self.hparams.sts_test_dir, self.hparams.supervise)
            self.data_test = SimcseDataset(
                data=test_data,
                split="test",
                supervise=self.hparams.supervise,
                tokenizer=self.tokenizer,
                max_length=self.hparams.max_length,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

from typing import List

import jsonlines
from torch.utils.data import Dataset


def load_data(name: str, path: str) -> List:
    """根据名字加载不同的数据集."""
    def load_snli_data(path):
        with jsonlines.open(path, 'r') as f:
            return [line.get('origin') for line in f]

    def load_lqcmc_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [line.strip().split('\t')[0] for line in f]

    def load_sts_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]

    assert name in ["snli", "lqcmc", "sts"]
    if name == 'snli':
        return load_snli_data(path)
    return load_lqcmc_data(path) if name == 'lqcmc' else load_sts_data(path)

class SimcseDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法."""
    def __init__(self, data: List, split, supervise, tokenizer, max_length):
        self.data = data
        self.split = split
        self.supervise = supervise
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        if self.split == "train":
            if self.supervise is False:
                return self.tokenizer([text, text], max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
            else:
                return self.tokenizer([text[0], text[1], text[2]], max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        elif self.split == "test":
            return self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        if self.split == "train":
            return self.text_2_id(self.data[index])
        elif self.split == "test":
            line = self.data[index]
            return self.text_2_id([line[0]]), self.text_2_id([line[1]]), int(line[2])

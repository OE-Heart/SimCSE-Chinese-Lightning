import torch
from torch import nn
from transformers import BertConfig, BertModel


class SimcseModel(nn.Module):
    def __init__(self, PLM_path, supervise, pooling, dropout=0.3):
        super(SimcseModel, self).__init__()
        if supervise is False:
            config = BertConfig.from_pretrained(PLM_path)
            config.attention_probs_dropout_prob = dropout   # 修改config的dropout系数
            config.hidden_dropout_prob = dropout
            self.bert = BertModel.from_pretrained(PLM_path, config=config)
        else:
            self.bert = BertModel.from_pretrained(PLM_path)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):

        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            return out.pooler_output            # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]

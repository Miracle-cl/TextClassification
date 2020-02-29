# some other files refer to Bert Folder

import torch
import torch.nn as nn
from transformers import (DistilBertConfig, DistilBertTokenizer, DistilBertPreTrainedModel, 
                          DistilBertModel, AdamW, get_linear_schedule_with_warmup, 
                          WEIGHTS_NAME, CONFIG_NAME, )


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


class DistilBertForClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.distilbert = DistilBertModel(config)
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # self.init_weights() # change initial weights

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        distilbert_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, hidden_size)
        pooled_output = hidden_state[:, 0]  # (bs, hidden_size)
        pooled_output = F.relu(self.pre_classifier(pooled_output))  # (bs, hidden_size)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# Initializing a DistilBERT configuration
configuration = DistilBertConfig(num_labels=133)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DistilBertForClassification(configuration)
model.to(device)
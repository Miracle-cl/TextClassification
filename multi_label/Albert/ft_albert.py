# some other files refer to Bert Folder

import torch
import torch.nn as nn
from transformers import (AlbertConfig, AlbertTokenizer, AlbertPreTrainedModel, AlbertModel,
                          get_linear_schedule_with_warmup, AdamW,
                          WEIGHTS_NAME, CONFIG_NAME, )


tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')


class AlBertForClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.albert = AlbertModel(config)
        self.albert = AlbertModel.from_pretrained('albert-base-v2')
        # self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.init_weights() # change initial weights

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1] # (bs, hidden_size)
        # pooled_output = F.relu(self.pre_classifier(pooled_output))  # (bs, hidden_size)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# Initializing an ALBERT-base style configuration
configuration = AlbertConfig(hidden_size=768,
                            num_attention_heads=12,
                            intermediate_size=3072,
                            num_labels=133)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlBertForClassification(configuration)


# Initializing an ALBERT-xlarge style configuration
# configuration = AlbertConfig(hidden_size=2048,
#     num_attention_heads=16,
#     num_hidden_layers=24,
#     intermediate_size=8192,
#     num_labels=133)
# need more cuda memory

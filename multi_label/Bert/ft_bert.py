import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertForClassification(BertPreTrainedModel):
    """
    if train, then BertModel.from_pretrained('bert-base-uncased'), 
        which will load pretrain weights;
    if predict, the BertModel(config), 
        which just load our own weights.
    """
    def __init__(self, config):
        super(BertForClassification, self).__init__(config)
        # self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel(config) # not load from pretrain
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.init_weights() # ignore will change weights
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

import os
import torch
import numpy as np

from transformers import BertConfig, BertTokenizer

from .ft_bert import BertForClassification

class ModelPredict:
    def __init__(self, id2label, active_value=0.5, max_num=3, max_len=256, 
                 model_path='./bert_models/'):
        self.max_len = max_len
        self.half_len = max_len // 2

        self.active_value = active_value
        self.max_num = max_num
        self.id2label = id2label

        self.model, self.tokenizer = self.load_model(model_path)
        self.pad_id = self.tokenizer.pad_token_id # 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model.to(self.device)

    def predict(self, sent):
         input_ids, att_mask = self.prepare_model_data(sent)
         logits = self.predict_prob(input_ids, att_mask)
         pred_labels = self.predict_labels(logits)
         return pred_labels

    @staticmethod
    def load_model(model_path):
        config = BertConfig.from_json_file(os.path.join(model_path, 'config.json'))
        model = BertForClassification(config)
        # initialize model with cpu
        # state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')
        # initialize model with gpu
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
        model.load_state_dict(state_dict)
        tokenizer = BertTokenizer(os.path.join(model_path, 'vocab.txt'), do_lower_case=True)
        return model, tokenizer

    def truncate_pad_sent(self, sent_id):
        ll = len(sent_id)
        if ll > self.max_len:
            # get first half_len ids and last half_len ids
            return sent_id[:self.half_len] + sent_id[-self.half_len:]
        if ll < self.max_len:
            return sent_id + [self.pad_id] * (self.max_len - ll)
        return sent_id

    def prepare_model_data(self, sent):
        encoded_sent = self.tokenizer.encode(sent, add_special_tokens=True)

        tp_sent_id = self.truncate_pad_sent(encoded_sent)
        assert len(tp_sent_id) == self.max_len, 'Length of sent id is wrong'

        att_mask = [int(token_id > self.pad_id) for token_id in tp_sent_id]
        return torch.tensor([tp_sent_id]).long(), torch.tensor([att_mask]).long()

    def predict_prob(self, input_ids, att_mask):
        self.model.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            att_mask = att_mask.to(self.device)
            logits = self.model(input_ids, att_mask)
            logits = torch.sigmoid(logits).squeeze(0)
        return logits.cpu().numpy()

    def fit_active_value(self, s1):
        return (s1 > self.active_value).astype(int)

    def cut_max_num(self, s1):
        max_ids = np.argsort(s1)[-self.max_num:]
        s3 = np.zeros_like(s1)
        s3[max_ids] = 1
        return s3

    def predict_labels(self, logits):
        s2 = self.fit_active_value(logits)

        # get max prob of predicts
        if sum(s2) > self.max_num:
            s2 = self.cut_max_num(logits)

        pids = np.where(s2 == 1)[0]

        pred_labels = [self.id2label[j] for j in pids]

        return pred_labels


if __name__ == '__main__':
    id2label = {0: 'a', 1: 'b'}
    model_predict = ModelPredict(id2label)
    sent = 'aa bb cc dd'
    pred_labels = model_predict.predict(sent)
    print(pred_labels)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class PhraseData(Dataset):
    def __init__(self, src, tgt):
        super(PhraseData, self).__init__()
        self.src = src
        self.tgt = tgt
        
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, i):
        return self.src[i], self.tgt[i]
    
def collate_func(seqs, pad_token=0):
    seq_lens = [len(seq) for seq in seqs]
    max_len = max(seq_lens)
    seqs = [seq + [pad_token] * (max_len - len(seq)) for seq in seqs]
    return torch.LongTensor(seqs), torch.LongTensor(seq_lens)    
    
def pair_collate_func(inps):
    pairs = sorted(inps, key=lambda p: len(p[0]), reverse=True)
    seqs, tgt = zip(*pairs)
    seqs, seq_lens = collate_func(seqs)
    return seqs, seq_lens, torch.FloatTensor(tgt) 


class BiGRU(nn.Module):
    def __init__(self, emb_dim, hidden_size, n_layers, num_classes, dropout, weights):
        super(BiGRU, self).__init__()
        self.emb = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
        self.dropout1 = nn.Dropout(p=dropout)
        self.gru = nn.GRU(emb_dim, hidden_size, n_layers, bidirectional=True, batch_first=True)
        self.maxpooling = nn.AdaptiveMaxPool1d(1)
        # self.dropout2 = nn.Dropout(p=dropout)
        self.linear = nn.Linear(2*hidden_size, num_classes)
        
    def forward(self, input_var, input_len):
        embeded = self.emb(input_var) # b x l x emb_dim
        embeded = self.dropout1(embeded)
        
        total_length = embeded.size(1)
        
        packed1 = torch.nn.utils.rnn.pack_padded_sequence(embeded, input_len, batch_first=True)
        self.gru.flatten_parameters()
        rnn1, hidden1 = self.gru(packed1)
        rnn1, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn1, batch_first=True, total_length=total_length) # b x l x 2hs

        rnn1 = rnn1.permute(0, 2, 1) # b x 2hs x l
        out = self.maxpooling(rnn1).squeeze(2) # b x 2hs x 1 -> b x 2hs
        # out = self.dropout2(out)
        out = self.linear(out)
        return out
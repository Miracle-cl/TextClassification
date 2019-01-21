import pickle
import numpy as np
import time
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

class Lang():
    def __init__(self, sents, stoplist, min_count=30):
        self.sents = sents
        self.stoplist = stoplist
        self.min_count = min_count
        self.word2idx = {"<PAD>": 0}
        self.idx2word = {0: "<PAD>"}
        self.n_words = self.process_sents()

    def process_sents(self):
        words = []
        for sent in self.sents:
            words += sent.split(' ')

        cc = 1
        counter = Counter(words)
        for word, num in counter.items():
            if num > self.min_count and word not in self.stoplist:
                self.word2idx[word] = cc
                self.idx2word[cc] = word
                cc += 1
        return cc

def collate_fn(insts, PAD_token=0):
    # if seq_pad in class then all seqs with same length
    maxlen = max([len(x) for x in insts])
    seq = np.array([x + [PAD_token] * (maxlen - len(x)) for x in insts])
    seq_lens = np.array([len(x) for x in insts])
    return torch.LongTensor(seq), torch.LongTensor(seq_lens)

def paired_collate_fn(insts):
    #src_insts, tgt_insts = list(zip(*insts))
    seq_pairs = sorted(insts, key=lambda p: len(p[0]), reverse=True)
    src_insts, tgt_insts = zip(*seq_pairs)
    src_insts = collate_fn(src_insts)
    # tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, tgt_insts)

class NewsPaperDatasets(Dataset):
    def __init__(self, src, tgt):
        # self.device = device
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

class SimpleGRU(nn.Module):
    def __init__(self, weights, embed_dim=300, hidden_size=256, n_layers=1, dropout=0.5, output_size=20):
        super(SimpleGRU, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
        self.dropout1 = nn.Dropout(p=dropout)
        self.gru = nn.GRU(embed_dim, hidden_size, n_layers, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(p=dropout)
        # self.fc = nn.Linear(2 * hidden_size, output_size)
        self.fc = nn.Linear(6 * hidden_size, output_size)

    def forward(self, input_var, input_len):
        embeded = self.embed(input_var)
        embeded = self.dropout1(embeded)
        total_length = embeded.size(1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeded, input_len, batch_first=True)
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(packed)
        # print(hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=total_length)

        # method-1: as it is a classification problem, we just grab the last hidden state -0.85
        # outputs = F.relu(self.dropout2( torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) ))

        # method-2: cat last 2 hidden state , avg-pooling and max-pooling - 0.86
        avgpool = F.adaptive_avg_pool1d(outputs.permute(0, 2, 1), 1).squeeze(2)
        maxpool = F.adaptive_max_pool1d(outputs.permute(0, 2, 1), 1).squeeze(2)
        outputs =  F.relu(self.dropout2( torch.cat((hidden[-2, :, :], hidden[-1, :, :], avgpool, maxpool), dim=1) ))

        outputs = self.fc(outputs)
        return outputs


def train_epoch(model, device, epoch, train_loader, test_loader, criterion, optimizer, clip=5., batch_sz=64):
    # print("There are {} batches in one epoch.".format( len(train_loader) ))
    model.train()

    train_loss = 0
    t0 = time.time()

    for i, batch in enumerate(train_loader, 1):
        src, src_lens, tgt = batch
        src = src.to(device)
        tgt = torch.tensor(tgt).to(device) # tgt without modified - cuda out of memory

        optimizer.zero_grad() # here same as optimizer.zero_grad()
        # hidden = hidden.detach()
        outputs = model(src, src_lens)
        # print(outputs.size(), tgt.size())
        loss = criterion(outputs, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        if i % 10 == 0:
            # print loss info every 20 Iterations
            log_str = "Epoch : {} , Iteration : {} , Time : {:.2f} , TrainLoss : {:.4f}".format \
                        (epoch, i, time.time()-t0, train_loss/i)
            print(log_str)
            t0 = time.time()

    train_loss = train_loss / len(train_loader)
    # print(train_loss)

    model.eval()
    eval_loss = 0

    corr = total = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            src, src_lens, tgt = batch
            total += len(tgt)
            src = src.to(device)
            tgt = torch.tensor(tgt).to(device)
            outputs = model(src, src_lens)
            loss = criterion(outputs, tgt)
            eval_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            corr += (pred.cpu().numpy() == tgt.cpu().numpy()).sum()

        eval_loss = eval_loss / len(test_loader)

        accuracy = corr / total
    return model, train_loss, eval_loss, accuracy

def main(batch_size = 64, multi_gpu = False):
    with open('newspaper_data.pkl', 'rb') as rf:
        data_split = pickle.load(rf)

    x_train, y_train = data_split['train']['x'], data_split['train']['y']
    x_test, y_test = data_split['test']['x'], data_split['test']['y']
    embed_matrix = np.load('EmbeddingMatrix.npy')

    train_loader = torch.utils.data.DataLoader(
                        NewsPaperDatasets(x_train, y_train),
                        num_workers = 2,
                        batch_size = batch_size,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    test_loader = torch.utils.data.DataLoader(
                        NewsPaperDatasets(x_test, y_test),
                        num_workers = 2,
                        batch_size = batch_size,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sgru = SimpleGRU(embed_matrix, n_layers=2)
    if multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sgru = torch.nn.DataParallel(sgru, device_ids=[0, 1], dim=0)
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    sgru.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(sgru.parameters())

    n_epochs = 20
    best_eval_loss = float('inf') # best eval accuracy 0.90

    for epoch in range(1, 1+n_epochs):
        sgru, train_loss, eval_loss, eval_acc = train_epoch(sgru, device, epoch, train_loader, test_loader, criterion, optimizer)

        print(">> Epoch : {} , TrainLoss : {:.4f} , EvalLoss : {:.4f} , EvalAccuracy : {:.4f}\n\n".format \
              (epoch, train_loss, eval_loss, eval_acc))

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(sgru.state_dict(), 'sgru_newspaper_0114.pt')

if __name__ == "__main__":
    main(batch_size = 256, multi_gpu = True)

    """
    line 82 self.gru.flatten_parameters() solve the below problems

    .../lib/python3.6/site-packages/torch/nn/modules/rnn.py:182:
    RuntimeWarning: RNN module weights are not part of single contiguous chunk of memory.
    This means they need to be compacted at every call, possibly greatly increasing memory usage.
    To compact weights again call flatten_parameters().
    self.num_layers, self.dropout, self.training, self.bidirectional)

    """
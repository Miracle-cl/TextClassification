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
    #maxlen = 24
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


class SimpleLSTM(nn.Module):
    def __init__(self, device, weights, embed_dim=300, batch_size=256, 
                 hidden_size=256, n_layers=1, dropout=0.5, output_size=20):
        super(SimpleLSTM, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
        self.dropout1 = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_size, n_layers, bidirectional=True)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2 * hidden_size, output_size)
        
    def forward(self, input_var, input_len, states):
        embeded = self.embed(input_var)
        embeded = self.dropout1(embeded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeded, input_len)
        outputs, states = self.lstm(packed, states)
        
        outputs, out_len = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        hidden = states[0]
        outputs = F.relu(self.dropout2( torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) ))
        outputs = self.fc(outputs)
        return outputs, states
    
    def init_hidden(self):
        states = (torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size, device=self.device), 
                  torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size, device=self.device))
        return states
    
def train_epoch(model, epoch, train_loader, test_loader, criterion, optimizer, clip=5., batch_sz=256):
    # print("There are {} batches in one epoch.".format( len(train_loader) ))
    model.train()
    states = model.init_hidden()
    
    train_loss = 0
    t0 = time.time()
    
    for i, batch in enumerate(train_loader, 1):
        src, src_lens, tgt = batch
        src = src.permute(1, 0).to(model.device)
        tgt = torch.tensor(tgt).to(model.device) # tgt without modified - cuda out of memory
        
        optimizer.zero_grad() # here same as optimizer.zero_grad()
        states = [state.detach() for state in states]
        print(states[0])
        outputs, states = model(src, src_lens, states)
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
            src = src.permute(1, 0).to(model.device)
            tgt = torch.tensor(tgt).to(model.device)
            outputs, _ = model(src, src_lens, states)
            loss = criterion(outputs, tgt)
            eval_loss += loss.item()
            
            _, pred = torch.max(outputs, 1)
            corr += (pred.cpu().numpy() == tgt.cpu().numpy()).sum()

        eval_loss = eval_loss / len(test_loader)
        
        accuracy = corr / total
    return model, train_loss, eval_loss, accuracy

def main():
    with open('newspaper_data.pkl', 'rb') as rf:
        data_split = pickle.load(rf)

    x_train, y_train = data_split['train']['x'], data_split['train']['y']
    x_test, y_test = data_split['test']['x'], data_split['test']['y']
    embed_matrix = np.load('EmbeddingMatrix.npy')

    train_loader = torch.utils.data.DataLoader(
                        NewsPaperDatasets(x_train, y_train),
                        num_workers = 2,
                        batch_size = 256,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    test_loader = torch.utils.data.DataLoader(
                        NewsPaperDatasets(x_test, y_test),
                        num_workers = 2,
                        batch_size = 256,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SimpleLSTM(device, embed_matrix, n_layers=2)
    model.to(model.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    n_epochs = 20
    best_eval_loss = float('inf')

    for epoch in range(1, 1+n_epochs):
        model, train_loss, eval_loss, eval_acc = train_epoch(model, epoch, train_loader, test_loader, criterion, optimizer)

        print(">> Epoch : {} , TrainLoss : {:.4f} , EvalLoss : {:.4f} , EvalAccuracy : {:.4f}\n\n".format \
              (epoch, train_loss, eval_loss, eval_acc))

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), 'lstm_newspaper_0114.pt')
                
if __name__ == "__main__":
    main()
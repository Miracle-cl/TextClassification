import pickle
import numpy as np
import time
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.elmo import Elmo, batch_to_ids

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
    
class ELMoGRU(nn.Module):
    def __init__(self, device, options_file, weight_file, embed_dim=256, batch_size=64, 
                 hidden_size=256, n_layers=1, dropout=0.5, output_size=20):
        super(ELMoGRU, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        
        # self.elmo = Elmo(options_file, weight_file, 1, dropout=0) # 0115
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0, requires_grad=False) #0116
        ## print([x.requires_grad == False for x in self.elmo._elmo_lstm.parameters()])
        ##  x.requires_grad == True for x in elmo1.scalar_mix_0.parameters()
        # self.elmo.weight.requires_grad = False // wrong
        self.dropout1 = nn.Dropout(p=dropout)
        self.gru = nn.GRU(embed_dim, hidden_size, n_layers, bidirectional=True)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2 * hidden_size, output_size)
        
    def forward(self, input_seqs, input_len, hidden):
        
        character_ids = batch_to_ids(input_seqs).to(self.device)
        embeded = self.elmo(character_ids)['elmo_representations'][0].permute(1, 0, 2)
        # print(embeded)
        
        #embeded = self.embed(input_var)
        embeded = self.dropout1(embeded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeded, input_len)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        
        # method-1: as it is a classification problem, we just grab the last hidden state -0.85
        outputs = F.relu(self.dropout2( torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) ))
        
        # method-2: cat last 2 hidden state , avg-pooling and max-pooling - 0.86
        # avgpool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).squeeze(2)
        # maxpool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).squeeze(2)
        # outputs =  F.relu(self.dropout2( torch.cat((hidden[-2, :, :], hidden[-1, :, :], avgpool, maxpool), dim=1) ))
        
        outputs = self.fc(outputs)
        return outputs, hidden
    
    def init_hidden(self):
        return torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size, device=self.device)
    
def get_batches(x, y, batch_size=8):
    n_batches = len(x) // batch_size
    x = x[ : (n_batches * batch_size)]
    y = y[ : (n_batches * batch_size)]
    for i in range(0, n_batches * batch_size, batch_size):
        bx, by = x[i : (i+batch_size)], y[i : (i+batch_size)]
        bxy = sorted(zip(bx, by), key=lambda p: len(p[0]), reverse=True)
        bx, by = zip(*bxy)
        bx_len = [len(p) for p in bx]
        yield bx, bx_len, by

        
def train_epoch(model, epoch, criterion, optimizer, x_train, y_train, x_test, y_test, input_lang, clip=1., batch_sz=256):
    n_batches_train = len(x_train) // batch_sz
    n_batches_test = len(x_test) // batch_sz
    
    log_infos = []
    
    model.train()
    hidden = model.init_hidden()

    train_loss = 0
    t0 = time.time()

    for i, batch in enumerate(get_batches(x_train, y_train, batch_size=batch_sz), 1):
        bx, bx_len, by = batch
        int_seq = [[input_lang.idx2word[idx] for idx in sent] for sent in bx]

        by = torch.tensor(by).to(model.device) # tgt without modified - cuda out of memory

        optimizer.zero_grad() # here same as optimizer.zero_grad()
        hidden = hidden.detach()
        outputs, hidden = model(int_seq, bx_len, hidden)

        loss = criterion(outputs, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        if i % 20 == 0:
            # print loss info every 20 Iterations
            log_str = "Epoch : {} , Iteration : {} , Time : {:.2f} , TrainLoss : {:.4f}".format \
                        (epoch, i, time.time()-t0, train_loss/i)
            print(log_str)
            log_infos.append(log_str)
            t0 = time.time()

    train_loss = train_loss / n_batches_train

    model.eval()
    eval_loss = 0

    corr = total = 0
    with torch.no_grad():
        for i, batch in enumerate(get_batches(x_test, y_test, batch_size=batch_sz), 1):
            bx, bx_len, by = batch
            total += len(by)
            int_seq = [[input_lang.idx2word[idx] for idx in sent] for sent in bx]

            by = torch.tensor(by).to(model.device)
            
            outputs, _ = model(int_seq, bx_len, hidden)
            loss = criterion(outputs, by)
            eval_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            corr += (pred.cpu().numpy() == by.cpu().numpy()).sum()
            if i % 20 == 0:
                log_str = "Epoch : {} , Iteration : {} , EvalLoss : {:.4f}".format(epoch, i, eval_loss/i)
                log_infos.append(log_str)
                print(log_str)

        eval_loss = eval_loss / n_batches_test

        accuracy = corr / total
    return model, train_loss, eval_loss, accuracy, log_infos

def main(batch_size = 8):
    with open('newspaper_data.pkl', 'rb') as rf:
        data_split = pickle.load(rf)

    x_train, y_train = data_split['train']['x'], data_split['train']['y']
    x_test, y_test = data_split['test']['x'], data_split['test']['y']
    input_lang = data_split['lang']
    
    options_file = "./small_emlo/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    weight_file = "./small_emlo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = ELMoGRU(device, options_file, weight_file, batch_size=batch_size)
    model.to(model.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    n_epochs = 20
    best_eval_loss = float('inf')
    log_txt = []
    
    for epoch in range(1, 1+n_epochs):
        starttime = time.time()
        model, train_loss, eval_loss, eval_acc, log_infos = train_epoch(model, epoch, criterion, optimizer, 
                                                               x_train, y_train, x_test, y_test, input_lang, clip=1., batch_sz=batch_size)
        used_time = (time.time() - starttime) / 60
        log_txt += log_infos
        log_str = ">> Epoch : {} , Time : {:.2f} , TrainLoss : {:.4f} , EvalLoss : {:.4f} , EvalAccuracy : {:.4f}\n\n".format \
                          (epoch, used_time, train_loss, eval_loss, eval_acc)
        print(log_str)
        log_txt.append(log_str)

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), 'elmo_newspaper_0116.pt')
            
    with open('elmo_log_info_0116.txt', 'w') as wrf:
        for item in log_txt:
            wrf.write(item)
            wrf.write('\n')
            
if __name__ == "__main__":
    main(8)
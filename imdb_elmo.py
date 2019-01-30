import pickle
import numpy as np
import time
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.elmo import Elmo, batch_to_ids

def collate_fn(insts, PAD_token=1):
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
    return (*src_insts, tgt_insts)

class IMDBdatasets(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


class ELMoGRU(nn.Module):
    def __init__(self, device, idx2word, options_file, weight_file, embed_dim=256,
                 hidden_size=256, n_layers=1, dropout=0.5, output_size=2):
        super(ELMoGRU, self).__init__()
        self.device = device
        self.idx2word = idx2word

        # self.elmo = Elmo(options_file, weight_file, 1, dropout=0) # 0115
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0, requires_grad=False) #0116
        ## print([x.requires_grad == False for x in self.elmo._elmo_lstm.parameters()])
        ##  x.requires_grad == True for x in elmo1.scalar_mix_0.parameters()
        # self.elmo.weight.requires_grad = False // wrong
        self.dropout1 = nn.Dropout(p=dropout)
        self.gru = nn.GRU(embed_dim, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, input_seq, input_len, hidden=None):
        
        # int_seq = [[self.idx2word[idx] for idx in sent] for sent in bx]

        # character_ids = batch_to_ids(int_seq).to(self.device)
        # embeded = self.elmo(character_ids)['elmo_representations'][0] # B x seqlen x embed_dim
        embeded = self.elmo_embedding(input_seq)
        total_length = embeded.size(1)

        embeded = self.dropout1(embeded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeded, input_len, batch_first=True)
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=total_length)

        # method-1: as it is a classification problem, we just grab the last hidden state -0.85
        outputs = F.relu(self.dropout2( torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) ))

        # method-2: cat last 2 hidden state , avg-pooling and max-pooling - 0.86
        # avgpool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).squeeze(2)
        # maxpool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).squeeze(2)
        # outputs =  F.relu(self.dropout2( torch.cat((hidden[-2, :, :], hidden[-1, :, :], avgpool, maxpool), dim=1) ))

        outputs = self.fc(outputs)
        # print("\tIn Model: input size", embeded.size(), "output size", outputs.size())
        return outputs
    
    def elmo_embedding(self, bx):
        int_seq = [[self.idx2word[idx.item()] for idx in sent if idx.item() != 1] for sent in bx] # pad_token = 1
        # print('len-', len(int_seq)) # num of sentences : if 2gpu the batch_size/2 
        character_ids = batch_to_ids(int_seq).to(self.device)
        embeded = self.elmo(character_ids)['elmo_representations'][0] # B x seqlen x embed_dim
        # print('embeded-', embeded.size())
        return embeded

def train_epoch(model, device, epoch, criterion, optimizer, train_loader, test_loader, idx2word, clip=1., batch_sz=8):
    log_infos = []

    model.train()

    train_loss = 0
    t0 = time.time()

    for i, batch in enumerate(train_loader, 1):
        src, src_len, tgt = batch
        # int_seq = [[idx2word[idx] for idx in sent] for sent in bx]
        tgt = torch.tensor(tgt).to(device) # tgt without modified - cuda out of memory

        optimizer.zero_grad() # here same as optimizer.zero_grad()
        outputs = model(src, src_len)
        # print("Outside: output_size", outputs.size())

        loss = criterion(outputs, tgt)
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

    train_loss = train_loss / len(train_loader)

    model.eval()
    eval_loss = 0

    corr = total = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            src, src_len, tgt = batch
            total += len(tgt)
            
            tgt = torch.tensor(tgt).to(device)

            outputs = model(src, src_len)
            loss = criterion(outputs, tgt)
            eval_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            corr += (pred.cpu().numpy() == tgt.cpu().numpy()).sum()
            if i % 20 == 0:
                log_str = "Epoch : {} , Iteration : {} , EvalLoss : {:.4f}".format(epoch, i, eval_loss/i)
                log_infos.append(log_str)
                print(log_str)

        eval_loss = eval_loss / len(test_loader)

        accuracy = corr / total
    return model, train_loss, eval_loss, accuracy, log_infos

def main(BATCH_SIZE = 16, MAX_LEN = 400, multi_gpu = True):
    with open('../data/imdb_data.pkl', 'rb') as rf:
        data_split = pickle.load(rf)
    
    x_train, y_train = data_split['train']['x'], data_split['train']['y']
    x_test, y_test = data_split['test']['x'], data_split['test']['y']
    x_train = [x[:MAX_LEN] for x in x_train]
    x_test = [x[:MAX_LEN] for x in x_test]
    word2idx = data_split['word2idx']
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    train_loader = torch.utils.data.DataLoader(
                        IMDBdatasets(x_train, y_train),
                        num_workers = 2,
                        batch_size = BATCH_SIZE,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    test_loader = torch.utils.data.DataLoader(
                        IMDBdatasets(x_test, y_test),
                        num_workers = 2,
                        batch_size = BATCH_SIZE,
                        collate_fn = paired_collate_fn,
                        shuffle = True,
                        drop_last = True)

    options_file = "./small_emlo/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    weight_file = "./small_emlo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    
    if multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ELMoGRU(device, idx2word, options_file, weight_file)
        model = nn.DataParallel(model, device_ids=[0, 1], dim=0)
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model = ELMoGRU(device, idx2word, options_file, weight_file)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    n_epochs = 15
    best_eval_loss = float('inf')
    log_txt = []
    
    print("There are {} batches in one epoch.".format( len(train_loader) + len(test_loader) ))
    for epoch in range(1, 1+n_epochs):
        starttime = time.time()
        model, train_loss, eval_loss, eval_acc, log_infos = train_epoch(model, device, epoch, criterion, optimizer,
                                                                        train_loader, test_loader, 
                                                                        idx2word, clip=1., batch_sz=BATCH_SIZE)
        used_time = (time.time() - starttime) / 60
        log_txt += log_infos
        log_str = ">> Epoch : {} , Time : {:.2f} , TrainLoss : {:.4f} , EvalLoss : {:.4f} , EvalAccuracy : {:.4f}\n\n".format \
                          (epoch, used_time, train_loss, eval_loss, eval_acc)
        print(log_str)
        log_txt.append(log_str)

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), 'elmo_imdb_0120.pt')

    with open('elmo_imdb_log_info_0120.txt', 'w') as wrf:
        for item in log_txt:
            wrf.write(item)
            wrf.write('\n')

if __name__ == "__main__":
    main(BATCH_SIZE = 50, MAX_LEN = 400, multi_gpu = True)
    # best eval accuracy 0.88

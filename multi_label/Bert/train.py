import os
import pickle
import time
import torch
torch.cuda.set_device(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import (BertConfig, BertModel, BertTokenizer, BertPreTrainedModel, AdamW, \
                          get_linear_schedule_with_warmup, WEIGHTS_NAME, CONFIG_NAME)

from .ft_bert import BertForClassification


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def prepare_dataloader(batch_size):
    with open('./bert_process_data.pkl', 'rb') as f:
        _process_data = pickle.load(f)
        
    train_inputs, val_inputs = _process_data['train_inputs'], _process_data['val_inputs']
    train_labels, val_labels = _process_data['train_labels'], _process_data['val_labels']
    train_masks, val_masks = _process_data['train_masks'], _process_data['val_masks']

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(val_inputs, val_masks, val_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    print("Train Data size : {} * {}".format(len(train_dataloader), batch_size))
    print("Val Data size : {} * {}".format(len(validation_dataloader), batch_size))
    return train_dataloader, validation_dataloader


def mean_column_wise_auc(y_true, y_pred):
    # return roc_auc_score(y_true, y_pred)
    assert y_true.shape[1] == y_pred.shape[1], 'Arrays must have the same dimension'
    list_of_aucs = []
    for column in range(y_true.shape[1]):
        # print(sum(y_true[:,column]), sum(y_pred[:,column]))
        if sum(y_true[:,column]) == 0:
            continue
        list_of_aucs.append(roc_auc_score(y_true[:,column],y_pred[:,column]))
    # print(list_of_aucs)
    return np.array(list_of_aucs).mean(), len((list_of_aucs))


def train_epoch(model, device, epoch, train_dataloader, validation_dataloader, 
                criterion, optimizer, scheduler, clip=5.):
    model.train()
    train_loss = 0
    t0 = time.time()
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader, 1):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(b_input_ids, b_input_mask)
        loss = criterion(outputs, b_labels)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

        if step % 1000 == 0:
            # print loss info every 20 Iterations
            log_str = "Epoch : {} , Iteration : {} , Time : {:.2f} , TrainLoss : {:.9f}".format \
                        (epoch, step, time.time()-t0, train_loss/step)
            print(log_str)
            t0 = time.time()
    train_loss /= len(train_dataloader)

    model.eval()
    eval_loss = 0
    y_true = None
    y_pred = None
    with torch.no_grad():
        for step, batch in enumerate(validation_dataloader, 1):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(b_input_ids, attention_mask=b_input_mask)
            loss = criterion(outputs, b_labels)
            eval_loss += loss.item()

            if y_true is None:
                y_true = b_labels
            else:
                y_true = torch.cat((y_true, b_labels), 0)

            if y_pred is None:
                y_pred = outputs
            else:
                y_pred = torch.cat((y_pred, outputs), 0)

        eval_loss /= len(validation_dataloader)
        y_pred = torch.sigmoid(y_pred)
        val_acu = mean_column_wise_auc(y_true.cpu().numpy(), y_pred.cpu().numpy())

    return model, optimizer, train_loss, eval_loss, val_acu


def save_transformers(model, tokenizer):
    output_dir = "./bert_models/"
    # Step 1: Save a model, configuration and vocabulary that you have fine-tuned
    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def check_params(model):
    params = list(model.named_parameters())
    for p in params:
        print(p[0], p[1].size(), p[1].requires_grad) # , p[1][:10]


def train(batch_size=16, multi_gpus=False, n_epochs=9):
    train_dataloader, validation_dataloader = prepare_dataloader(batch_size)

    # Initializing a BERT configuration
    configure = BertConfig(num_labels=133)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = BertForClassification(configure)
    if multi_gpus:
        model = nn.DataParallel(model, device_ids=[0, 1], dim=0)
    model.to(device)
    print(model.config)

    check_params(model)

    criterion = nn.BCEWithLogitsLoss()

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * n_epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    print('========= Begin Training ==========')
    clip = 2.0
    best_eval_loss = float('inf')
    for epoch in range(1, 1+n_epochs):
        model, optimizer, train_loss, eval_loss, val_auc = train_epoch(model, device, epoch, 
                                                    train_dataloader, validation_dataloader, 
                                                    criterion, optimizer, scheduler, clip=clip)

        print(">> Epoch : {} , TrainLoss : {:.9f} , EvalLoss : {:.9f}".format \
            (epoch, train_loss, eval_loss))
        print(">> Validation AUC: {}\n".format(val_auc))

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_transformers(model, tokenizer)
            # torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    train(batch_size=16, multi_gpus=False)
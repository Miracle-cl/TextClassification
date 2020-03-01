# not standard py file
import pickle
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def sent2tokenids(inputs, tokenizer):
    input_ids = [] # List[List[int]]

    for sent in inputs:
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
    # print(len(input_ids))
    return input_ids


def prepare_attention_mask(new_inp_ids, pad_id):
    assert pad_id == 0, 'bug of pad id'
    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in new_inp_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > pad_id) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    print('Length of Attention mask : {}'.format(len(attention_masks)))
    return attention_masks


def truncate_pad_id(input_ids, tokenizer, max_len=256):
    half_len = max_len // 2

    # truncation and padding
    pad_id = tokenizer.pad_token_id # 0

    new_inp_ids = []
    for x in input_ids:
        ll = len(x)
        if ll > max_len:
            new_inp_ids.append(x[:half_len] + x[-half_len:])
        elif ll < max_len:
            new_inp_ids.append(x + [pad_id] * (max_len-ll))
        else:
            new_inp_ids.append(x)
        assert len(new_inp_ids[-1]) == max_len

    print('Length of inputs : {}'.format(len(new_inp_ids)))
    attention_masks = prepare_attention_mask(new_inp_ids, pad_id)
    return new_inp_ids, attention_masks


if __name__ == '__main__':
    sents = ['aa bb', 'cc dd'] # List of str (sentence)
    labels = [[], []] # multi_labels
    test_sents = []
    test_labels = []

    # >> 1st: sent -> sent_id
    sent_ids = sent2tokenids(sents, tokenizer)
    test_sent_ids = sent2tokenids(test_sents, tokenizer)

    # >> 2nd: truncate and pad sent_ids to max_len
    input_ids, attention_masks = truncate_pad_id(sent_ids, tokenizer)
    test_input_ids, test_attention_masks = truncate_pad_id(test_sent_ids, tokenizer)

    # >> 3rd: split train set and val set
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels,
                                                                random_state=2020, test_size=0.15)
    ## Do the same for the masks.
    train_masks, val_masks, _, _ = train_test_split(attention_masks, labels,
                                                random_state=2020, test_size=0.15)

    print(len(train_inputs), len(train_masks), len(train_labels), len(val_labels))

    # >> 4th: convert list or numpy to tensor and save dataset
    train_inputs = torch.tensor(train_inputs).long()
    train_masks = torch.tensor(train_masks).long()
    train_labels = torch.tensor(train_labels).float()

    val_inputs = torch.tensor(val_inputs).long()
    val_masks = torch.tensor(val_masks).long()
    val_labels = torch.tensor(val_labels).float()

    test_inputs = torch.tensor(test_input_ids).long()
    test_masks = torch.tensor(test_attention_masks).long()
    test_labels = torch.tensor(test_labels).float()

    _process_data = {'train_inputs': train_inputs, 'val_inputs': val_inputs, 'test_inputs': test_inputs,
                    'train_labels': train_labels, 'val_labels': val_labels, 'test_labels': test_labels,
                    'train_masks': train_masks, 'val_masks': val_masks, 'test_masks': test_masks,
                    }
    with open('./process_data.pkl', 'wb') as f:
        pickle.dump(_process_data, f)


    # >> 5th: convert tensor to dataloader for training
    batch_size = 16

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(val_inputs, val_masks, val_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Create the DataLoader for our testing set.
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    print(len(test_dataloader))

import numpy as np
import pandas as pd #csv file
import torch, torchtext
from torch import nn, optim
from torch.optim import Adam
from torchtext import data

class ImprovedRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)

        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded =self.dropout(self.embedding(text))

        #pack sequence
        packed_embedded = nn.utils.run.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)

#define train step
def train(model, iterator, optimizer, criterion):
    epoch_loss, epoch_acc, epoch_denom = 0, 0, 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions.reshape(-1,1), batch.label.float().reshape(-1,1))
        acc = accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_denom += len(batch)
    return epoch_loss/epoch_denom, epoch_acc, epoch_denom

#similar to previous exercise, we define our accuracy function
def accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum()
    return acc

#define evaluate step
def evaluate(model, iterator, criterion):
    epoch_loss, epoch_acc, epoch_denom = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label.float())
            acc = accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_denom += len(batch)
    return epoch_loss/epoch_denom, epoch_acc, epoch_denom



def toy():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device =", device)
        
    input_data_path = "/input/imdb-dataset-sentiment-analysis-in-csv-format/"
    
    #define fields for our input dataset
    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    LABEL = data.Field(sequential = False, use_vocab = False)

    #Define a list of tuples of fields
    trainval_fields = [("text", TEXT), ("label", LABEL)]

    #Contruct dataset
    train_data, val_data = data.TabularDataset.splits(path=input_data_path, train="Train.csv", validation="Valid.csv", format="csv", skip_header=True, fields=trainval_fields)

    #Build Vocab using pretrained
    MAX_VOCAB_SIZE=25000
    TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors='fasttext.simple.300d')
    BATCH_SIZE = 64

    train_iterator, val_iterator = data.BuckedIterator.splits((train_data, val_data), batch_size = BATCH_SIZE, sort_key = lambda x:len(x.text), sort_within_batch = True, device = device)

    
    #define model input parameters
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5

    #create model instance
    model = ImprovedRNN(INPUT_DIM,
                        EMBEDDING_DIM,
                        HIDDEN_DIM,
                        OUTPUT_DIM,
                        N_LAYERS,
                        BIDIRECTIONAL,
                        DROPOUT,
                        PAD_IDX)

    #copy pretrained vector weights
    model.embedding.weight.data.copy_(pretrained_embeddings)

    #initialize the embedding with 0 for pad as well as unknown tokens
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    print(model.embedding.weight.data)

    #define optimizer, loss function and load to gpu
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    
    #finally lets train our model for 5 epochs
    N_EPOCHS = 5

    for epoch in range(N_EPOCHS):
        train_loss, train_acc, train_num = train(model, train_iterator,
                                                 optimizer, criterion)
        valid_loss, valid_acc, val_num = evaluate(model, val_iterator, criterion)
        print("Epoch-", epoch)
        print(f'\tTrain Loss: {train_loss: .3f} | Train Predicted Correct: {train_acc}
        | Train Denom: {train_num} |
        PercAccuracy: {train_acc/train_num}')

        print(f'\tValid Loss: {valid_loss: .3f}
        | Valid Predicted Correct: {valid_acc}
        | Val Denom: {val_num}
        | PercAccuracy: {train_acc/train_num}')

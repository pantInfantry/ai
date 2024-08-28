import numpy as np #linear algebra
import pandas as pd # data processing, csv file i/o (e.g. pd.read_csv)
import torch
from torch import nn, optim

#import torchtext
from torchtext.datasets import IMDB
import torchtext.data as data
#import torchtext.legacy.data as data
#from torchtext.legacy.data import Field
#from torchtext.legacy import Field
#from torchtext.legacy.data import *
#from torchtext.legacy import data
#from torchtext import Field

class RNNModel(nn.Module):
    def __init__(self, embedding_dim, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.Embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embed = self.Embedding(text)
        output, hidden = self.rnn(embed)
        out = self.fc(hidden.squeeze(0))

        return (out)

#define training step
def train(model, data_iterator, optimizer, loss_function):
    epoch_loss, epoch_acc, epoch_denom =0, 0, 0

    model.train() #Explicity set model to train mode

    for i, batch in enumerate(data_iterator):
        optimizer.zero_grad()
        predictions = model(batch.text)

        loss = loss_function(predictions.reshape(-1,1), batch.label.float().reshape(-1,1))
        acc = accuracy(predictions.reshpae(-1,1), batch.label.reshape(-1,1))

        loss.backwoard()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_denom += len(batch)
    return epoch_loss / epoch_denom, epoch_acc, epoch_denom

#Define evaluation step
def evaluate(model, data_iterator, loss_function):
    epoch_loss, epoch_acc, epoch_denom = 0, 0, 0
    model.eval() #Explcitly set model to eval mode
    for i, batch in enumerate(data_iterator):
        with torch.no_grad():
            predictions = model(batch.text)
            loss = loss_function(predictions.reshape(-1,1), batch.label.float().reshape(-1,1))

            acc = accuracy(predictions.reshape(-1,1), batch.label.reshape(-1,1))

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_denom += len(batch)
    return epoch_loss/epoch_denom, epoch_acc, epoch_denom

#compute binary accuracy
def accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))

    #count the number of correctly predicted outcomes
    correct = (rounded_preds == y).float()
    acc = correct.sum()
    return acc

def toy():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Device=", device)

    input_data_path = "./input/archive/"

    print(torch.__version__)

    #read the csv dataset using pandas
    df = pd.read_csv(input_data_path + "Train.csv")
    print("DF.shape :\n", df.shape)
    print("df.label = ", df.label.value_counts())
    #df.head()

    #define a custom tokenizer
    my_tokenizer = lambda x:str(x).split()

    #define fields for out input dataset
    TEXT = data.Field(sequential=True, lower=True, tokenize=my_tokenizer,
                      use_vocab=True)

    LABEL = data.Field(sequential=False, use_vocab = False)

    #define input fields as a list of tuples of fields
    trainval_fields = [("text", TEXT), ("label", LABEL)]

    #Contruct dataset
    train_data, val_data = data.TabularDataset.splits(path=input_data_path,
                                                      train = "Train.csv", validation = "Valid.csv", format = "csv", skip_header=True, fields= trainval_fields)

    #build vocabulary
    MAX_VOCAB_SIZE = 25000
    TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)

    #define iterators for train and validation
    train_iterator = data.BucketIterator(train_data, device=device,
                                         batch_size=32
                                         ,sort_key=lambda x:len(x.text)
                                         ,sort_within_batch = False
                                         ,repeat = False)
    val_iterator = data.BucketIterator(val_data, device=device, batch_size=32
                                       ,sort_key=lambda x:len(x.text)
                                       ,sort_within_batch=False
                                       ,repeat=False)
    print(TEXT.vocab.freqs.most_common()[:10])


    #define model
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1

    #Create model instance
    model = RNNModel(EMBEDDING_DIM, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

    #define optimizer, loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    criterion = nn.BCEWithLogitsLoss()

    #transfer component to GPU, if available
    Model = model.to(device)
    criterion = criterion.to(device)

    n_epochs = 5

    for epoch in range(n_epochs):
        #train and evaluate
        train_loss, train_acc, train_num = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc, val_num = evaluate(model, val_iterator, criterion)
        print(f'\tTrain Loss: {train_loss: .3f} | Train Predicted Correct : {train_acc} | Train Denom: {train_num} | PercAccuracy: {train_acc/train_num}')
        
        print(f'\tValid Loss: {valid_loss: .3f} | Valid Predicted Correct: {valid_acc} | Val Denom: {val_num} | PercAccuracy: {train_acc/train_num}')

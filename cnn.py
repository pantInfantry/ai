import torch
from torch.utils.data import DataLoader, TensorDataset

#neural net imports
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.autograd import Variable

#import external libraries
import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

#%matplotlib inline

#define convolution neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(ConvNet, self).__init__()

        #First unit of convolution
        self.conv_unit_1 = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        #Second unit of convolution
        self.conv_unit_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        #fully connected layers
        self.fc1 = nn.Linear(7*7*32,128)
        self.fc2 = nn.Linear(128,10)

    #connect the units
    def forward(self,x):
        out = self.conv_unit_1(x)
        out = self.conv_unit_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

#define functions for model evaluation and generating predictions
def make_predictions(data_loader, model):
    #explcitly set the model to eval mode
    model.eval()

    test_preds = torch.LongTensor()
    actual = torch.LongTensor()

    for data, target in data_loader:
        if torch.cuda.is_available():
            data = data.cuda()
        output = model(data)

        #predict output/take the index of the output with max value
        preds = output.cpu().data.max(1, keepdim=True)[1]

        #combine tensors from each batch
        test_preds = torch.cat((test_preds, preds), dim= 0)
        actual = torch.cat((actual, target), dim = 0)

    return actual, test_preds

#Evaluate model
def evaluate(data_loader, model):
    model.eval()
    loss = 0
    correct = 0

    for data, target in data_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        output = model(data)
        loss += F.cross_entropy(output, target, size_average=False).data.item()
        predicted = output.data.max(1, keepdim=True)[1]
        correct += (target.reshape(-1,1) == predicted.reshape(-1,1)).float().sum()
    loss /= len(data_loader.dataset)
    print('\nAverage Val Loss:{:.4f}, Val Accuracyh: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100.*correct / len(data_loader.dataset)))
    
def toy():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    input_folder_path = "./"

    #image in the csv file
    #one row , one image 28*28

    train_df = pd.read_csv(input_folder_path + "train.csv")

    train_labels = train_df['label'].values

    train_images = (train_df.iloc[:,1:].values).astype('float32')

    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, random_state=2020,
                     test_size = 0.2)

    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    val_images = val_images.reshape(val_images.shape[0], 1, 28, 28)

    for i in range(0,6):
        plt.subplot(160 + (i+1))
        plt.imshow(train_images[i].reshape(28,28), cmap = plt.get_cmap('gray'))
        plt.title(train_labels[i])
    plt.show()

    #convert images data type
    train_images_tensor = torch.tensor(train_images)/255.0
    train_images_tensor = train_images_tensor.view(-1,1,28,28)
    train_labels_tensor = torch.tensor(train_labels)

    #Create a train TensorDataset
    train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)

    #convert validation images' data type
    val_images_tensor = torch.tensor(val_images)/255.0
    val_images_tensor = val_images_tensor.view(-1,1, 28,28)
    val_labels_tensor = torch.tensor(val_labels)

    #create a validation tensor dataset
    val_tensor = TensorDataset(val_images_tensor, val_labels_tensor)

    print("Train Labels shape:", train_labels_tensor.shape)
    print("Train Images Shape:", train_images_tensor.shape)
    print("Validation Labels Shape:", val_labels_tensor.shape)
    print("Validation Images Shape:", val_images_tensor.shape)

    #Load train and validation tensorDataset into the data generator for Training
    train_loader = DataLoader(train_tensor, batch_size = 64
                              , num_workers = 2, shuffle = True)
    val_loader = DataLoader(val_tensor, batch_size=64, num_workers =2, shuffle=True)
    

    #create model  for instance
    model = ConvNet(10).to(device)

    #Define loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    print(model)

    num_epochs =5

    #train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            #Fordward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            #backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #after each epoch print train loss and validation loss + accuracy
        print('Epoch [{}/{}, Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        evaluate(val_loader, model)

    #make predictions on validation dataset
    actual, predicted = make_predictions(val_loader, model)
    actual, predicted = np.array(actual).reshape(-1,1), np.array(predicted).reshape(-1,1)

    print("Validation Accuray-", round(accuracy_score(actual, predicted),4)*100)
    print("\n Confusion matrix\n", confusion_matrix(actual, predicted))

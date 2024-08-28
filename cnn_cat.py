#import required libraries
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import glob, os
import matplotlib.image as mpimg

new_path = "./input/"
images = []

#define  Convolutional network
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        #first unit of convolution
        self.conv_unit_1 = nn.Sequential(nn.Conv2d(3,16, kernel_size=3,
                                        stride=1, padding = 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2))#112
        #second unit of convolution
        self.conv_unit_2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) #56

        #third unit of convolution
        self.conv_unit_3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) #28

        #fourth unit of convolution
        self.conv_unit_4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) #14

        #fully connected layers
        self.fc1 = nn.Linear(14*14*128, 128)
        self.fc2 = nn.Linear(128,1)
        self.final = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_unit_1(x)
        out = self.conv_unit_2(out)
        out = self.conv_unit_3(out)
        out = self.conv_unit_4(out)

        #reshpae the output
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.final(out)
        return (out)

def evaluate(model, data_loader):
    loss=[]
    correct = 0
    with torch.no_grad():
        for image, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            model.eval()
            output = model(images)
            predicted = output > 0.5
            correct += (labels.reshape(-1,1) == predicted.reshape(-1,1)).float().sun()
            #clear memory
            del([images, labels])
            if device == "cuda":
                torch.cuda.empty_cache()
    print('\nVal Accuracy: {}/{} ({:.3f}%)\n'.format(
        correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
        
def toy():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Device:", device)

    #collect cat images
    for img_path in glob.glob(os.path.join(new_path, "train", "cat", "*.jpg"))[:5]:
        global images
        images.append(mpimg.imread(img_path))

    #collect dog images
    for img_path in glob.glob(os.path.join(new_path, "train", "dog", "*.jpg"))[:5]:
        images.append(mpimg.imread(img_path))

    #plot a grid of cats and dogs
    plt.figure(figsize=(20,10))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(int(len(images)/columns) + 1, columns, i+1)
        plt.imshow(image)
    plt.show()

    #compose sequence of transformations for image
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #load in each dataset and apply transformations using the
    #torchvision.datasets as datasets library
    train_set = datasets.ImageFolder(os.path.join(new_path, "train")
                                     , transform = transformations)

    val_set = datasets.ImageFolder(os.path.join(new_path, "test")
                                   , transform = transformations)

    #put into a Dataloader using torch library
    train_loader = torch.utils.data.DataLoader(train_set
                                               ,batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 32,
                                             shuffle=True)

    # instance model and training the model
    num_epochs = 10
    loss_function = nn.BCELoss() #binary cross entropy loss
    model = ConvNet()
    #model.cuda()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    #train the model
    total_step = len(train_loader)
    print("Total Batches:", total_step)

    for epoch in range(num_epochs):
        model.train()
        train_loss=0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            outputs = model(images)
            loss = loss_function(outputs.float(), labels.float().view(-1,1))

            #backward and optimize
            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()
            train_loss += loss.item() * labels.size(0)
        #after each epoch print train loss and validation loss + accuracy
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_pochs,
            loss.item()))

        #evaluate model after each training epoch
        evaluate(model, val_loader)

import torch
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
import torch.nn as nn
import glob, os
import matplotlib.image as mpimg

images = []

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
    # prepare data########################################
    new_path = "./input/"
    
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

    ########################
    new_model = models.vgg16(pretrained=True)

    #freeze model weights
    for param in new_model.parameters():
        param.requires_grad = False

    print(new_model.classifier)

    #define our custom model last layer
    new_model.classifier[6] = nn.Sequential(
        nn.Linear(new_model.classifier[6].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256,1),
        nn.Sigmoid())

    #Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in new_model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in new_model.parameters() if p.requires_grad)

    print(f'{total_trainable_params:,} training parameters.')


    #define epochs, optimizer and loss function
    num_epochs = 10
    loss_function = nn.BCELoss() #binary cross entropy Loss

    if torch.cuda.is_available():
        new_model.cuda()
    adam_optimizer = torch.optim.Adam(new_model.parameters(), lr = 0.001)

    #train the model
    total_step = len(train_loader)
    print("Total Batches:", total_step)

    for epoch in range(num_epochs):
        new_model.train()
        train_loss = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            outputs = new_model(images)
            loss = loss_function(outputs.float(), labels.float().view(-1,1))

            #backward and optimize
            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()
            train_loss += loss.item() * labels.size(0)
        #after each epoch print train loss and validation loss + accuracy
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs,
            loss.item()))

        #after each epoch evaluate model
        evaluate(new_model, val_loader)

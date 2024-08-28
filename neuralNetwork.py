import torch as tch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot

#import required libraries
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def neural():
    samples = 5000

    train_split = int(samples*0.8)

    X, y = make_blobs(n_samples=samples, centers=2, n_features=64, cluster_std=10, random_state = 2020)

    y = y.reshape(-1,1)

    X,y=tch.from_numpy(X), tch.from_numpy(y)
    X,y = X.float(), y.float()

    X_train, x_test = X[:train_split], X[train_split:]

    Y_train, y_test = y[:train_split], y[train_split:]

    print("X_train.shape:", X_train.shape)
    print("x_test.shape:", x_test.shape)
    print("Y_train.shape:", Y_train.shape)
    print("y_test.shape:", y_test.shape)
    print("x.dtype:",X.dtype)
    print("y.dtype:", y.dtype)

    #Create an object of the Neural Network class
    model = NeuralNetwork()

    loss_function = nn.BCELoss() #Binary cross entropy loss

    #define optimizer
    adam_optimizer = tch.optim.Adam(model.parameters(), lr = 0.001)

    #Define epochs and batch size
    num_epochs = 10
    batch_size = 16

    #Calling the function for training and pass model, optimizer, loss and related parameters
    #print("num_epochs=", str(num_epochs))
    adam_loss = train_network(model, adam_optimizer, loss_function, num_epochs, batch_size, X_train, Y_train, x_test)

    #2 RMSProp optimizer
    rmsprp_optimizer = tch.optim.RMSprop(model.parameters(), lr = 0.01, alpha = 0.9, eps = 1e-08, weight_decay = 0.1, momentum = 0.1, centered=True)
    print("RMSProp...")
    rmsprop_loss = train_network(model, rmsprp_optimizer, loss_function,num_epochs, batch_size, X_train, Y_train, x_test)

    #3 SGD optimizer
    sgd_optimizer = tch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("SGD...")
    sgd_loss = train_network(model, sgd_optimizer, loss_function, num_epochs, batch_size, X_train, Y_train, x_test)

    #Plot the losses for each optimizer across epochs
    import matplotlib.pyplot as plt
    #%matplotlib inline
    epochs = range(0,10)

    ax = plt.subplot(111)
    ax.plot(adam_loss, label="ADAM")
    ax.plot(sgd_loss, label="SGD")
    ax.plot(rmsprop_loss, label="RMSProp")
    ax.legend()

    plt.xlabel("Epochs")
    plt.ylabel("Overall Loss")
    plt.ylabel("Overall Loss")
    plt.title("Loss across epochs for different optimizers")
    plt.show()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        tch.manual_seed(2020)
        self.fc1 = nn.Linear(64,256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1024)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(1024,1)
        self.final = nn.Sigmoid()

    def forward(self,x):
        op = self.fc1(x)
        op = self.relu1(op)
        op = self.fc2(op)
        op = self.relu2(op)
        op = self.out(op)
        y = self.final(op)
        return y


def train_network(model, optimizer, loss_function, num_epochs, batch_size, X_train, Y_train, lambda_L1 = 0.0):
    
    loss_across_epochs = []
    for epoch in range(num_epochs):
        train_loss = 0.0

        #is in the loop for num_epochs; trained for many times
        model.train()
        
        for i in range(0, X_train.shape[0], batch_size):
            input_data = X_train[i:min(X_train.shape[0], i + batch_size)]
            labels = Y_train[i:min(X_train.shape[0], i + batch_size)]

            #backpropragation related
            optimizer.zero_grad()

            #Forward pass?
            output_data = model(input_data)
            
            loss = loss_function(output_data, labels)
            L1_loss = 0;

            #
            for p in model.parameters():
                L1_loss = L1_loss + p.abs().sum()

            #Add L1 penalty to loss
            loss = loss + lambda_L1 * L1_loss

            # loss backpropogate
            loss.backward()

            #update weigh
            optimizer.step()
            
            #train_loss += loss.item() * batch_size
            train_loss += loss.item() * input_data.size(0)
            

        #loss_across_epochs.extend([train_loss])
        loss_across_epochs.append(train_loss/X_train.size(0))
        if epoch % 500 == 0:
            print("Epoch: {} - Loss:{:.4f}".format(epoch, train_loss/X_train.size(0)))
        
        #predict
#    y_test_pred = model(x_test)
 #   a = np.where(y_test_pred > 0.5, 1, 0)
    return (loss_across_epochs)

#define function for evaluating NN
def evaluate_model(model, x_test, y_test, X_train, Y_train, loss_list):
    model.eval() #Explicitly set to evaluate mode

    #Predict on Train and Validation Datasets
    y_test_prob = model(x_test)
    y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
    Y_train_prob = model(X_train)
    Y_train_pred = np.where(Y_train_prob > 0.5, 1, 0)
    
    #Compute Training and Validation Metrics
    print("\n Model Performance -")
    print("Training Accuracy-", round(accuracy_score(Y_train, Y_train_pred), 3))
    print("Training Precision-", round(precision_score(Y_train, Y_train_pred), 3))
    print("Training Recall-", round(recall_score(Y_train, Y_train_pred),3))
    print("Training ROCAUC", round(roc_auc_score(Y_train, Y_train_prob.detach().numpy()), 3))

    print("Validation accuracy-", round(accuracy_score(y_test, y_test_pred), 3))
    print("Validation Precision-", round(precision_score(y_test, y_test_pred), 3))
    print("Validation Recall-", round(recall_score(y_test, y_test_pred), 3))
    print("Validation ROCAUC", round(roc_auc_score(y_test, y_test_prob.detach().numpy()), 3))
    print("\n")

    #Plot the Loss curve and ROC Curve
    plt.figure(figsize = (20,5))
    plt.subplot(1,2,1)
    plt.plot(loss_list)
    plt.title('Loss across epochs')
    plt.ylabel('Loss')
    plt.subplot(1,2,2)

    #Validation
    fpr_v, tpr_v, _ = roc_curve(y_test, y_test_prob.detach().numpy())
    roc_auc_v = auc(fpr_v, tpr_v)

    #Training
    fpr_t, tpr_t, _ = roc_curve(Y_train, Y_train_prob.detach().numpy())
    roc_auc_t = auc(fpr_t, tpr_t)

    plt.title('Receiver operating characteristic:Validation')
    plt.plot(fpr_v, tpr_v, 'b', label = 'Validation AUC = %0.2f' %roc_auc_v)
    plt.plot(fpr_t, tpr_t, 'r', label = 'Training AUC = %0.2f' % roc_auc_t)
    plt.legend(loc = 'lower right')

    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.show()

    

def load_data():
    df = pd.read_csv("./bank.csv")
    print("DF Shape:", df.shape)
    df.head()

    print("Distribution of Target Values in Dataset -")
    df.deposit.value_counts()

    #check if we have 'na' values within the dataset
    df.isna().sum()

    df.dtypes.value_counts()


    #extract categorical columns from dataset
    categorical_columns = df.select_dtypes(include="object").columns
    print("Categorical cols:",list(categorical_columns))

    #For each categorical column if values in (Yes/no) convert into a 1/0 Flag
    for col in categorical_columns:
        if df[col].nunique() == 2:
            df[col] = np.where(df[col] == "yes", 1, 0)

    df.head()

    #for the remaining categorical variables;
    #create one-hot encoded version of the dataset
    new_df = pd.get_dummies(df)

    #Define target and predictors for the model
    target = "deposit"
    predictors = set(new_df.columns) - set([target])
    print("new_df.shape:",new_df.shape)
    new_df[list(predictors)].head()

    new_df = new_df.astype(np.float32)
    #Split dataset into Train/Test [80:20]
    X_train,x_test, Y_train,y_test = train_test_split(new_df[list(predictors)],new_df[target],test_size= 0.2)
    #Convert Pandas dataframe, first to numpy and then to Torch Tensors
    X_train = tch.from_numpy(X_train.values)
    x_test = tch.from_numpy(x_test.values)
    Y_train = tch.from_numpy(Y_train.values).reshape(-1,1)
    y_test = tch.from_numpy(y_test.values).reshape(-1,1)

    #Print the dataset size to verify
    print("X_train.shape:",X_train.shape)
    print("x_test.shape:",x_test.shape)
    print("Y_train.shape:",Y_train.shape)
    print("y_test.shape:",y_test.shape)


    ##########################################
    #Define training variables
    num_epochs = 500  #L1 Regularization
    batch_size = 128
    loss_function = nn.BCELoss() #Binary Cross Entropy Loss

    #Hyperparameters
    #weight_decay = 0.0 #set to 0; no L2 Regularizer; passed into the Optimizer
    weight_decay = 0.001 #Enable L2 Regularization
    
    #lambda_L1 = 0.0 #set to 0; no L1 reg; manually added in loss(train_network)
    lambda_L1 = 0.0001 # Enable L1 Regularization
    
    #Create a model instance
    model = NeuralNetwork2()

    #define optimizer
    adam_optimizer = tch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = weight_decay)

    #Train model
    adam_loss = train_network(model, adam_optimizer, loss_function, num_epochs,
                              batch_size, X_train, Y_train,
                              lambda_L1 = lambda_L1)

    evaluate_model(model, x_test, y_test, X_train, Y_train, adam_loss)
    

#vanilla neural network
#define Network with Dropout Layers
class NeuralNetwork2(nn.Module):
    #adding dropout layer to reduce overfitting
    def __init__(self):
        super().__init__()
        tch.manual_seed(2020)
        self.fc1 =nn.Linear(48,96)
        self.fc2 = nn.Linear(96, 192)
        self.fc3 = nn.Linear(192,384)
        
        self.relu = nn.ReLU()
        self.out = nn.Linear(384,1)
        self.final = nn.Sigmoid()

        self.drop = nn.Dropout(0.1) #dropout layer

    def forward(self,x):
        op = self.drop(x) #dropout for input layer
        op = self.fc1(op)
        op = self.relu(op)

        op = self.drop(op) #Dropout for hidden layer 1
        op = self.fc2(op)
        op = self.relu(op)

        op = self.drop(op) # dropout for hidden layer 2
        op = self.fc3(op)
        op = self.relu(op)

        op = self.drop(op) #Dropout for hiddent layer 3
        op = self.out(op)
        y = self.final(op)
        
        return y

    

import matplotlib.pyplot as plt
import numpy as np



def toy(degree = 3):
    # 100 values, equidistantly,
    x = np.linspace(-1, 1, 100)
    signal = 2 + x + 2 * x * x
    
    # error
    noise = np.random.normal(0, 0.1, 100)
    y = signal + noise
    
    '''plt.plot(signal, 'b');
    plt.plot(y, 'g');
    plt.plot(noise, 'r')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["Without Noise", "With Noise", "Noise"], loc=2)
    plt.show()
    '''
    
    #extract training from the toy dataset
    x_train = x[0:80]
    y_train = y[0:80]
    print("Shape of x_train:", x_train.shape)
    print("Shape of y_train:", y_train.shape)
    
    create_model(x_train, y_train, degree, x, y)


def create_model(x_train, y_train,degree, x, y):
    degree += 1
    X_train= np.column_stack([np.power(x_train, i) for i in range(0, degree)])
    model = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(), X_train)),
                          X_train.transpose()), y_train)

    plt.plot(x, y, 'g')
    plt.xlabel("x")
    plt.ylabel("y")
    predicted = np.dot(model, [np.power(x,i) for i in range(0, degree)])
    plt.plot(x, predicted, 'r')
    plt.legend(["Actual", "Predicted"], loc = 2)

    plt.title("Model with degree = 3")
    train_rmse1 = np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80],
                                        y_train - predicted[0:80])))

    test_rmse1= np.sqrt(np.sum(np.dot(y[80:] - predicted[80:],
                                       y[80:] - predicted[80:])))

    print("Train RMSE(Degree ="+str(degree)+"):", round(train_rmse1, 2))
    print("Test RMSE (Degree = "+str(degree)+"):", round(test_rmse1,2))
    plt.show()

    
    
def toy_lambda():
    #setting seed for reproducibility
    np.random.seed(20)

    x = np.linspace(-1, 1, 100)
    signal = 2 + x + 2 * x * x
    noise = np.random.normal(0, 0.1, 100)

    y = signal + noise

    x_train = x[0:80]
    y_train = y[0:80]

    train_rmse = []
    test_rmse =[]
    degree = 80

    lambda_reg_values = np.linspace(0.01, 0.99, 100)
    for lambda_reg in lambda_reg_values:
        X_train = np.column_stack([np.power(x_train, i) for i in range(0, degree)])
        model = np.dot(np.dot(np.linalg.inv(np.dot(X_train.transpose(), X_train) + lambda_reg * np.identity(degree)), X_train.transpose()), y_train)

        predicted = np.dot(model, [np.power(x,i) for i in range(0, degree)])

        train_rmse.append(np.sqrt(np.sum(np.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80]))))
        test_rmse.append(np.sqrt(np.sum(np.dot(y[80:] - predicted[80:], y[80:] - predicted[80:]))))

    #Plot the performance over train and test dataset
    plt.plot(lambda_reg_values, train_rmse)
    plt.plot(lambda_reg_values, test_rmse)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("RMSE")
    plt.legend(["Train", "Test"], loc = 2)
    plt.show()
        

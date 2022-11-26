import pandas as pd, numpy as np, os, math

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

SlidingWindowLengthInSeconds = 1 * 60

NN_ARCHITECTURE = [
    {"input_dim": 3, "output_dim": 3, "activation": "sigmoid"},
    {"input_dim": 3, "output_dim": 3, "activation": "sigmoid"},
    {"input_dim": 3, "output_dim": 3, "activation": "sigmoid"},
    {"input_dim": 3, "output_dim": 1, "activation": "tanh"}
]

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def tanh_backward(dA, Z):
    t = tanh(Z)
    return dA * (1 - t**2)

def init_layers(nn_architecture, seed = 99):
    # random seed initiation
    np.random.seed(seed)
    # number of layers in our neural network
    number_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1

        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1

    return params_values

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="sigmoid"):
    # calculation of the input value for the activation function
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    # selection of activation function
    if activation == "tanh":
        activation_func = tanh
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')

    # return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr

        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        # saving calculated values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory

def get_cost_value(Y_hat, Y):
    # Mean Squared Error
    return np.square(np.subtract(Y_hat,Y)).mean()

# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, Z_curr, A_prev, activation="sigmoid"):
    # number of examples
    m = A_prev.shape[1]

    # selection of activation function
    if activation == "tanh":
        backward_activation_func = tanh_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    # calculation of the activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)

    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    # number of examples
    m = Y.shape[1]
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(Y_hat.shape)

    # initiation of gradient descent algorithm
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]

        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, Z_curr, A_prev, activ_function_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values

def updateAdam(params_values, grads_values, nn_architecture, learning_rate):
    # ADAM hacks https://www.youtube.com/watch?v=zUZRUTJbYm8
    m0 = np.zeros(len(params_values))
    v0 = np.zeros(len(params_values))
    t = 0
    beta1 = 0.9
    beta2 = 0.999
    e = 1e-8

    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        t += 1
        gt = grads_values["dW" + str(layer_idx)]
        mt = beta1 * m0[t - 1] + (1 - beta1) * gt
        vt = beta2 * m0[t - 1] + (1 - beta2) * gt**2
        mtHat = m0[t]/(1 - beta1**t)
        vtHat = v0[t]/(1 - beta2**t)

        params_values["W" + str(layer_idx)] -= gt - learning_rate * mtHat/(math.sqrt(vtHat) + e)
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values

def trainAdam(
        X,
        Y,
        nn_architecture = NN_ARCHITECTURE,
        epochs = 1000,
        learning_rate = 0.01,
        accuracy_threshold = 0.97,
        accuracy_count_threshold = 200,
        verbose=False):

    return train(
        X,
        Y,
        updateAdam,
        nn_architecture,
        epochs,
        learning_rate,
        momentum_constant=None,
        accuracy_threshold=accuracy_threshold,
        accuracy_count_threshold=accuracy_count_threshold,
        verbose=verbose)

def updateMomentum(params_values, grads_values, nn_architecture, learning_rate, momentum_constant = 0.9):
    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)] + momentum_constant * params_values["W" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values

def trainMomentum(
        X,
        Y,
        nn_architecture = NN_ARCHITECTURE,
        epochs = 1000,
        learning_rate = 0.01,
        momentum_constant = 0.9,
        accuracy_threshold = 0.97,
        accuracy_count_threshold = 200,
        verbose=False):

    return train(
        X,
        Y,
        updateMomentum,
        nn_architecture,
        epochs,
        learning_rate,
        momentum_constant,
        accuracy_threshold,
        accuracy_count_threshold,
        verbose)

def train(
        X,
        Y,
        updateFunction,
        nn_architecture,
        epochs,
        learning_rate,
        momentum_constant,
        accuracy_threshold,
        accuracy_count_threshold,
        verbose):

    # initiation of neural net parameters
    params_values = init_layers(nn_architecture, 2)
    # initiation of lists storing the history
    # of metrics calculated during the learning process
    cost_history = []
    accuracy_history = []

    sameAccuracyCounter = 0

    # performing calculations for subsequent iterations
    for i in range(epochs):
        # step forward
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)

        # calculating metrics and saving them in history
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        if accuracy == accuracy_history[i - 1]:
            sameAccuracyCounter += 1

        if accuracy > accuracy_threshold or sameAccuracyCounter == accuracy_count_threshold:
            return params_values

        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        # updating model state

        if momentum_constant is None:
            params_values = updateFunction(params_values, grads_values, nn_architecture, learning_rate)
        else:
            params_values = updateFunction(params_values, grads_values, nn_architecture, learning_rate, momentum_constant)

        if(i % 1 == 0):
            if(verbose):
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.20f} - learning rate: {:.3f}".format(i, cost, accuracy, learning_rate))

    return params_values

def normalize(df, columns):
    yMin = -0.9
    yMax = 0.9

    yConstatnt = yMax - yMin

    result = df.copy()
    for column in columns:
        max_value = df[column].max()
        min_value = df[column].min()
        result[column] = yMin + (yConstatnt * ((df[column] - min_value) / (max_value - min_value)))
    return result

if __name__ == '__main__':
    trainingSetFileName = 'trainigSet.csv'
    targetColumnName = 'target'
    networkInput = ['size','frequency','recency']

    if not os.path.isfile(trainingSetFileName):
        df = pd.read_csv('getTileStatistics.csv')
        workingSetDictionary = {}

        for index, row in df.loc[::-1].iterrows():

            tilePosition = row['tile_position']
            currentRequestedTime = row['requested_time']

            if tilePosition in workingSetDictionary:
                workingSetEntry = workingSetDictionary[tilePosition]

                laterRequestedTime = workingSetEntry[0]

                if laterRequestedTime - currentRequestedTime < SlidingWindowLengthInSeconds:
                    df.iloc[workingSetEntry[1], df.columns.get_loc(targetColumnName)] = 1.0

            workingSetDictionary[tilePosition] = [currentRequestedTime, index]

        columnsToWorkOn = networkInput.copy()

        normalizedSet = normalize(df, columnsToWorkOn)

        columnsToWorkOn.append(targetColumnName)

        trainingSet = normalizedSet[columnsToWorkOn]

        trainingSet.to_csv(trainingSetFileName, index=False)

    trainingSet = pd.read_csv(trainingSetFileName)

    X = trainingSet[networkInput].values
    y = trainingSet[targetColumnName].values

    TEST_SIZE = 0.1
    epochs = 1000

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    print("~"*100)
    clf = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=epochs, activation='logistic', solver='lbfgs').fit(X_train, y_train)
    lbfgsParams = clf.get_params()
    print("lbfgs:           {0}".format(clf.score(X_test, y_test)))

    clf = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=epochs, activation='logistic', solver='sgd').fit(X_train, y_train)
    sgdParams = clf.get_params()
    print("sgd:             {0}".format(clf.score(X_test, y_test)))

    clf = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=epochs, activation='logistic', solver='adam').fit(X_train, y_train)
    adamParams = clf.get_params()
    print("adam:            {0}".format(clf.score(X_test, y_test)))

    params_values_adam = trainAdam(
        np.transpose(X_train),
        np.transpose(y_train.reshape((y_train.shape[0], 1))))

    Y_test_hat, customParams = full_forward_propagation(np.transpose(X_test), params_values_adam, NN_ARCHITECTURE)

    acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
    print("custom adam:     {0}".format(acc_test))

    params_values_momentum = trainMomentum(
        np.transpose(X_train),
        np.transpose(y_train.reshape((y_train.shape[0], 1))))

    Y_test_hat, customParams = full_forward_propagation(np.transpose(X_test), params_values_momentum, NN_ARCHITECTURE)

    acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
    print("custom momentum: {0}".format(acc_test))
    print("~"*100)
    print()
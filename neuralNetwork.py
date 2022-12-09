import pandas as pd, numpy as np, os, sqlite3

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

SlidingWindowLengthInSeconds = 1 * 60
DB_NAME = 'tiles.db'

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

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, Z_curr, A_prev, activ_function_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values

# Based on https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
def train(
        X,
        Y,
        method='adam',
        nn_architecture = NN_ARCHITECTURE,
        epochs = 1000,
        learning_rate = 0.001,
        momentum_constant = 0.9,
        accuracy_threshold = 0.97,
        accuracy_count_threshold = 100,
        verbose=False):

    # initiation of neural net parameters
    params_values = init_layers(nn_architecture, 2)
    # initiation of lists storing the history
    # of metrics calculated during the learning process
    cost_history = []
    accuracy_history = []

    sameAccuracyCounter = 0

    # ADAM info https://www.youtube.com/watch?v=zUZRUTJbYm8
    # https://www.youtube.com/watch?v=JXQT_vxqwIs
    if method == 'adam':
        m = {}
        v = {}
        # iteration over network layers
        for idx, layer in enumerate(nn_architecture):
            # we number network layers from 1
            layer_idx = idx + 1

            # extracting the number of units in layers
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            m[layer_idx] = np.zeros(shape=(layer_output_size, layer_input_size))
            v[layer_idx] = np.zeros(shape=(layer_output_size, layer_input_size))

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

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
        else:
            sameAccuracyCounter = 0

        if accuracy > accuracy_threshold or sameAccuracyCounter == accuracy_count_threshold: # converged
            return params_values

        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)

        # updating model state
        if method == 'adam':
            for layer_idx, layer in enumerate(nn_architecture, 1):
                gt = grads_values["dW" + str(layer_idx)]
                m[layer_idx] = beta1 * m[layer_idx] + (1 - beta1) * gt
                v[layer_idx] = beta2 * v[layer_idx] + (1 - beta2) * gt ** 2
                mtHat = m[layer_idx] / (1 - beta1 ** (i + 1))
                vtHat = v[layer_idx] / (1 - beta2 ** (i + 1))

                params_values["W" + str(layer_idx)] -= gt - learning_rate * mtHat / (np.sqrt(vtHat) + epsilon)
                params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

        elif method == 'momentum':            
            for layer_idx, layer in enumerate(nn_architecture, 1):
                params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)] + momentum_constant * params_values["W" + str(layer_idx)]
                params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

        else:
            raise Exception('\'{0}\' training method is not supported'.format(method))

        if(verbose):
            print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.20f} - learning rate: {:.3f}".format(i, cost, accuracy, learning_rate))

    return params_values

def normalize_df(df, columns):
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
    cacheabilityColumnName = 'cacheability'
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
                    df.iloc[workingSetEntry[1], df.columns.get_loc(cacheabilityColumnName)] = 1.0

            workingSetDictionary[tilePosition] = [currentRequestedTime, index]

        columnsToWorkOn = networkInput.copy()

        normalizedSet = normalize_df(df, columnsToWorkOn)

        columnsToWorkOn.append(cacheabilityColumnName)

        trainingSet = normalizedSet[columnsToWorkOn]

        trainingSet.to_csv(trainingSetFileName, index=False)

    trainingSet = pd.read_csv(trainingSetFileName)

    X = trainingSet[networkInput].values
    y = trainingSet[cacheabilityColumnName].values

    TEST_SIZE = 0.1
    epochs = 1000

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    # clf = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=epochs, activation='logistic', solver='lbfgs').fit(X_train, y_train)
    # print("lbfgs:           {0}".format(clf.score(X_test, y_test)))

    # clf = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=epochs, activation='logistic', solver='sgd').fit(X_train, y_train)
    # print("sgd:             {0}".format(clf.score(X_test, y_test)))

    # clf = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=epochs, activation='logistic', solver='adam').fit(X_train, y_train)
    # print("adam:            {0}".format(clf.score(X_test, y_test)))

    params_values_adam = train(
        np.transpose(X_train),
        np.transpose(y_train.reshape((y_train.shape[0], 1))),
        method='adam')

    Y_test_hat, _ = full_forward_propagation(np.transpose(X_test), params_values_adam, NN_ARCHITECTURE)

    acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
    print("custom adam:     {0}".format(acc_test))

    # params_values_momentum = train(
    #     np.transpose(X_train),
    #     np.transpose(y_train.reshape((y_train.shape[0], 1))),
    #     method='momentum')

    # Y_test_hat, _ = full_forward_propagation(np.transpose(X_test), params_values_momentum, NN_ARCHITECTURE)

    # acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
    # print("custom momentum: {0}".format(acc_test))
    # print("~"*100)

    con = sqlite3.connect(DB_NAME)

    result = pd.read_sql_query("SELECT tileMatrix, tileRow, tileCol, size, recency, frequency, cacheability FROM tiles", con)

    con.commit()

    workData = normalize_df(result, networkInput)

    targets, _ = full_forward_propagation(np.transpose(workData[networkInput]), params_values_adam, NN_ARCHITECTURE)

    workData[cacheabilityColumnName] = np.transpose(targets[0])

    rows = workData[['cacheability', 'tileMatrix', 'tileRow', 'tileCol']].values

    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()

    cur.executemany("UPDATE tiles SET cacheability = ? WHERE tileMatrix = ? AND tileRow = ? AND tileCol = ?", rows)

    con.commit()
    cur.close()
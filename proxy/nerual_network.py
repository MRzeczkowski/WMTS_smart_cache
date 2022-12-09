import numpy as np
import copy
from math import sqrt
from math import isnan
from os import linesep


# Based on https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
class Nerual_Network:
    def __init__(
            self,
            params_values=None):

        self._architecture = [
            {"input_dim": 3, "output_dim": 3, "activation": "sigmoid"},
            {"input_dim": 3, "output_dim": 3, "activation": "sigmoid"},
            {"input_dim": 3, "output_dim": 3, "activation": "sigmoid"},
            {"input_dim": 3, "output_dim": 1, "activation": "tanh"}
        ]

        # creating a temporary memory to store the information needed for a backward step
        self._memory = {}

        self.params_values = params_values

        if self.params_values is None:
            self._init_layers()

    def _init_layers(self):
        generator = np.random.default_rng()

        # parameters storage initiation
        self.params_values = {}

        for idx, layer in enumerate(self._architecture):
            layer_idx = idx + 1

            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            # initiating the values of the W matrix
            # and vector b for subsequent layers
            sizes_sum = layer_input_size + layer_output_size
            r = sqrt(6.0 / sizes_sum)

            if layer["activation"] == 'tanh':
                low, high = -r, r
            else:
                r *= 4.0
                low, high = -r, r

            self.params_values['W' + str(layer_idx)] = generator.uniform(
                low, high, (layer_output_size, layer_input_size))

            self.params_values['b' + str(layer_idx)
                               ] = np.zeros((layer_output_size, 1))

    def _single_layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="sigmoid"):
        # calculation of the input value for the activation function
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation == "tanh":
            activation_func = self._tanh
        elif activation == "sigmoid":
            activation_func = self._sigmoid
        else:
            raise Exception('Non-supported activation function')

        # return of calculated activation A and the intermediate Z matrix
        return activation_func(Z_curr), Z_curr

    def full_forward_propagation(self, X):
        # X vector is the activation for layer 0 
        A_curr = X

        for idx, layer in enumerate(self._architecture):
            layer_idx = idx + 1

            # transfer the activation from the previous iteration
            A_prev = A_curr

            # getting activation function, weights and biases for current layer
            activ_function_curr = layer["activation"]
            W_curr = self.params_values["W" + str(layer_idx)]
            b_curr = self.params_values["b" + str(layer_idx)]

            # calculation of activation for the current layer
            A_curr, Z_curr = self._single_layer_forward_propagation(
                A_prev, W_curr, b_curr, activ_function_curr)

            # saving calculated values in the memory, used later for backward propagation
            self._memory["A" + str(idx)] = A_prev
            self._memory["Z" + str(layer_idx)] = Z_curr

        # return of prediction vector
        return A_curr

    def _sigmoid(self, Z):
        return 1.0 / (1.0 + np.exp(-Z))

    def _sigmoid_backward(self, dA, Z):
        sig = self._sigmoid(Z)
        return dA * sig * (1.0 - sig)

    def _tanh(self, Z):
        return np.tanh(Z)

    def _tanh_backward(self, dA, Z):
        t = self._tanh(Z)
        return dA * (1.0 - t ** 2.0)

    # Mean Squared Error
    def _get_cost_value(self, Y_hat, Y):
        return np.square(np.subtract(Y_hat, Y)).mean()

    # an auxiliary function that converts probability into class
    def convert_prob_into_class(self, probs, threshold=0.5):
        probs_ = np.copy(probs)
        probs_[probs_ > threshold] = 1.0
        probs_[probs_ <= threshold] = 0.0
        return probs_

    def get_accuracy_value(self, Y_hat, Y):
        Y_hat_ = self.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()

    def _single_layer_backward_propagation(self, dA_curr, W_curr, Z_curr, A_prev, activation="sigmoid"):
        # number of examples
        m = A_prev.shape[1]

        # selection of activation function
        if activation == "tanh":
            backward_activation_func = self._tanh_backward
        elif activation == "sigmoid":
            backward_activation_func = self._sigmoid_backward
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

    def _full_backward_propagation(self, Y_hat, Y):
        grads_values = {}

        # a hack ensuring the same shape of the prediction vector and labels vector
        Y = Y.reshape(Y_hat.shape)

        # initiation of gradient descent algorithm
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1.0 - Y, 1.0 - Y_hat))

        for layer_idx_prev, layer in reversed(list(enumerate(self._architecture))):
            layer_idx_curr = layer_idx_prev + 1

            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = self._memory["A" + str(layer_idx_prev)]
            Z_curr = self._memory["Z" + str(layer_idx_curr)]

            W_curr = self.params_values["W" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = self._single_layer_backward_propagation(
                dA_curr, W_curr, Z_curr, A_prev, activ_function_curr)

            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values

    def train(
            self,
            X,
            Y,
            epochs=1000,
            learning_rate=0.001,
            method="adam",
            momentum_constant=0.9,
            n_iter_no_change=10):

        trainingRaport = 'Training raport:'

        def append_line_to_report(log):
            nonlocal trainingRaport
            trainingRaport += linesep + log

        tolerance_for_optimization = 1.0e-4

        # ADAM info https://www.youtube.com/watch?v=zUZRUTJbYm8
        # https://www.youtube.com/watch?v=JXQT_vxqwIs
        if method == 'adam':
            means = {}
            variances = {}
            # initializing means and variances for each weight as zeros
            for idx, layer in enumerate(self._architecture):
                layer_idx = idx + 1

                layer_input_size = layer["input_dim"]
                layer_output_size = layer["output_dim"]

                means[layer_idx] = np.zeros(
                    shape=(layer_output_size, layer_input_size))
                variances[layer_idx] = np.zeros(
                    shape=(layer_output_size, layer_input_size))

            # using constants proposed in original paper
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1.0e-8

        elif method == 'sgd':
            changes = {}
            # we save changes made to each layer
            for idx, layer in enumerate(self._architecture):
                layer_idx = idx + 1
                changes[layer_idx] = 0.0

        else:
            raise Exception(
                '\'{0}\' training method is not supported'.format(method))

        # initiation of lists storing the history
        # of metrics calculated during the learning process
        cost_history = []
        accuracy_history = []

        no_change_counter = 0
        best_param_values = copy.deepcopy(self.params_values)
        best_accuracy = 0.0
        reinitialization_count = 0

        # outermost loop used in case we don't converge at all - we will reinitialize the network a few times
        while True:
            # performing calculations for subsequent iterations
            for i in range(epochs):

                # taking a random mini-batch - this is stochastic gradient descent
                batch_indexes = np.random.randint(X.shape[1], size=200)
                X_batch, Y_batch = X[:, batch_indexes], Y[:, batch_indexes]

                Y_hat = self.full_forward_propagation(X_batch)

                # calculating metrics and saving them in history
                cost = self._get_cost_value(Y_hat, Y_batch)
                cost_history.append(cost)
                accuracy = self.get_accuracy_value(Y_hat, Y_batch)
                accuracy_history.append(accuracy)

                append_line_to_report(
                    "Iteration: {:05} - cost: {:.5f} - accuracy: {:.20f}".format(
                        i, cost, accuracy))

                if isnan(cost):
                    append_line_to_report(
                        "Encountered numeric error - ending session")
                    break

                # saving best parameters based on the accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_param_values = copy.deepcopy(self.params_values)

                # checking if we make any progess in cost or accuracy
                if abs(cost_history[i - 1] - cost) < tolerance_for_optimization or abs(accuracy - accuracy_history[i - 1]) < tolerance_for_optimization:
                    no_change_counter += 1
                else:
                    no_change_counter = 0

                # seems like not much progess is made so we'll exit this session, reinitialize and try again
                if no_change_counter == n_iter_no_change:
                    break

                # calculating gradients
                grads_values = self._full_backward_propagation(Y_hat, Y_batch)

                # updating model state
                if method == 'adam':
                    # using `Algorithm 1` from https://arxiv.org/abs/1412.6980
                    for layer_idx, layer in enumerate(self._architecture, 1):
                        gt = grads_values["dW" + str(layer_idx)]

                        means[layer_idx] = beta1 * \
                            means[layer_idx] + (1.0 - beta1) * gt

                        variances[layer_idx] = beta2 * \
                            variances[layer_idx] + (1.0 - beta2) * gt ** 2.0

                        mtHat = means[layer_idx] / (1.0 - beta1 ** (i + 1.0))

                        vtHat = variances[layer_idx] / \
                            (1.0 - beta2 ** (i + 1.0))

                        self.params_values["W" + str(layer_idx)] -= gt - \
                            learning_rate * mtHat / (np.sqrt(vtHat) + epsilon)

                        self.params_values["b" + str(layer_idx)] -= learning_rate * \
                            grads_values["db" + str(layer_idx)]

                elif method == 'sgd':
                    # based on https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/
                    for layer_idx, layer in enumerate(self._architecture, 1):
                        # calculating change based on current gradient and previous change
                        new_change = learning_rate * \
                            grads_values["dW" + str(layer_idx)] + \
                            momentum_constant * changes[layer_idx]

                        # saving current change
                        changes[layer_idx] = new_change

                        self.params_values["W" + str(layer_idx)] -= new_change
                        self.params_values["b" + str(layer_idx)] -= learning_rate * \
                            grads_values["db" + str(layer_idx)]

            # 95% is good enough - if we achive it we terminate training
            if best_accuracy > 0.95:
                break

            # checking if network has been reintialized too many times
            # with current `learing_rate` we may never converge
            if reinitialization_count == 10:
                append_line_to_report(
                    'Didn\'t converge - consider changing `learning_rate` hyperparameter')

                proposed_learning_rate = 0.0
                if method == 'adam':
                    proposed_learning_rate = 0.001

                if method == 'sgd':
                    proposed_learning_rate = 0.05

                append_line_to_report(
                    'For `{:0}` around {:.3f} seems to be a good value'.format(
                        method, proposed_learning_rate))
                break

            # network reintialization
            # sometimes networks initial values are too bad to converge to good accuracy
            # we'll try again with different starting point
            append_line_to_report(
                'Best accuracy so far: {0}'.format(best_accuracy))
            append_line_to_report(
                'Insufficient accuracy, reinitializing parameters and running again')
            self._init_layers()
            reinitialization_count += 1

        # saving best found parameters
        self.params_values = best_param_values
        append_line_to_report(
            'Best accuracy: {0}'.format(best_accuracy))

        if not reinitialization_count == 0:
            append_line_to_report(
                'Reinitialized {0} times'.format(reinitialization_count))

        return trainingRaport

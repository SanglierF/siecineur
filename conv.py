import math
import random
import copy

import numpy as np
SIG = "sig"
RELU = "relu"
MOMENTUM = "momentum"
NESTROVA = "nestrova"
ADAGRAD = "adagard"
ADADELTA = "adadelta"
ADAM = "adam"
XAVIER = "xavier"
HE = "he"
MAX = "max"

class Conv:

    def __init__(self, input_number, hidden_layer_number, output_number, alpha=0.05,
                 weight_random=0.1, hidden_layer_neuron_numbers=None, default_hlayer_neuron_numbers=30, default_act=RELU,
                 activation_function=None, error_threshold=0.15, max_epochs=20, batch_size=100, acc_freeze=3, optimalization=None, opteta=0.7, adambeta1=0.9, adambeta2=0.999, winit=HE, filternumber=8):
        self.input_number = input_number
        self.hidden_layer_number = hidden_layer_number
        self.hidden_layer_neuron_numbers = default_hlayer_neuron_numbers
        self.hl_neuron_numbers = self.initialize_hl_param(hidden_layer_neuron_numbers, self.hidden_layer_number, self.hidden_layer_neuron_numbers)  # if None populate
        self.default_act = default_act
        self.act_types = self.initialize_hl_functions(None, self.hidden_layer_number, self.default_act)

        self.output_number = output_number
        self.alpha = alpha
        self.winit = winit
        self.weight_random = self.weight_random = weight_random
        self.act_func = activation_function
        self.error_threshold = error_threshold
        self.max_epochs = max_epochs
        self.epochs = 0
        self.batch_size = batch_size

        self.acc_freeze = acc_freeze

        self.hidden_layer_weights = []
        self.bias_layer = []

        self.output_layer_weights = None  # if hidden layer then connect to hidden else to first or ignore such case
        self.output_bias = None
        self.softmax_layer = None

        self.net_val_layer = None  # pobudzenie
        self.activation_layer = None  # activation value
        self.errors_layer = None  # for backpropagation


        self.filters = filternumber
        self.kernelsize = 3
        self.kernelstep = 1
        self.poolsize = 2
        self.poolstep = 2
        self.poolcat = MAX

        self.filter_list = []  # add here 32 filters
        self.conv_act = None
        self.filter_biases = []

        self.pool_list = []

        self.input_list = []

        self.initialize_arrays()  # przesuniety z poczatku train aby można było podmieniać


        # numpy.rot90(orignumpyarray,2) - o 180 stopni

    def train(self, training_data, training_labels, valid_data, valid_labels):
        current_epoch = 0
        error_val = 999999999
        error_training = 999999999

        best_arror_val = 99999.99

        best_hidden_layer_weights, best_bias_layer = self.get_weights() #  zapamiętaj najlepsze wagi w tym przypadku poczatkowo wylosowane
        best_hidden_layer_weights = copy.deepcopy(best_hidden_layer_weights)
        best_bias_layer = copy.deepcopy(best_bias_layer)
        acc_stop = 0

        training_errors = []
        val_accuracy = []
        val_errors = []
        stop_reason = "end of training"

        while current_epoch < self.max_epochs and error_training > self.error_threshold:
            """Shuffle training data aby sie nie powtarzala kolejnosc przy nowej epoce albo to omin elo
            wez pierwszy batch wzorow przejdz przez forward prop
            po zakończeniu dla wszystkich wyników wykonaj backprop z zapamietanych wartosci sieci
            oblicz funkcje kosztu
            idz do kolejnego batcha az do konca
            nowa epoka nowy shuffling danych itd
            """

            c = list(zip(training_data, training_labels))

            random.shuffle(c)

            training_data, training_labels = zip(*c)

            batch_index = 0

            batch_training = training_data[batch_index:(batch_index + self.batch_size)]
            batch_labels = training_labels[batch_index:(batch_index + self.batch_size)]

            while batch_index < len(training_data):

                print("nrbatcha" + batch_index.__str__())

                self.input_list = []

                errors_lay = []
                error_lay_bias = []
                all_act_vals = []
                all_net_vals = []

                # przechowywanie danych do back conv
                all_filter_acts = []
                all_filters_errors = []

                for i in range(self.batch_size):
                    act_vals, net_vals, filter_acts, filter_nets, flattenedpool, spindexes = self.forwardpropagation(batch_training[i]) # TODO add act_filters, net_filters?
                    error_layers, error_layers_bias = self.backpropagation(net_vals, self.vectorize_label(batch_labels[i], self.output_number), act_vals)
                    # err = error_layers[len(error_layers)-1]  # Error z warstwy najbliżej wejscia TODO maybe bias?
                    err = error_layers_bias[len(error_layers_bias) - 1]  # Error z warstwy najbliżej wejscia TODO maybe bias?

                    error_conv = self.conv_back(err, flattenedpool, filter_nets, filter_acts, spindexes)
                    all_filters_errors.append(error_conv)

                    errors_lay.append(error_layers)
                    error_lay_bias.append(error_layers_bias)
                    all_act_vals.append(act_vals)
                    all_net_vals.append(net_vals)

                self.change_weights(errors_lay, error_lay_bias, all_act_vals)
                self.conv_change_weights(all_filters_errors)

                batch_index = batch_index + self.batch_size


            # obliczanie kosztu dla training
            """
            cost = 0
            for i in range(len(training_data)):
                act_vals, net_vals = self.forwardpropagation(training_data[i])
                cost = cost + self.categorical_crossentropy(act_vals[len(act_vals) - 1],
                                                            self.vectorize_label(training_labels[i], self.output_number))
            error_training = cost/len(training_data)
            """


            # obliczanie kosztu dla val
            cost = 0
            for i in range(len(valid_data)):
                act_vals, net_vals, t, t1, t2, t3 = self.forwardpropagation(valid_data[i])
                cost = cost + self.categorical_crossentropy(act_vals[len(act_vals) - 1], self.vectorize_label(valid_labels[i], self.output_number))
            error_val = cost/len(valid_data)
            current_epoch += 1
            accuracy_val = self.accuracy(valid_data, valid_labels)
            print(current_epoch)
            print("training error")
            print(error_training)
            print("val error")
            print(error_val)
            print("accuracy val")
            print(accuracy_val)

            print(self.hidden_layer_weights[len(self.hidden_layer_weights)-1][4])

            #check if gradient exploded
            grad = np.sum(self.hidden_layer_weights[len(self.hidden_layer_weights)-1][4])
            array_has_nan = np.isnan(grad)
            if array_has_nan:
                print("Gradient exploded")
                acc_stop = 0
                print(best_hidden_layer_weights[len(self.hidden_layer_weights)-1][4])
                self.set_weights(best_hidden_layer_weights, best_bias_layer)
                stop_reason = "Exploded gradient"
                return training_errors, val_errors, val_accuracy, stop_reason

            training_errors.append(error_training)
            val_accuracy.append(accuracy_val)
            val_errors.append(error_val)

            if error_val < best_arror_val:  # change na
                acc_stop = 0
                best_arror_val = error_val
                best_hidden_layer_weights, best_bias_layer = self.get_weights()
                best_hidden_layer_weights = copy.deepcopy(best_hidden_layer_weights)
                best_bias_layer = copy.deepcopy(best_bias_layer)
            else:
                acc_stop += 1
                print("acc stop value:")
                print(acc_stop)
                if acc_stop > self.acc_freeze:
                    print("Restore weights")
                    acc_stop = 0
                    self.set_weights(best_hidden_layer_weights, best_bias_layer)
                    stop_reason = "early stopping"
                    return training_errors, val_errors, val_accuracy, stop_reason

        print("always restore best weights")
        self.set_weights(best_hidden_layer_weights, best_bias_layer)
        return training_errors, val_errors, val_accuracy, stop_reason

    """
    returns array of neurons in each hidden layer
    """
    @staticmethod
    def initialize_hl_param(hlayer_neuron_numbers, hidden_layer_number, hidden_layer_neuron_numbers=20):
        if hlayer_neuron_numbers is None:
            hidden_layer_neuron_numbers = np.full(hidden_layer_number, hidden_layer_neuron_numbers)
        else:
            hidden_layer_neuron_numbers = hlayer_neuron_numbers
        return hidden_layer_neuron_numbers

    """returns function types for each layer"""
    @staticmethod
    def initialize_hl_functions(acttypes, hidden_layer_number, default_act=RELU):
        act_types = []
        if acttypes is None:
            for i in range(hidden_layer_number):
                act_types.append(default_act)
        else:
            act_types = acttypes
        return act_types

    """returns vectorized label 1 in class"""
    @staticmethod
    def vectorize_label(label, classes):
        yzad = np.zeros((classes, 1))
        yzad[label] = 1.0
        return yzad

    """returns tuple of 2 hidden layer weights, hidden bias"""
    def get_weights(self):
        return self.hidden_layer_weights, self.bias_layer

    def set_weights(self, hidden_layer_weights, bias_layer):
        self.hidden_layer_weights = hidden_layer_weights
        self.bias_layer = bias_layer

    def check(self):
        pass

    def accuracy(self, test_inputs, test_labels):
        correct = 0.0
        for i in range(len(test_inputs)):
            yvect, net_val, t, t1, t2, t3 = self.forwardpropagation(test_inputs[i])
            label = np.argmax(yvect[len(yvect)-1], axis=0)  # index of max value
            if label == test_labels[i]:
                correct = correct + 1
        return correct/len(test_inputs)

    def initialize_arrays(self):

        # add conv layer weights
        for i in range(self.filters):
            weight_var_conv = 2.0 / self.kernelsize
            filter_weights = np.random.normal(0, weight_var_conv, size=(self.kernelsize, self.kernelsize))
            bias = np.random.normal(0, weight_var_conv, size=(1, 1))
            self.filter_list.append(filter_weights)
            self.filter_biases.append(bias)

        # add flatten to hidden weights
        flatpoollayer = 13 * 13 * self.filters
        if self.winit is XAVIER:
            weight_var = 2.0/(flatpoollayer + self.hl_neuron_numbers[0]) # or sqrt(6/xin+xout)
        elif self.winit is HE:
            weight_var = 2.0/flatpoollayer # or sqrt(6/xin)
        else:
            weight_var = self.weight_random
        layer = np.random.normal(0, weight_var, size=(self.hl_neuron_numbers[0], flatpoollayer))
        bias = np.random.normal(0, weight_var, size=(self.hl_neuron_numbers[0], 1))
        self.hidden_layer_weights.append(layer)
        self.bias_layer.append(bias)

        # add hidden to hidden
        for i in range(1, self.hidden_layer_number):
            if self.winit is XAVIER:
                weight_var = 2.0/(self.hl_neuron_numbers[i] + self.hl_neuron_numbers[i-1])
            elif self.winit is HE:
                weight_var = 2.0/self.hl_neuron_numbers[i-1]
            else:
                weight_var = self.weight_random
            layer = np.random.normal(0, weight_var, size=(self.hl_neuron_numbers[i], self.hl_neuron_numbers[i-1]))  # ile tam czegoś
            bias = np.random.normal(0, weight_var, size=(self.hl_neuron_numbers[i], 1))
            self.hidden_layer_weights.append(layer)
            self.bias_layer.append(bias)

        # add hidden to output
        if self.winit is XAVIER:
            weight_var = 2.0/(self.output_number + self.hl_neuron_numbers[self.hidden_layer_number-1])
        elif self.winit is HE:
            weight_var = 2.0/self.hl_neuron_numbers[self.hidden_layer_number-1]
        else:
            weight_var = self.weight_random
        self.output_layer_weights = np.random.normal(0, weight_var,
                                                     size=(self.output_number, self.hl_neuron_numbers[self.hidden_layer_number-1]))  # ile tam czegoś
        self.hidden_layer_weights.append(self.output_layer_weights)
        self.output_bias = np.random.normal(0, weight_var, size=(self.output_number, 1))
        self.bias_layer.append(self.output_bias)

    def conv_layers(self, training_input):

        matrix = training_input.reshape(28, 28)  # reshape do macierzy
        self.input_list.append(matrix)
        conv_layer_neurons = 26 * 26  # liczba neuronów w jednej warstwie conv
        net_val_convs = []
        act_val_convs = []
        # oblicz pobudzenie oraz aktywacje dla convolucji
        for i in range(self.filters):
            net_val = np.zeros(shape=(26, 26))
            act_val = np.zeros(shape=(26, 26))
            for x in range(28-2):
                for y in range(28-2):
                    net_val[x][y] = np.sum(np.multiply(self.filter_list[i], matrix[x:(x + 3), y: (y + 3)]))
                    act_val[x][y] = self.relu(net_val[x][y])

            net_val_convs.append(net_val)
            act_val_convs.append(act_val)

        # max pool
        pools = []
        poolsindexes = []
        for i in range(self.filters):
            singlepool = np.zeros(shape=(13, 13))
            singlepoolsindexes = np.zeros(shape=(13, 13))
            for x in range(13):
                for y in range(13):
                    singlepool[x][y] = np.amax(act_val_convs[i][(x*2): ((x*2)+2), (y*2): ((y*2)+2)])
                    singlepoolsindexes[x][y] = np.argmax(act_val_convs[i][(x*2): ((x*2)+2), (y*2): ((y*2)+2)])
            pools.append(singlepool)
            poolsindexes.append(singlepoolsindexes)

        #flatten
        flattenedpool = pools[0].flatten()
        flattenedpoolnumber = self.filters*13*13
        for i in range(1, self.filters):
            flattenedpool = np.concatenate([flattenedpool, pools[i].flatten()])
        flattenedpool = np.reshape(flattenedpool, (flattenedpoolnumber,  1))
        return net_val_convs, act_val_convs, flattenedpool, poolsindexes  # return conv activations and net_vals

    def conv_back_trick(self, input_error, flattenedpool, net_vals, act_vals):
        errors = np.transpose(input_error)
        flatnets = net_vals[0].flatten()
        for i in range(1, self.filters):
            flatnets = np.concatenate([flatnets, net_vals[i].flatten()])
        #flattenacts for trick
        flatacts = act_vals[0].flatten()
        for i in range(1, self.filters):
            flatacts = np.concatenate([flatacts, act_vals[i].flatten()])

        convgrad = np.zeros(shape=(26*26*8, 50))
        i = 0
        while i in range(26*26*8):
            if flatacts[i] == flattenedpool[i//4]: # no division
                convgrad[i] = errors[i]
                i = i//4 + 4
            else:
                i = i+1

        for i in range(self.filters):
            convgrad[i] = convgrad[i] * self.relu_der(flatnets[i])

        converrors = np.split(convgrad, self.filters)
        for i in range(self.filters):
            # converrors[i] = np.transpose(converrors[i])
            converrors[i] = converrors[i].reshape(26, 26, 50)

        return converrors


    def conv_back(self, input_error, flattenedpool, net_vals, act_vals, spindexes):

        # convgradient
        input_err = np.reshape(input_error, newshape=(50, ))

        conv_gradients2 = []
        for i in range(self.filters):
            conv_grad = np.zeros(shape=(26, 26, 50))
            for x in range(13):
                for y in range(13):
                    index = spindexes[i][x][y]
                    x1 = int(index % 2)
                    y1 = 0
                    if index > 1:
                        y1 = 1
                    conv_grad[(x*2 + x1)][(y*2 + y1)] = input_err

            conv_grad2 = np.matmul(np.transpose(conv_grad), self.activation_function_der(net_vals[i]))
            conv_grad2 = np.transpose(conv_grad2)
            conv_grad2 = np.sum(conv_grad2, axis=2)
            conv_gradients2.append(conv_grad2)
        """
            #exp flatten
            cgrad = conv_grad.reshape(26*26, 50)
            nval = np.reshape(net_vals[i].flatten(), newshape=(26*26, 1))
            conv_grad2 = np.dot(cgrad, nval)
            #conv_grad2 = np.multiply(np.transpose(conv_grad, (2, 0, 1)), self.activation_function_der(net_vals[i]))

            conv_grad2 = conv_grad2.reshape(26, 26)

            conv_gradients2.append(conv_grad2)  # transpose for usefulness
            """



        return conv_gradients2


    def conv_back2(self, input_error, flattenedpool, net_vals, act_vals, spindexes):
        # returned to
        poolerrors = np.hsplit(input_error, self.filters)
        filteerrors = []
        for i in range(self.filters):
            filteerrors.append(np.transpose(poolerrors[i].reshape(50, 13, 13))) # TODO maybe delete transpose idk\

        # convgradient
        conv_gradients = []
        conv_gradients2 = []
        for i in range(self.filters):
            conv_gradients.append(np.zeros(shape=(26, 26)))

        for i in range(self.filters):
            conv_grad = np.zeros(shape=(26, 26, 50))
            for x in range(13):
                for y in range(13):
                    index = spindexes[i][x][y]
                    x1 = int(index % 2)
                    y1 = 0
                    if index > 1:
                        y1 = 1
                    conv_grad[(x*2 + x1)][(y*2 + y1)] = filteerrors[i][x][y]

            conv_grad2 = np.matmul(np.transpose(conv_grad), self.activation_function_der(net_vals[i]))
            conv_gradients2.append(np.transpose(conv_grad2))  # transpose for usefulness


        return conv_gradients2



    def conv_change_weights(self, errors):

        alphadiv = self.alpha / self.batch_size

        error_sums = []
        for i in range(self.filters):
            error_sums.append(np.zeros(shape=(3, 3)))


        for i in range(len(errors)): # calculating weight change
            for j in range(self.filters):
                for x in range(28-2):
                    for y in range(28-2):
                        error_sums[j] += np.sum(errors[i][j][x][y] * self.input_list[i][x: (x + 3), y: (y + 3)])  # iterating window for filter weight changes

        for i in range(self.filters): # TODO calculate weight change
            self.filter_list[i] = self.filter_list[i] - (alphadiv * error_sums[i])


    def forwardpropagation(self, training_input):  # return output column with label chances
        activation_layer = []
        net_val_layer = []

        convnets, convact, flattenedpool, spindexes = self.conv_layers(training_input)

        activation_layer.append(flattenedpool)
        net_val_layer.append(flattenedpool)

        for i in range(len(self.hidden_layer_weights)-1):
            lval = self.net_val(self.hidden_layer_weights[i], self.bias_layer[i], activation_layer[i]) # zwraca obliczone pobudzenie dla każdej warstwy
            net_val_layer.append(lval)
            lact = self.activation_function(lval, self.act_types[i])
            activation_layer.append(lact)
        oval = self.net_val(self.hidden_layer_weights[len(self.hidden_layer_weights)-1], self.bias_layer[len(self.bias_layer)-1], activation_layer[len(activation_layer) - 1])
        net_val_layer.append(oval)
        oact = self.softmax(oval)
        activation_layer.append(oact)
        return activation_layer, net_val_layer, convact, convnets, flattenedpool, spindexes

    """param net_val net_values, yzad_vector
        zwraca warsty bledu, kolumna bledow dla biasiow, macierz bledow dla reszty wag
        index 0 = ostatnia warstwa - wyjscia
        index len - pierwsza warstwa
    """

    def backpropagation(self, net_val, yzad_vector, act_vals):  # label musi być wektorem wyjść z mnist gita  TODO zwraca mi dla warstw połączonej aż do flattenedpool i jego biasów same błędy na change weights dodać metoda
        error_layers = []
        error_layers_bias = []
        layers = len(net_val)

        error_last = self.error_last(act_vals[layers-1], yzad_vector)
        error_layers_bias.append(error_last)

        error = np.dot(error_last, act_vals[layers-2].transpose())
        error_layers.append(error)

        for i in range(2, layers):
            z = net_val[-i]
            der = self.activation_function_der(z, self.act_types[-(i-1)])
            error = np.dot(self.hidden_layer_weights[-i+1].transpose(), error_layers_bias[i-2]) * der
            error_layers_bias.append(error)
            error = np.dot(error, act_vals[-i-1].transpose()) # jesli to warstwa najblizej wejscia to aktywacje to wektor wejsciowy
            error_layers.append(error)

        return error_layers, error_layers_bias  # ERRORY od najnizszej warstwy do najblizej wejscia

    def change_weights(self, error_layers, error_layers_bias, activation_layers):

        alphadiv = self.alpha / self.batch_size

        bias_error_sum = [np.zeros(b.shape) for b in self.bias_layer]
        weights_error_sum = [np.zeros(w.shape) for w in self.hidden_layer_weights]

        # sum errors
        for i in range(len(error_layers)):
            for j in range(len(weights_error_sum)):
                weights_error_sum[j] += error_layers[i][len(error_layers[0])-1-j]
                bias_error_sum[j] += error_layers_bias[i][len(error_layers_bias[0])-1-j]

        # calculate delta w
        for i in range(len(weights_error_sum)):
            weights_error_sum[i] = alphadiv * weights_error_sum[i]
            bias_error_sum[i] = alphadiv * bias_error_sum[i]


        # update weights
        for i in range(len(self.hidden_layer_weights)):
            self.hidden_layer_weights[i] = self.hidden_layer_weights[i] - weights_error_sum[i]
            self.bias_layer[i] = self.bias_layer[i] - bias_error_sum[i]

    def net_val(self, input_layer_weights, bias, activation_layer):
        lval = np.dot(input_layer_weights, activation_layer)
        lval = np.add(lval, bias)
        return lval

    def activation_function(self, lval, act_type='relu'):
        if act_type is RELU:
            vact = np.vectorize(self.relu)
        elif act_type is SIG:
            vact = np.vectorize(self.sigmoid)
        else:
            vact = np.vectorize(self.relu)
        return vact(lval)

    def activation_function_der(self, lval, act_type='relu'):
        if act_type is RELU:
            vact = np.vectorize(self.relu_der)
        elif act_type is SIG:
            vact = np.vectorize(self.sigmoid_der)
        else:
            vact = np.vectorize(self.relu_der)
        return vact(lval)

    def sigmoid(self, x):
        return 1 / (1 + np.math.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        act = 0.0
        if x > 0.0:
            act = x
        return act

    def relu_der(self, x):
        act = 0.0
        if x > 0.0:
            act = 1.0
        return act

    def tanh(self, x):
        return 2 / (1 + np.math.exp(-1 * (2 * x))) - 1

    def tanh_der(self, x):
        return 1 - self.tanh(x)**2

    def categorical_crossentropy(self, ypred_vector, yzad_vector):
        return -1 * np.sum(np.nan_to_num(yzad_vector * np.log(ypred_vector)))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def softmax_der(self, softmax_output_vector, index):  # Niepotrzebne przy używaniu crossentropy
        res = 0.0
        for j in range(softmax_output_vector):  # przerobic na macierzowe
            if j is index:
                res += softmax_output_vector[index]*(1.0 - softmax_output_vector[j])
            else:
                res += -1.0 * softmax_output_vector[index] * softmax_output_vector[j]
        return res

    def error_last(self, output_layer_act, yzad):
        return np.subtract(output_layer_act, yzad)

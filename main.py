from datetime import datetime, timedelta
import time

import perceptrons as per
import numpy as np
import generator as ger
import csv
import mnist_loader as ml # https://github.com/mnielsen/neural-networks-and-deep-learning <- data and mnist loader for python3
import mulitlayerperceptron as mlp
import conv as cnv
import copy


def main():
    raport4_2()
    raport4()
    exit()


def printresults(name, data):
    path_to_results = "results/"
    file_format = ".csv"
    file_name = path_to_results + name + file_format

    with open(file_name, "w") as file:
        for line in data:
            file.write(str(line))
            file.write("\n")


def printresults2(name, training_errors, val_errors, val_accuracy, stop_reason, accuracy):
    path_to_results = "results/"
    file_format = ".csv"
    file_name = path_to_results + name + file_format

    with open(file_name, "w") as file:
        file.write("training_errors, val_errors, val_accuracy")
        file.write("\n")
        for i in range(len(training_errors)):
            file.write((str(training_errors[i]) + "," + str(val_errors[i]) + "," + str(val_accuracy[i])))
            file.write("\n")
        file.write("\n")
        file.write("stop reason = " + stop_reason + "," + "accuracy = " + str(accuracy))


# Zadanie1
def thresholdrange():

    # Generate BINARNY data
    x, y = ger.generateand(100)  # + 4 podstawowe wzory
    training_vectors_inputs_binary = x
    training_vectors_outputs_binary = y

    print(training_vectors_inputs_binary)
    print(training_vectors_outputs_binary)

    # Generate BIPOLAR data
    error_threshold = 0.3  # max = 1
    x, y = ger.generateandbipolar(100, error_threshold) # + 4 podstawowe wzory
    training_vectors_inputs_bipolar = x
    training_vectors_outputs_bipolar = y

    # print(training_vectors_inputs_binary)
    # print(training_vectors_outputs_binary)

    # INITIALIZE EMPTY LISTS
    results10, results09, results07, results05, results03, results01 = ([] for i in range(6))

    for i in range(10):
        print("ADALINE")
        perceptron = per.Perceptron(alpha=0.1, initial_weights_dist=0.3, threshold=1.0)
        perceptron.train(training_vectors_inputs_binary, training_vectors_outputs_binary)
        print("epoki:", end=' ')
        print(perceptron.show_epochs())

        print("test")
        print(perceptron.test(np.array([0, 0]), 0), end=' ')
        print(perceptron.test(np.array([1, 0]), 0), end=' ')
        print(perceptron.test(np.array([0, 1]), 0), end=' ')
        print(perceptron.test(np.array([1, 1]), 1), end=' ')
        print(perceptron.test(np.array([0.71, 0.71]), 1))
        results10.append(perceptron.show_epochs())

    for i in range(10):
        print("BINARY PERCEPTRON")
        perceptron = per.Perceptron(alpha=0.1, initial_weights_dist=0.3, threshold=0.7)
        perceptron.train(training_vectors_inputs_binary, training_vectors_outputs_binary)
        print("epoki:", end=' ')
        print(perceptron.show_epochs())

        print("test")
        print(perceptron.test(np.array([0, 0]), 0), end=' ')
        print(perceptron.test(np.array([1, 0]), 0), end=' ')
        print(perceptron.test(np.array([0, 1]), 0), end=' ')
        print(perceptron.test(np.array([1, 1]), 1), end=' ')
        print(perceptron.test(np.array([0.71, 0.71]), 1))
        results09.append(perceptron.show_epochs())

    for i in range(10):
        print("BINARY PERCEPTRON")
        perceptron = per.Perceptron(alpha=0.1, initial_weights_dist=0.3, threshold=0.3)
        perceptron.train(training_vectors_inputs_binary, training_vectors_outputs_binary)
        print("epoki:", end=' ')
        print(perceptron.show_epochs())

        print("test")
        print(perceptron.test(np.array([0, 0]), 0), end=' ')
        print(perceptron.test(np.array([1, 0]), 0), end=' ')
        print(perceptron.test(np.array([0, 1]), 0), end=' ')
        print(perceptron.test(np.array([1, 1]), 1), end=' ')
        print(perceptron.test(np.array([0.71, 0.71]), 1))
        results07.append(perceptron.show_epochs())

    for i in range(10):
        print("BINARY PERCEPTRON")
        perceptron = per.Perceptron(alpha=0.1, initial_weights_dist=0.3, threshold=0.1)
        perceptron.train(training_vectors_inputs_binary, training_vectors_outputs_binary)
        print("epoki:", end=' ')
        print(perceptron.show_epochs())

        print("test")
        print(perceptron.test(np.array([0, 0]), 0), end=' ')
        print(perceptron.test(np.array([1, 0]), 0), end=' ')
        print(perceptron.test(np.array([0, 1]), 0), end=' ')
        print(perceptron.test(np.array([1, 1]), 1), end=' ')
        print(perceptron.test(np.array([0.71, 0.71]), 1))
        results05.append(perceptron.show_epochs())

    print(results07)

    printresults("1-1-d10", results10)
    printresults("1-1-d07", results09)
    printresults("1-1-d03", results07)
    printresults("1-1-d01", results05)



    """
    print("BINARY PERCEPTRON")
    perceptron = per.Perceptron(alpha=0.1, activation=per.BINARY, threshold=1.0, initial_weights_dist=0.9)
    perceptron.train(training_vectors_inputs_binary, training_vectors_outputs_binary)
    print("epoki:", end=' ')
    print(perceptron.show_epochs())

    print("test")
    print(perceptron.test(np.array([0, 0]), 0), end=' ')
    print(perceptron.test(np.array([1, 0]), 0), end=' ')
    print(perceptron.test(np.array([0, 1]), 0), end=' ')
    print(perceptron.test(np.array([1, 1]), 1), end=' ')
    print(perceptron.test(np.array([0.71, 0.71]), 1))
    results09.append(perceptron.show_epochs())
    
    
    
    print("BIPOLAR PERCEPTRON")
    perceptron = per.Perceptron(alpha=0.01, activation=per.BIPOLAR, bias=0.5, initial_weights_dist=0.8)
    perceptron.train(training_vectors_inputs_bipolar, training_vectors_outputs_bipolar)
    print("epoki:", end=' ')
    print(perceptron.show_epochs())

    print("test")
    print(perceptron.test(np.array([-1, -1]), -1), end=' ')
    print(perceptron.test(np.array([1, -1]), -1), end=' ')
    print(perceptron.test(np.array([-1, 1]), -1), end=' ')
    print(perceptron.test(np.array([1, 1]), 1), end=' ')
    print(perceptron.test(np.array([0.71, 0.71]), 1))


    
    # ADALINE z biasem jak wyzej

    # training_vectors_inputs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1], [0.7, 0.7], [0.86, 0.86]])
    # training_vectors_outputs = np.array([[-1], [-1], [-1], [1], [1], [1]])
    print("ADALINE")
    perceptron = per.Adaline(alpha=0.0005, err_threshold=0.3, initial_weights_dist=0.2)
    perceptron.train(training_vectors_inputs, training_vectors_outputs)
    print("epoki:", end=' ')
    print(perceptron.show_epochs())
    print("test")
    print(perceptron.test(np.array([-1, -1]), -1), end=' ')
    print(perceptron.test(np.array([1, -1]), -1), end=' ')
    print(perceptron.test(np.array([-1, 1]), -1), end=' ')
    print(perceptron.test(np.array([1, 1]), 1), end=' ')
    print(perceptron.test(np.array([0.71, 0.71]), 1))
    """

def singletest():
    print("single_test")
    training_data, validation_data, test_data = ml.load_data() # inputs - 784, ouptuts - 10
    neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, max_epochs=50, acc_freeze=9, default_hlayer_neuron_numbers=50, batch_size=100, optimalization=mlp.ADAM, opteta=0.9, default_act=mlp.SIG, winit=mlp.XAVIER)
    #  neural_network.train(training_data[0], training_data[1], validation_data[0], validation_data[1])
    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    t0 = time.clock()
    print("start_time")
    print(start_time)
    neural_network.train(training_data[0], training_data[1], validation_data[0], validation_data[1])
    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    t1 = time.clock()
    print("end_time")
    print(end_time)
    elapsed = t1 - t0
    print("elapsed time")
    print(str(timedelta(seconds=elapsed)))

    print("celnosc: ")
    print(neural_network.accuracy(test_data[0], test_data[1]))

def raport3_1():
    print("raport3_1")
    repeat_times = 3
    training_data, validation_data, test_data = ml.load_data()  # inputs - 784, ouptuts - 10

    neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=30, acc_freeze=14,
                                      default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG)
    hidden_layer_weights_r, bias_layer_r = neural_network.get_weights()  # zapamietaj poczatkowe wagi
    hidden_layer_weights = copy.deepcopy(hidden_layer_weights_r)
    bias_layer = copy.deepcopy(bias_layer_r)

    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    t0 = time.clock()
    print("start_time")
    print(start_time)

    for i in range(repeat_times):
        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=30, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG, optimalization=mlp.MOMENTUM)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "momentumclassic-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=30, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG, optimalization=mlp.NESTROVA)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "momentumnesterowa-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=30, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG, optimalization=mlp.ADAGRAD)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "AdaGrad-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=30, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG, optimalization=mlp.ADADELTA, opteta=0.9) #opteta to tutaj rho czyli decay rate
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "adadelta-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=30, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG, optimalization=mlp.ADAM)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "adam-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    t1 = time.clock()
    print("end_time")
    print(end_time)
    elapsed = t1 - t0
    print("elapsed time")
    print(str(timedelta(seconds=elapsed)))

def raport3_2():
    print("raport3_2")
    repeat_times = 2
    training_data, validation_data, test_data = ml.load_data()  # inputs - 784, ouptuts - 10


    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    t0 = time.clock()
    print("start_time")
    print(start_time)

    for i in range(repeat_times):
        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=50, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG,
                                          winit=mlp.XAVIER)
        hidden_layer_weights_r, bias_layer_r = neural_network.get_weights()  # zapamietaj poczatkowe wagi
        hidden_layer_weights_r = copy.deepcopy(hidden_layer_weights_r)
        bias_layer_r = copy.deepcopy(bias_layer_r)
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "xaviersigmo-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=50, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.RELU,
                                          winit=mlp.XAVIER)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "xavierrelu-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=50, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG,
                                          winit=mlp.HE)
        hidden_layer_weights_r, bias_layer_r = neural_network.get_weights()  # zapamietaj poczatkowe wagi
        hidden_layer_weights_r = copy.deepcopy(hidden_layer_weights_r)
        bias_layer_r = copy.deepcopy(bias_layer_r)
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "hesigmoid-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=50, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.RELU,
                                          winit=mlp.HE)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "herelu-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)


    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    t1 = time.clock()
    print("end_time")
    print(end_time)
    elapsed = t1 - t0
    print("elapsed time")
    print(str(timedelta(seconds=elapsed)))

def raport4():
    print("raport4")
    training_data, validation_data, test_data = ml.load_data()  # inputs - 784, ouptuts - 10
    neural_network = cnv.Conv(784, 1, 10, alpha=0.005, max_epochs=20, acc_freeze=14,
                                      default_hlayer_neuron_numbers=50, batch_size=100, winit=cnv.XAVIER, activation_function=cnv.SIG, optimalization=cnv.ADAM)
    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    t0 = time.clock()
    print("start_time")
    print(start_time)
    training_data = training_data[0][0: 1000], training_data[1][0:1000]
    validation_data = validation_data[0][0: 100], validation_data[1][0:100]
    training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0], training_data[1], validation_data[0], validation_data[1])
    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    t1 = time.clock()
    print("end_time")
    print(end_time)
    elapsed = t1 - t0
    print("elapsed time")
    print(str(timedelta(seconds=elapsed)))

    print("celnosc: ")
    # accuracy = neural_network.accuracy(test_data[0], test_data[1])
    accuracy = 0
    print(accuracy)

    printresults2("conv-test", training_errors, val_errors, val_accuracy, stop_reason, accuracy)

def raport4_2():
    print("raport4_2")
    repeat_times = 3
    training_data, validation_data, test_data = ml.load_data()  # inputs - 784, ouptuts - 10

    neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.005, max_epochs=20, acc_freeze=14,
                                      default_hlayer_neuron_numbers=50, batch_size=100, winit=mlp.XAVIER,
                                      activation_function=mlp.SIG, optimalization=mlp.ADAM)
    hidden_layer_weights_r, bias_layer_r = neural_network.get_weights()  # zapamiętaj najlepsze wagi w tym przypadku poczatkowo wylosowane
    hidden_layer_weights = copy.deepcopy(hidden_layer_weights_r)
    bias_layer = copy.deepcopy(bias_layer_r)

    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    t0 = time.clock()
    print("start_time")
    print(start_time)

    training_data = training_data[0][0: 1000], training_data[1][0:1000]
    validation_data = validation_data[0][0: 100], validation_data[1][0:100]

    for i in range(repeat_times):
        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.005, max_epochs=20, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, winit=mlp.XAVIER,
                                          activation_function=mlp.SIG, optimalization=mlp.ADAM)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        # accuracy = neural_network.accuracy(test_data[0], test_data[1])
        accuracy = 0
        tname = "conv-mlp-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)


    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    t1 = time.clock()
    print("end_time")
    print(end_time)
    elapsed = t1 - t0
    print("elapsed time")
    print(str(timedelta(seconds=elapsed)))

def raport2_1():
    repeat_times = 3
    training_data, validation_data, test_data = ml.load_data()  # inputs - 784, ouptuts - 10

    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    t0 = time.clock()
    print("start_time")
    print(start_time)

    neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.3, max_epochs=50, acc_freeze=14,
                                      default_hlayer_neuron_numbers=300, batch_size=100, default_act=mlp.SIG)
    hidden_layer_weights_r, bias_layer_r = neural_network.get_weights()  # zapamiętaj najlepsze wagi w tym przypadku poczatkowo wylosowane
    hidden_layer_weights = copy.deepcopy(hidden_layer_weights_r)
    bias_layer = copy.deepcopy(bias_layer_r)


    # pierwszy parametr
    for i in range(repeat_times):
        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.3, max_epochs=50, acc_freeze=14,
                                          default_hlayer_neuron_numbers=1200, batch_size=100, default_act=mlp.SIG)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0], training_data[1], validation_data[0], validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "300neurons-" + str(i+1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    t1 = time.clock()
    print("end_time")
    print(end_time)
    elapsed = t1 - t0
    print("elapsed time")
    print(str(timedelta(seconds=elapsed)))


def raport2_2():
    print("raport2_2")
    repeat_times = 3
    training_data, validation_data, test_data = ml.load_data()  # inputs - 784, ouptuts - 10

    neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.01, max_epochs=50, acc_freeze=14,
                                      default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG)
    hidden_layer_weights_r, bias_layer_r = neural_network.get_weights()  # zapamiętaj najlepsze wagi w tym przypadku poczatkowo wylosowane
    hidden_layer_weights = copy.deepcopy(hidden_layer_weights_r)
    bias_layer = copy.deepcopy(bias_layer_r)

    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    t0 = time.clock()
    print("start_time")
    print(start_time)

    for i in range(repeat_times):

        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.001, weight_random=0.01, max_epochs=10, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "0001alpha-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.01, weight_random=0.01, max_epochs=50, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "001alpha-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.01, max_epochs=50, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "01alpha-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=1.0, weight_random=0.01, max_epochs=50, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "10alpha-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)


    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    t1 = time.clock()
    print("end_time")
    print(end_time)
    elapsed = t1 - t0
    print("elapsed time")
    print(str(timedelta(seconds=elapsed)))

def raport2_3():
    print("raport2_3")
    repeat_times = 3
    training_data, validation_data, test_data = ml.load_data()  # inputs - 784, ouptuts - 10

    neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=50, acc_freeze=14,
                                      default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG)
    hidden_layer_weights_r, bias_layer_r = neural_network.get_weights()  # zapamiętaj najlepsze wagi w tym przypadku poczatkowo wylosowane
    hidden_layer_weights = copy.deepcopy(hidden_layer_weights_r)
    bias_layer = copy.deepcopy(bias_layer_r)

    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    t0 = time.clock()
    print("start_time")
    print(start_time)

    for i in range(repeat_times):
        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=50, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.SIG)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "sig-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

        neural_network = mlp.Mlperceptron(784, 1, 10, alpha=0.1, weight_random=0.1, max_epochs=50, acc_freeze=14,
                                          default_hlayer_neuron_numbers=50, batch_size=100, default_act=mlp.RELU)
        neural_network.set_weights(copy.deepcopy(hidden_layer_weights_r), copy.deepcopy(bias_layer_r))
        training_errors, val_errors, val_accuracy, stop_reason = neural_network.train(training_data[0],
                                                                                      training_data[1],
                                                                                      validation_data[0],
                                                                                      validation_data[1])
        accuracy = neural_network.accuracy(test_data[0], test_data[1])
        tname = "relu-" + str(i + 1)
        printresults2(tname, training_errors, val_errors, val_accuracy, stop_reason, accuracy)

    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    t1 = time.clock()
    print("end_time")
    print(end_time)
    elapsed = t1 - t0
    print("elapsed time")
    print(str(timedelta(seconds=elapsed)))



if __name__ == "__main__":
    main()

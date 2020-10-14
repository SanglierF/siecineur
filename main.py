import perceptrons as per
import numpy as np
import generator as ger
import csv

def main():
    thresholdrange()


def printresults(name, data):
    path_to_results = "results/"
    file_format = ".csv"
    file_name = path_to_results + name + file_format

    with open(file_name, "w") as file:
        for line in data:
            file.write(str(line))
            file.write("\n")


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

    exit()


if __name__ == "__main__":
    main()

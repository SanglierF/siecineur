import numpy as np
BINARY = "binary"
BIPOLAR = "bipolar"
# należy zrównoważyć zbiór uczący dodać żeby było po równo klasyfikacji


class Adaline:

    def __init__(self, initial_weights_dist=1.0, alpha=0.5, err_threshold=0.3, max_epochs=10000):
        self.activation = BIPOLAR
        self.initial_weights_dist = initial_weights_dist
        self.alpha = alpha  # alpha od 0 do 1
        self.bias = 1.0
        self.threshold = 0
        self.weights = None
        self.training_inputs = None
        self.training_outputs = None
        self.epochs = 0
        self.max_epochs = max_epochs
        self.err_threshold = err_threshold

    # bias jest zawsze równy 1
    # metoda uczenia ALC adaptive linear combiner
    # ADALINE jest tylko bipolarny
    # delta wartość rzeczywista
    # różnica w metodzie uczenia LMS - least mean squared
    # LMS (wektory - wektorwag * wektorx)^2 całośc uśrednić
    # zmiana wag w(t+1) + alpha * gradientbłędu
    # po przyblizeniu gradientu -2*błąd*x
    # w(t+1) = w(t) + alpha * (błąd / czy przybliżenie?) * wektorx
    def train(self, training_vectors_inputs, training_vectors_outputs):
        self.training_outputs = training_vectors_outputs
        if self.bias != 0:
            self.training_inputs = add_bias(training_vectors_inputs)
        else:
            self.training_inputs = training_vectors_inputs
        row, col = self.training_inputs.shape
        self.weights = initialize_weights(self.initial_weights_dist, col)
        err_lms = 99999.9

        while err_lms >= self.err_threshold and self.epochs < self.max_epochs:
            i = 0
            self.epochs = self.epochs + 1
            for i in range(row):
                act = activation_f(self.training_inputs[i], self.weights)
                err = error_simple(self.training_outputs[i][0], act)
                self.weights = change_weights_adaline(self.weights, self.alpha, err, self.training_inputs[i])
                i = i + 1
            err_lms = error_lms(self.training_inputs, self.training_outputs, self.weights)

        print(err_lms)
        print(self.weights)

    def test(self, test_input, test_output):
        to = test_output
        if self.bias != 0:
            ti = np.append(test_input, 1)
        else:
            ti = test_input
        act = activation_f(ti, self.weights)
        imp = impulse(act, self.activation, self.threshold)
        return imp

    def show_epochs(self):
        return self.epochs


class Perceptron:

    def __init__(self, activation=BINARY, initial_weights_dist=1.0, alpha=0.5, bias=0.0, threshold=1.0, max_epochs=10000):
        self.activation = activation
        self.initial_weights_dist = initial_weights_dist
        self.alpha = alpha  # alpha od 0 do 1
        self.bias = bias
        self.threshold = 0.0
        if self.bias == 0.0:
            self.threshold = threshold
        self.weights = None
        self.training_inputs = None
        self.training_outputs = None
        self.epochs = 0
        self.max_epochs = max_epochs

    # przed inicjalizacja wag, okresl alpha, okresl theta(threshold, próg) albo bias ( dynamiczny próg)
    # 1 podaj pierwszy wzorzecz uczacy
    # 2 policz całkowite pobudzenie
    # 3 sprawdz prog impulsu act wieksza od theta
    # 4 oblicz blad
    # 5 aktualizuj wagi
    # powtorz dla kolejnego wzorca kroki 2-5
    # przejscie przez wszystkie wzorce uczace nazywa się epoką - epoch
    # po epoce sprawdz wszystkie wzorce i czy są poprawne albo sprawdz czy zostały zmienione wagi i zakoncze jesli nie
    # powtarzaj epoki az do konca
    def train(self, training_vectors_inputs, training_vectors_outputs):
        weights_changed = True
        self.training_outputs = training_vectors_outputs
        if self.bias != 0:
            self.training_inputs = add_bias(training_vectors_inputs)
        else:
            self.training_inputs = training_vectors_inputs
        row, col = self.training_inputs.shape
        self.weights = initialize_weights(self.initial_weights_dist, col)

        while weights_changed and self.epochs < self.max_epochs:
            i = 0
            self.epochs = self.epochs+1
            weights_changed = False
            for i in range(row):
                act = activation_f(self.training_inputs[i], self.weights)
                imp = impulse(act, self.activation, self.threshold)
                err = error_simple(self.training_outputs[i][0], imp)
                if err != 0:
                    weights_changed = True
                    self.weights = change_weights(self.weights, self.alpha, err, self.training_inputs[i])
                i = i + 1

        print(self.weights)

    def test(self, test_input, test_output):
        to = test_output
        if self.bias != 0:
            ti = np.append(test_input, 1)
        else:
            ti = test_input
        act = activation_f(ti, self.weights)
        imp = impulse(act, self.activation, self.threshold)
        return imp

    def show_epochs(self):
        return self.epochs


def change_inputs(): # zmien inputy na -1 albo 0 w zależności od binary albo bipolar i może znormalizuj połówkowe czy coś
    pass


def add_bias(vectors_inputs): # przerob na bias z zakresu wartosci
    row, col = vectors_inputs.shape
    training_inputs = np.append(vectors_inputs, np.ones((row, 1)), 1)
    return training_inputs


def initialize_weights(initial_weights_dist, columns):
    weights_vector = np.random.uniform(-initial_weights_dist, initial_weights_dist, columns)
    return weights_vector


def change_weights(weights, alpha, error, inputs):
    deltaweights = alpha * error * inputs
    changedweights = np.add(weights, deltaweights)
    return changedweights


def change_weights_adaline(weights, alpha, error, inputs):
    deltaweights = 2 * alpha * error * inputs
    changedweights = np.add(weights, deltaweights)
    return changedweights


def error_simple(desired_output, output):
    return desired_output - output


def error_lms(inputs, outputs, weight_vector):
    rows, cols = inputs.shape
    all_e2 = np.sum(np.subtract(outputs, activation_f(inputs, weight_vector)))
    return np.power(all_e2, 2) / rows


def activation_f(input_vector, weight_vector):
    return np.dot(input_vector, weight_vector)


def impulse(act, activation, threshold):
    impulse = 0
    if activation == BINARY:
        impulse = binary(act, threshold)
    else:
        impulse = bipolar(act, threshold)
    return impulse


def binary(act, threshold):
    return 1 if act > threshold else 0


def bipolar(act, threshold): # inputs nalezy tez zmienic na -1 jakas pomocnicza funkcja zmieniajaca to jesli konieczne
    return 1 if act > threshold else -1

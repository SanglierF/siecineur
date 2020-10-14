import random as rand
import numpy as np

def generateand(number=100):
    outputtrue = True
    inputs = np.array(([1, 1], [1, 0], [0, 1], [0, 0]))
    outputs = np.array(([1], [0], [0], [0]))
    for i in range(number):
        outputtrue = bool(rand.getrandbits(1))
        if outputtrue:
            xi1 = rand.uniform(0.6, 1.00)
            xi2 = rand.uniform(0.6, 1.00)
            o = np.array([1])
            input_vector = np.array([xi1, xi2])
        else:
            if bool(rand.getrandbits(1)):
                xi1 = rand.uniform(0.6, 1.0)
                xi2 = rand.uniform(0.0, 0.4)
            elif bool(rand.getrandbits(1)):
                xi1 = rand.uniform(0.0, 0.4)
                xi2 = rand.uniform(0.6, 1.0)
            else:
                xi1 = rand.uniform(0.0, 0.4)
                xi2 = rand.uniform(0.0, 0.4)
            o = np.array([0])
            input_vector = np.array([xi1, xi2])
        inputs = np.vstack([inputs, input_vector])
        outputs = np.vstack([outputs, o])
    return inputs, outputs

def generateandbipolar(number=100, error_threshold=0.3):
    outputtrue = True
    inputs = np.array(([-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]))
    outputs = np.array(([-1], [-1], [-1], [1]))
    for i in range(number):
        outputtrue = bool(rand.getrandbits(1))
        if outputtrue:
            xi1 = rand.uniform(1.0-error_threshold, 1.00)
            xi2 = rand.uniform(1.0-error_threshold, 1.00)
            o = np.array([1])
            input_vector = np.array([xi1, xi2])
        else: # zmien
            if bool(rand.getrandbits(1)):
                xi1 = rand.uniform(-1.0, -1.0+error_threshold)
                xi2 = rand.uniform(1.0-error_threshold, 1.0)
            elif bool(rand.getrandbits(1)):
                xi1 = rand.uniform(1.0-error_threshold, 1.0)
                xi2 = rand.uniform(-1.0, -1.0+error_threshold)
            else:
                xi1 = rand.uniform(-1.0, -1.0+error_threshold)
                xi2 = rand.uniform(-1.0, -1.0+error_threshold)
            o = np.array([-1])
            input_vector = np.array([xi1, xi2])
        inputs = np.vstack([inputs, input_vector])
        outputs = np.vstack([outputs, o])
    return inputs, outputs

import numpy as np


# Initializes weights (w) and biases (b) given data (x) labels (y) # of data samples (m)
def hyperinit(m, alpha, iterat, numlayers, numnodes, activ):
    hparam = {}
    hparam["m"] = m
    hparam["alpha"] = alpha
    hparam["iterat"] = iterat
    hparam["numlayers"] = numlayers
    hparam["numnodes"] = numnodes
    hparam["activ"] = activ
    # parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * .01

    return hparam

# Initializes all layers. Weights, biases, activation type.
# Input and output layers are separate. Output layer always uses activation: "Sigmoid"
def layerinit(hparam, x, y, m):

    # Input layer initialization
    numlayers = hparam["numlayers"]
    numnodes = hparam["numnodes"]
    layers = {}
    w = np.random.randn(x.shape[0], numnodes) * .1  # Random array * 'small number' to create small initial weights
    b = np.array(np.zeros(shape=(numnodes, 1)))
    layers["1"] = {}
    layers["1"]["w"] = w
    layers["1"]["b"] = b
    layers["1"]["activation"] = hparam["activ"]
    print("new input layer 1 created")
    print("w shape:", w.shape)
    print("b shape:", layers["1"]["b"].shape)

    # Hidden layer initialization.
    for i in range(2, numlayers):
        layers[str(i)] = {}
        layers[str(i)]["w"] = np.random.randn(numnodes, numnodes) * .1 # Random array * 'small number' to create small initial weights
        layers[str(i)]["b"] = np.array(np.zeros(shape=(numnodes, 1)))
        layers[str(i)]["activation"] = hparam["activ"]
        print("new hidden layer", i, "created")
        print("w shape:", layers[str(i)]["w"].shape)
        print("b shape:", layers[str(i)]["b"].shape)

    # Output layer initialization.
    layers[str(numlayers)] = {}
    layers[str(numlayers)]["w"] = np.random.randn(numnodes, y.shape[0]) * .1
    layers[str(numlayers)]["b"] = np.array(np.zeros(shape=(y.shape[0], 1)))
    layers[str(numlayers)]["activation"] = str("Sigmoid")
    print("new output layer", numlayers, "created")
    print("w shape:", layers[str(numlayers)]["w"].shape)
    print("b shape:", layers[str(numlayers)]["b"].shape)
    print("layers:", layers.keys())

    return layers
import numpy as np

import Activation


# Runs the network through forward propagation given the activation type. Ex. "Sigmoid"
def forwardprop(x, y, hparam, layers):

    numlayers = hparam["numlayers"]

    layers["1"]["z"] = np.dot((layers["1"]["w"]).T, x) + layers["1"]["b"]
    #print("input z shape:", layers["1"]["z"].shape)
    layers["1"]["z"] /= x.shape[0]
    layers["1"]["A"] = Activation.sigmoid(layers["1"]["z"])
    #print("input A shape:", layers["1"]["A"].shape)

    for i in range(2, hparam["numlayers"]):
        layers[str(i)]["z"] = np.dot(layers[str(i)]["w"].T,layers[str(i-1)]["A"]) + layers[str(i)]["b"]
        layers[str(i)]["z"] /= layers[str(i+1)]["w"].shape[0]
        #print("hidden z shape:", layers[str(i)]["z"].shape)
        #print("i", i)

        # Sigmoid function on the weights of the current layer
        if layers[str(i)]["activation"] == "Sigmoid":
            layers[str(i)]["A"] = Activation.sigmoid(layers[str(i)]["z"])

        #print("hidden A shape:", layers[str(i)]["A"].shape)


        # Tanh function on the weights on the weights of the current layer
        if layers[str(i)]["activation"] == "Tanh":
            layers[str(i)]["A"] = Activation.tanh(layers[str(i)]["z"])

    layers[str(numlayers)]["z"] = np.dot(layers[str(numlayers)]["w"].T, layers[str(numlayers-1)]["A"])
    layers[str(numlayers)]["z"] += layers[str(numlayers)]["b"]
    layers[str(numlayers)]["z"] /= y.shape[0]
    #print("output z shape:", layers[str(numlayers)]["z"].shape)
    layers[str(hparam["numlayers"])]["A"] = Activation.sigmoid(layers[str(numlayers)]["z"])
    #print("output A shape:", layers[str(hparam["numlayers"])]["A"].shape)


    return layers
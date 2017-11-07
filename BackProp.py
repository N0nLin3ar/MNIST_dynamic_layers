import numpy as np


# Calculates the backpropagation for the network
def backprop(x, y, hparam, layers):

    # Output layer backprop
    layers = outlbackprop(y, hparam, layers)

    # Hidden layer backprop
    layers = hidlbackprop(hparam, layers)
    
    # Input layer backprop
    layers = inlbackprop(x, hparam, layers)

    layers = weight_upd(layers, hparam)

    return layers


# Calculates backpropagation for the output layer only. Returns dict (layers).
def outlbackprop(y, hparam, layers):

    numlayers = hparam["numlayers"]

    # Calculate output layer (dz), (dw), (db)
    layers[str(numlayers)]["dz"] = layers[str(numlayers)]["A"] - y # derivative z = derivative A with respect to Loss
    #print("\n", "output dz", layers[str(numlayers)]["dz"].shape)

    # Calculate output layer (db)
    layers[str(numlayers)]["db"] = (1/numlayers) * np.sum(layers[str(numlayers)]["dz"], axis=1, keepdims=True)  # derivative for the bias = derivative z
    #print("output db", layers[str(numlayers)]["db"].shape)

    # Calculate output layer (dw)
    layers[str(hparam["numlayers"])]["dw"] = (1/numlayers) * np.dot(layers[str(numlayers)]["dz"], layers[str(numlayers - 1)]["A"].T)  # derivative w = derivative w with respect to A
    #print("output dw", layers[str(hparam["numlayers"])]["dw"].shape)

    #print("updated output w", layers[str(numlayers)]["w"].shape)

    return layers


# Calculates backprop for hidden layers. **Not including the first or output layer.
def hidlbackprop(hparam, layers):

    numlayers = hparam["numlayers"]

    for i in reversed(range(2, numlayers)):
        #print("i", i)
        # Calculate output layer (dz)
        layers[str(i)]["dz"] = (layers[str(i)]["A"] * (1-layers[str(i)]["A"])) * np.dot(layers[str(i+1)]["w"], layers[str(i+1)]["dz"])  # derivative z = derivative z with respect to Loss
        #print("dz", layers[str(i)]["dz"].shape)

        # Calculate output layer (db)
        layers[str(i)]["db"] = (1 / hparam["m"]) * np.sum(layers[str(i)]["dz"], axis=1, keepdims=True)  # derivative for the bias = derivative z
        #print("db", layers[str(i)]["db"].shape)
        #print("output layer", layers[str(i)]["A"].shape)

        # Calculate output layer (dw)
        layers[str(i)]["dw"] = (1 / hparam["m"]) * np.dot(layers[str(i)]["dz"], layers[str(i-1)]["A"].T) # derivative w = derivative w with respect to A
        #print("dw", layers[str(i)]["dw"].shape)

    return layers


# Calculates backpropagation for the output layer only. Returns dict (layers).
def inlbackprop(x, hparam, layers):

    # Calculate output layer (dz), (dw), (db)
    layers["1"]["dz"] = np.dot(layers["2"]["w"].T, layers["2"]["dz"]) * (layers["1"]["A"] * (1-layers["1"]["A"]))
    #print("dz", layers["1"]["dz"].shape)

    # Calculate output layer (dw)
    layers["1"]["dw"] = (1 / hparam["m"]) * np.dot(layers["1"]["dz"], x.T)  # derivative w = derivative w with respect to A
    #print("dw", layers["1"]["dw"].shape)

    # Calculate output layer (db)
    layers["1"]["db"] = (1 / hparam["m"]) * np.sum(layers["1"]["dz"], axis=1,keepdims=True)  # derivative for the bias = derivative z
    #print("db", layers["1"]["db"].shape)
    #print("output layer", layers["1"]["A"].shape)

    return layers

def weight_upd(layers, hparam):

    numlayers = hparam["numlayers"]

    # Output layer updates to (w) and (b)
    layers[str(numlayers)]["w"] -= hparam["alpha"] * layers[str(numlayers)]["dw"].T  # weight update
    layers[str(numlayers)]["b"] -= hparam["alpha"] * layers[str(numlayers)]["db"]

    for i in range(2, numlayers):
        # Layer updates to (w) and (b)
        layers[str(i)]["w"] -= hparam["alpha"] * layers[str(i)]["dw"].T  # weight update
        layers[str(i)]["b"] -= hparam["alpha"] * layers[str(i)]["db"]

    #Layer updates to (w) and (b)
    layers["1"]["w"] -= hparam["alpha"] * layers["1"]["dw"].T  # weight update
    layers["1"]["b"] -= hparam["alpha"] * layers["1"]["db"]

    return layers
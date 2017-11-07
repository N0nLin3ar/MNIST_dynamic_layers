import numpy as np

import Initialize
import ForwardProp
import BackProp
import MNIST_read


# Takes data: (x) w dims:(n, m), (y) w dims:(n,m),
# Hyperparameters: num per training set, learning rate, num iterations, num layers, num nodes, activation type
def nnmodel(x, y, m=500, alpha = .001, iterat = 1000, numlayers=4, numnodes=15, activ="Sigmoid"): #Sets defaults

    print("\n", "NN Model: 11-06-17 \n Python Version:3.5 \n")
    np.random.seed(1) # Sets random seed for the all random.randn calls

    hparam = Initialize.hyperinit(m, alpha, iterat, numlayers, numnodes, activ)
    layers = Initialize.layerinit(hparam, x, y, m)

    for i in range(iterat):
        layers = ForwardProp.forwardprop(x, y, hparam, layers)
        layers = BackProp.backprop(x, y, hparam, layers)

        if ((i+1)/100).is_integer():
            print("iteration #:", i+1)
            print("\n", "Cost (y^ - y):")
            print((1 / hparam["m"]) * np.sum((layers[str(numlayers)]["A"] - y), axis=1, keepdims=True))

    print("\n", "Test x, y values:")
    x_test, y_test = MNIST_read.read(m, 2 *m)
    ForwardProp.forwardprop(x_test, y_test, hparam, layers)
    print("\n", "Cost (y^ - y):")
    print((1 / hparam["m"]) * np.sum((layers[str(numlayers)]["A"] - y), axis=1, keepdims=True))

    return layers

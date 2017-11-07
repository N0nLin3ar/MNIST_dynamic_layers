import MNIST_read
import NNModel


m = 500  # Sets the number of examples per batch
alpha = .001 # Sets the learning rate for the network
iterat = 1000 # Sets the number of iterations

mnist_index = 0    # Current index in the dataset

# read data into an array x(data), y(labels)
x, y = MNIST_read.read(m, mnist_index)  # Reads x and y from the csv file and returns numpy arrays
print("x.shape", x.shape)
print("y.shape", y.shape)

# initialize the model with x, y, (layers, nodes, type of activation)
layers = NNModel.nnmodel(x, y, m, alpha, iterat)

print("\n", "...Done")

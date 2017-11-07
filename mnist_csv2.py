# This script converts the MNIST .idx1-ubyte files to .csv files.
# This is not my script. Original script credit goes to: Joseph Redmon https://pjreddie.com/projects/mnist-in-csv/

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

# Download the .idx1-ubyte files from http://yann.lecun.com/exdb/mnist/ to the local Python project path.
# Replace the path names below with the local path name.
convert("/Python project directory goes here/train-images.idx3-ubyte", "/Python project directory goes here/train-labels.idx1-ubyte",
        "mnist_train.csv", 60000)
convert("/Python project directory goes here/t10k-images.idx3-ubyte", "/Python project directory goes here/t10k-labels.idx1-ubyte",
        "mnist_test.csv", 10000)

import os
import struct
import numpy as np
from PIL import Image

count = 0

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def saveImg(img, label, kind="train"):
    img = img.reshape(28, 28)
    path = 'E:\\Study\\2019-Summer\\SURF\\CNN\\images\\' + kind + '\\'
    label = path + str(label) + "_" + str(count) + ".jpg"
    im = Image.fromarray(img)
    im.save(label)


if __name__ == "__main__":
    images_train, labels_train = load_mnist("E:\Study\\2019-Summer\SURF\CNN\MNIST_data")
    images_test, labels_test = load_mnist("E:\Study\\2019-Summer\SURF\CNN\MNIST_data", kind='t10k')

    train = zip(images_train, labels_train)
    test = zip(images_test, labels_test)

    for (img, label) in train:
        count += 1
        saveImg(img, label)

    for (img, label) in test:
        count += 1
        saveImg(img, label, kind="test")




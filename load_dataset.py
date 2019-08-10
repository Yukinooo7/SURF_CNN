import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = 'E:\\Study\\2019-Summer\\SURF\\CNN\\data\\kaggle\\PetImages'
CATEGORIES = ['Dog', 'Cat']

IMG_SIZE = 100

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        clas_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), 0)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                training_data.append([new_array, clas_num])
            except Exception as e:
                pass


create_training_data()

random.shuffle(training_data)

X = []  # features
y = []  # labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# save model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
#
# pickle_in = open("X.pickle", "rb")
# X = pickle.load(pickle_in)

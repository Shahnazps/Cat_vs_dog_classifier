import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DataDir = r"/home/shahnaz/Documents/academics/machine learning projects/cat-dog classifier/PetImages/"
CATEGORIES = ["Dog", "Cat"]

# preprocessing part of the program
# Access the dataset
for i in CATEGORIES:
    path = os.path.join(DataDir, i)
    for img in os.listdir(path):
        # converting image to array - cv2.imread is used for conversion
        # to reduce the size of the image, grayscale is used
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break

img_size = 100

# resize the image
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(new_array, cmap='gray')
plt.show()

# creating training data
training_data = []


def create_training_data():
    for i in CATEGORIES:
        path = os.path.join(DataDir, i)
        class_num = CATEGORIES.index(i)

        for img in os.listdir(path):
            # eliminating corrupt image
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()
print("Length ", len(training_data))

# shuffling the training_data
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample)

# x and y for training the data
x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

print(x[0].reshape(-1, img_size, img_size, 1))

x = np.array(x).reshape(-1, img_size, img_size, 1)

print(x.shape)

# store the converted data so that we needn't load the data each time
# for this task pickle is used
# pickle serializes objects so that they can be saved to a file and loaded in a program again later on.


pickle_out = (open(
    "/home/shahnaz/Documents/academics/machine learning projects/cat-dog classifier/x.pickle", "wb"))
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = (open(
    "/home/shahnaz/Documents/academics/machine learning projects/cat-dog classifier/y.pickle", "wb"))
pickle.dump(y, pickle_out)
pickle_out.close()

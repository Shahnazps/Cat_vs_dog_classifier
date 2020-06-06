import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model

CATEGORIES = ["Dogie", "Catie"]
image = r"/home/shahnaz/Documents/academics/machine learning projects/cat-dog classifier/Test_images/face1.jpg"


def prepare(image_path):
    img_size = 100
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)


model = tf.keras.models.load_model(
    r"/home/shahnaz/Documents/academics/machine learning projects/cat-dog classifier/Dog_vs_Cat_CNN.model")
prediction = model.predict([prepare(image)])
print(Categories[int(prediction[0][0])])
img = mpimg.imread(image)
imgplot = plt.imshow(img)
plt.title(Categories[int(prediction[0][0])])
plt.show()

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import pickle
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

# loading x and y from x.pickle and y.pickle

pickle_in = open(
    r"/home/shahnaz/Documents/academics/machine learning projects/cat-dog classifier/x.pickle", "rb")
x = pickle.load(pickle_in)

pickle_in = open(
    r"/home/shahnaz/Documents/academics/machine learning projects/cat-dog classifier/y.pickle", "rb")
y = pickle.load(pickle_in)

print(x.shape)

# Normalization of features
x = x/255.0
print(x)

# model

'''sequential model - is appropriate for a plain stack of layers 
    # Define Sequential model with 3 layers
    model = keras.Sequential(
        [
            layers.Dense(2, activation="relu", name="layer1"),
            layers.Dense(3, activation="relu", name="layer2"),
            layers.Dense(4, name="layer3"),
        ]
    )
    # Call model on a test input
    x = tf.ones((3, 3))
    y = model(x) '''

# model
model = Sequential()
# layers = 256 window size = 3x3
model.add(Conv2D(32, (3, 3), activation="relu",
                 kernel_initializer="he_uniform", padding="same", input_shape=x.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation="relu",
                 kernel_initializer="he_uniform", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation="relu",
                 kernel_initializer="he_uniform", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))


model.add(Flatten())
model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

# compile model
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
# 70% for training 30% for validation
model.fit(x, y, batch_size=4, epochs=20, validation_split=0.3)
model.save("/home/shahnaz/Documents/academics/machine learning projects/cat-dog classifier/Dog_vs_Cat_CNN.model")

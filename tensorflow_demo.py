
#tensorflow

import  tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((train_images.shape[0], 28 * 28 )).astype('float32') / 255.0
test_images = test_images.reshape((test_images.shape[0], 28 * 28 )).astype('float32') / 255.0

model = Sequential([
    Input(shape=(784,)),
    Dense(128, activation="relu", input_shape=(28 * 28)),
    Dense(10, activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_acc:", test_acc)

import os

import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


MODEL_PATH = "model.hdf5"


fashion_mnist = keras.datasets.fashion_mnist
train, test = fashion_mnist.load_data()
train_images, train_labels = train
test_images, test_labels = test
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def show_sizes():
    print("train_labels: ", train_images.shape)
    print("len(train_labels) :", len(train_labels))
    print("train_labels: ", train_labels)
    print("test_images.shape: ", test_images.shape)
    print("len(test_labels): ", len(test_labels))
    print("test_labels: ", test_labels)


def show_images(images, labels, length, more_labels=None):
    plt.figure(figsize=(20, 20))
    for i in range(length ** 2):
        plt.subplot(length, length, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i], cmap=plt.cm.binary)
        if more_labels is not None:
            label = ", ".join((class_names[labels[i]],
                               class_names[more_labels[i]]))
        else:
            label = class_names[labels[i]]
        plt.xlabel(label)
    plt.show()


def train_or_load():
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
    else:

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax),
        ])
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss="sparse_categorical_crossentropy",
                      metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=5)
        model.save(MODEL_PATH)

        test_loss, test_accuracy = model.evaluate(test_images, test_labels)
        print("test_loss: ", test_loss)
        print("test_accuracy: ", test_accuracy)

    return model


def predict(model):
    predictions = model.predict(test_images)
    predictions = np.argmax(predictions, axis=1)

    failures = predictions != test_labels
    failed_images = test_images[failures]
    predicted_labels = predictions[failures]
    failed_labels = test_labels[failures]

    show_images(failed_images, predicted_labels, 8, failed_labels)


def main():
    global train_images, test_images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # show_sizes()
    # show_images(5)

    model = train_or_load()
    # predict(model)

    image = PIL.Image.open('image.jpeg')
    image = image.resize((28, 28), PIL.Image.ANTIALIAS)
    image = image.convert("L")
    image = np.asarray(image)
    image = image / 255.0
    image = 1 - image

    # plt.figure()
    # plt.imshow(image, cmap=plt.cm.binary)
    # plt.show()

    prediction = model.predict(np.reshape(image, (1, 28, 28)))
    prediction = np.argmax(prediction)
    print(class_names[prediction])


if __name__ == "__main__":
    main()

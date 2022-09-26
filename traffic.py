import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 3
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Iterating over every number categories.
    for i in range(NUM_CATEGORIES):

        # Joining the path of every sub directory of number categories
        path = os.path.join(data_dir,str(i))

        # Iterating over the file.
        for file in os.listdir(path):

            # Joining the image path.
            image_path = os.path.join(path,file)

            # Reading the image with open cv.
            img = cv2.imread(image_path)

            # Resizing the image according to the given constraints.
            img1 = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)

            # Appending the images and corresponding labels to images and labels list.
            images.append(img1)
            labels.append(i)
            
    return (images,labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Initializing a sequential model of neural network using tensorflow's keras api.
    models = tf.keras.models.Sequential()

    
    # Adding a convolutated layer followed by max-pooling.
    models.add(tf.keras.layers.Conv2D(32, (3, 3), padding = "valid", activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    models.add(tf.keras.layers.MaxPooling2D((2, 2)))
    

    # Adding a dropout layer to avoid overfitting.
    models.add(tf.keras.layers.Dropout(0.2))

    # Repeating the same above steps again but this time with 64 filters in convolution layer and different drop out rate in drop out layer.
    models.add(tf.keras.layers.Conv2D(64, (3, 3), padding = "valid", activation='relu'))
    models.add(tf.keras.layers.MaxPooling2D((2, 2)))
    models.add(tf.keras.layers.Dropout(0.2))

    # Once Again performing the same steps with different filters and different drop out.
    models.add(tf.keras.layers.Conv2D(128, (3, 3), padding = "valid", activation='relu'))
    models.add(tf.keras.layers.MaxPooling2D((2, 2)))
    models.add(tf.keras.layers.Dropout(0.2))

    

    
    # Flattening out for getting the required dimension for neural network.
    models.add(tf.keras.layers.Flatten())

    # Adding a hidden layer with 128 units.
    models.add(tf.keras.layers.Dense(128, activation='relu'))

    # Adding a Drop out to avoid overfitting in the hidden layer.
    models.add(tf.keras.layers.Dropout(0.2))

    
   

    # Adding the output layer with required number of categories for output and "softmax" as activation function.
    models.add(tf.keras.layers.Dense(NUM_CATEGORIES,activation = "softmax"))

    # Compiling the models with specified optimizer, loss and metrics.
    models.compile(optimizer='adam',
             loss='categorical_crossentropy',
              metrics=['accuracy'],
              )

    return models


if __name__ == "__main__":
    main()

# Image Classification with Keras : "Are you close to Santa ?"

# The exercise is composed of several questions, please do them in order and be careful to respect the names of the variables. Do not hesitate to contact the DataScientest team if you encounter problems on help@datascientest.com.

# The purpose of this exercise is to create an image recognition model capable of predicting whether there is a Santa Claus on the image. The model should be as accurate as possible by following a method that minimizes data size for maximum accuracy.

# Image Generator

# Before using images from a database, it is often necessary to pre-process the images.

# In deep learning, the neural network must train on batches of images from the database. The ImageDataGenerator class allows you to generate batches of transformed images to train your network. In this test, you will have to rely on the documentation click here.

# Import ImageDataGenerator from tensorflow.keras.preprocessing.image.

# Create train_datagen an instance of ImageDataGenerator with the parameters :

# rescale = 1./255,
# shear_range = 0.2,
# zoom_range = 0.2,
# rotation_range=40,
# width_shift_range=0.2,
# height_shift_range=0.2,
# horizontal_flip=True,
# fill_mode='nearest'
# Create test_datagen an instance of ImageDataGenerator with as only parameter rescale = 1./255.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Data Import

# There are two folders called images and images_test. Each of them contains two sub-folders of images: not_santa which contains only images of Santa Claus, and not_santa and the other one of random images of objects and people who are not Santa Claus.

# To import and transform images with the ImageDataGenerator class, we use the flow_from_directory method, which takes the path to the directories as an argument. It automatically detects images and classes in two different categories because the images are in two different directories.

# For more information on flow_from_directory, click here.

# To import your images run the following code.
# We take batches of 32

training_set = train_datagen.flow_from_directory(
    "images", target_size=(64, 64), batch_size=32, class_mode="categorical"
)

test_set = test_datagen.flow_from_directory(
    "images_test", target_size=(64, 64), batch_size=32, class_mode="categorical"
)

# Display of the "Augmented" Data

# Let's observe the result of an augmentation of data.

# Run the following cell to observe augmented data.
import matplotlib.pyplot as plt

batches_real = test_datagen.flow_from_directory(
    "images", target_size=(512, 512), batch_size=16, class_mode="categorical", seed=1234
)
batches_augmented = train_datagen.flow_from_directory(
    "images", target_size=(512, 512), batch_size=16, class_mode="categorical", seed=1234
)

x_batch_augmented, y_batch_augmented = next(batches_augmented)
x_batch_real, y_batch_real = next(batches_real)

for i in range(16):
    image_augmented = x_batch_augmented[i]
    image_real = x_batch_real[i]

    title_add_on = "random image"
    if y_batch_augmented[i][1]:
        title_add_on = "santa"

    plt.subplot(221)
    plt.imshow(image_real)
    plt.title("original " + title_add_on)

    plt.subplot(222)
    plt.imshow(image_augmented)
    plt.title("augmented " + title_add_on)

    plt.show()

# What are the transformations made on the augmented data?
# In your opinion, how is the augmentation of data useful for the training of a neural network?

# Rescaling ensures that pixel values are between 0 and 1.
# with shear we slant the image with a range of +- 20% of the shear angle
# with zoom we zoom into or out of images in a range of +- 20%
# with rotation we rotate the image by + or - 40 degree
# with width or height shift we shift the image horizontally or vertically up to +/- 20% of widht/height
# with flip we flip the image horizontally
# we fill newly created pixels with the nearest pixel value

# This augmentation makes sense or is useful, since we create different patterns for the image to detect.
# Therefore, we enhance the ability (hopefully) of the network to detect the correct picture

# Creation of the neural network

# To classify these images, you can use the following neural network:

# The output of the .summary() method of your template should look like this :
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_5 (Conv2D)            (None, 64, 64, 32)        128
# _________________________________________________________________
# max_pooling2d_5 (MaxPooling2 (None, 32, 32, 32)        0
# _________________________________________________________________
# conv2d_6 (Conv2D)            (None, 30, 30, 32)        9248
# _________________________________________________________________
# max_pooling2d_6 (MaxPooling2 (None, 15, 15, 32)        0
# _________________________________________________________________
# flatten_3 (Flatten)          (None, 7200)              0
# _________________________________________________________________
# dense_5 (Dense)              (None, 128)               921728
# _________________________________________________________________
# dense_6 (Dense)              (None, 2)                 258
# =================================================================
# Total params: 931,233
# Trainable params: 931,233
# Non-trainable params: 0
# _________________________________________________________________

# Implement your model in an instance named classifier.

# Train your model until you reach a validation accuracy val_accuracy greater than 0.85, by choosing the training parameters wisely.

# NB : With the latest version of Tensorflow, fit manage generators.

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Sequential

classifier = Sequential()
classifier.add(
    Conv2D(filters=32, kernel_size=(1, 1), activation="relu", input_shape=(64, 64, 3))
)

classifier.add(MaxPooling2D(pool_size=2, strides=2))

classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))

classifier.add(MaxPooling2D(pool_size=2, strides=2))

classifier.add(Flatten())

classifier.add(Dense(128, activation="linear"))

classifier.add(Dense(2, activation="softmax"))

classifier.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

classifier.summary()

from tensorflow.keras.callbacks import Callback


class EarlyStoppingThreshold(Callback):
    def __init__(self, threshold=0.85):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get("val_accuracy")
        if val_accuracy is not None and val_accuracy > self.threshold:
            print(
                f"Early stopping since validation accuracy has reached: {self.threshold}"
            )
            self.model.stop_training = True


early_stopping = EarlyStoppingThreshold(threshold=0.85)

nb_img_train = training_set.samples
nb_img_test = test_set.samples

batch_size = 32

training_history = classifier.fit(
    training_set,
    epochs=20,
    steps_per_epoch=nb_img_train // batch_size,
    validation_data=test_set,
    validation_steps=nb_img_test // batch_size,
    callbacks=[early_stopping],
)

# Testing the model

# Execute the following code to obtain the probability of a Thor image.
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg
import numpy as np

txt = "thor.jpg"  # Préciser le chemin local
test_image = image.load_img(txt, target_size=(64, 64))
test_image = image.img_to_array(test_image) / 255
test_image = np.expand_dims(test_image, axis=0)

proba = 100 * classifier.predict(test_image)[0]

if len(proba) == 1:
    if proba[0] < 50:
        santa_or_not = "Not Santa"
        proba = round(100 - proba[0], 2)
    else:
        santa_or_not = "Santa"
        proba = round(proba[0], 2)
else:
    if proba[1] < 50:
        santa_or_not = "Not Santa"
        proba = round(100 - proba[1], 2)
    else:
        santa_or_not = "Santa"
        proba = round(proba[1], 2)

img = mpimg.imread(txt)
plt.axis("off")
plt.text(
    -10,
    -15,
    santa_or_not + ": " + str(proba) + "%",
    color=(1, 0, 0),
    fontsize=20,
    fontweight="extra bold",
)
imgplot = plt.imshow(img)

# Repeat the test with the following pictures:

# chien.jpg

# chat.jpg

# selfie.jpg

# santa_rock.jpeg

txt = "chien.jpg"  # Préciser le chemin local
test_image = image.load_img(txt, target_size=(64, 64))
test_image = image.img_to_array(test_image) / 255
test_image = np.expand_dims(test_image, axis=0)

proba = 100 * classifier.predict(test_image)[0]

if len(proba) == 1:
    if proba[0] < 50:
        santa_or_not = "Not Santa"
        proba = round(100 - proba[0], 2)
    else:
        santa_or_not = "Santa"
        proba = round(proba[0], 2)
else:
    if proba[1] < 50:
        santa_or_not = "Not Santa"
        proba = round(100 - proba[1], 2)
    else:
        santa_or_not = "Santa"
        proba = round(proba[1], 2)

img = mpimg.imread(txt)
plt.axis("off")
plt.text(
    -10,
    -15,
    santa_or_not + ": " + str(proba) + "%",
    color=(1, 0, 0),
    fontsize=20,
    fontweight="extra bold",
)
imgplot = plt.imshow(img)

txt = "thor.jpg"  # Préciser le chemin local
test_image = image.load_img(txt, target_size=(64, 64))
test_image = image.img_to_array(test_image) / 255
test_image = np.expand_dims(test_image, axis=0)

proba = 100 * classifier.predict(test_image)[0]

if len(proba) == 1:
    if proba[0] < 50:
        santa_or_not = "Not Santa"
        proba = round(100 - proba[0], 2)
    else:
        santa_or_not = "Santa"
        proba = round(proba[0], 2)
else:
    if proba[1] < 50:
        santa_or_not = "Not Santa"
        proba = round(100 - proba[1], 2)
    else:
        santa_or_not = "Santa"
        proba = round(proba[1], 2)

img = mpimg.imread(txt)
plt.axis("off")
plt.text(
    -10,
    -15,
    santa_or_not + ": " + str(proba) + "%",
    color=(1, 0, 0),
    fontsize=20,
    fontweight="extra bold",
)
imgplot = plt.imshow(img)

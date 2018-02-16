from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
from keras.applications.vgg16 import VGG16
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob

class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

def getModel():
    vgg16_conv = VGG16(weights="imagenet", include_top=False)
    vgg16_conv.summary()

    for layer in vgg16_conv.layers:
        layer.trainable = False

    input = Input(shape=(224,224, 3), name="image_input")
    output_vgg16_conv = vgg16_conv(input)
    
    x = Flatten(name="flatten")(output_vgg16_conv)
    x = Dense(1024, activation="relu", name="fc1")(x)
    x = Dropout(0.5)(x)
    #x = Dense(4096, activation="relu", name="fc2")(x)
    #x = Dense(2, activation="softmax", name="predictions")(x)
    x = Dense(1, activation="sigmoid", name="predictions")(x)

    model = Model(inputs=input, outputs=x)

    model.summary()
    return model

def getImagesInDirectory(directory):
    imageDir = directory + "/**/*.jp*"
    print(imageDir)
    for filename in glob.glob(imageDir, recursive=True):
        yield os.path.join(directory, filename)

def countImages(directory):
    return len([item for item in getImagesInDirectory(directory)])

def loadData(trainDirectory, testDirectory, batchSize, class_mode = "categorical"):
    train_datagen = ImageDataGenerator(rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        trainDirectory,
        target_size=(224, 224),
        batch_size=batchSize,
        class_mode=class_mode)

    validation_generator = test_datagen.flow_from_directory(
        testDirectory,
        target_size=(224, 224),
        batch_size=batchSize,
        class_mode=class_mode)

    return train_generator, validation_generator

def main(argv):
    if len(argv) <= 2:
        print("usage: vgg16_faces.py <trainDirectory> <testDirectory>")
        exit(1)

    trainDirectory = argv[1]
    testDirectory = argv[2]

    #configure TensorFlow to try and prevent high memory usage on the GPU
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config = config)
    #set_session(session)	

    epochs = 60
    #batchSize = 32
    batchSize = 30
    print("Loading data...")
    train_generator, validation_generator = loadData(trainDirectory, testDirectory, batchSize, "binary")
    numberOfTrainingImages = countImages(trainDirectory)
    print(numberOfTrainingImages)
    numberOfValidationImages = countImages(testDirectory)
    print(numberOfValidationImages)
    # Test pretrained model
    #model = ZFNet('vgg16_weights.h5')
    print("Compiling model...")
    model = getModel()
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9)#, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=["accuracy"])

    print("Training model...")
    history = AccuracyHistory()
    modelCheckpoint = ModelCheckpoint("vgg16_faces_{epoch:02d}_{val_acc:.2f}.hdf5", monitor="val_acc", verbose=1, save_best_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=numberOfTrainingImages/batchSize,
        epochs=epochs,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=numberOfValidationImages/batchSize,
        callbacks=[history, modelCheckpoint])

    #model.save('vvg16_faces.h5')

    #score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    #out = model.predict(im)
    #print np.argmax(out)
    plt.plot(range(1,epochs + 1),history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

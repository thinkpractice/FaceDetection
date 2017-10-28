from keras.models import Model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.callbacks import Callback, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob

def getModel():
    vgg16_conv = VGG16(weights="imagenet")
    vgg16_conv.summary()
    return vgg16_conv

def main(argv):
    if len(argv) <= 1:
        print("usage: vgg16_faces.py <image filename>")
        exit(1)

    imageFilename = argv[1]

    model = getModel()

    img = image.load_img(imageFilename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    print('Predicted:', decode_predictions(features)[0])

if __name__ == "__main__":
    main(sys.argv)

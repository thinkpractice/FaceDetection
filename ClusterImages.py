import numpy as np
import glob
import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, GlobalMaxPooling2D
from keras import applications
from sklearn.cluster import KMeans
from collections import defaultdict
from random import shuffle
import matplotlib.pyplot as plt
import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# dimensions of our images.
img_width, img_height = 400, 400
numberOfClusters = 4
top_model_weights_path = 'bottleneck_fc_model.h5'
epochs = 50

def getImagesInDirectory(directory, recursive=False):
    return image.list_pictures(directory)

def imagesFromDirectory(imagePaths, targetSize, batchSize):
    imagesInBatch = []
    for index, imageFile in enumerate(imagePaths):
        imageBytes = image.load_img(imageFile, target_size=targetSize)
        imageArray = image.img_to_array(imageBytes) / 255.
        imagesInBatch.append(imageArray)
        if (index + 1) % batchSize == 0:
            batch = np.array(imagesInBatch)
            imagesInBatch = []
            yield batch

def calculateBottlebeckFeatures(imagePaths, nb_train_samples, batch_size):
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    model.summary()

    #x = Conv2D(numberOfClusters, (3,3))(model.output)
    #x = GlobalMaxPooling2D()(x)
    generator = imagesFromDirectory(imagePaths, (img_height, img_width), batch_size)
    bottleneck_features = model.predict_generator(
        generator, nb_train_samples // batch_size)

    return bottleneck_features.reshape(bottleneck_features.shape[0], bottleneck_features.shape[1] * bottleneck_features.shape[2] * bottleneck_features.shape[3])

def main():
    train_data_dir = '../Faces/MoreData/Train'
    nb_train_samples = 9000
    batch_size = 20

    #Get image features from CNN
    print("Getting image files")	
    imagePaths = getImagesInDirectory(train_data_dir, True)
    shuffle(imagePaths)
    print ("Extracting features")
  
    features = calculateBottlebeckFeatures(imagePaths, nb_train_samples, batch_size)
    print("bottleneck_features={}".format(features.shape))	

    #Perform a clustering on image features
    print("Peforming clustering")
    kmeans = KMeans(n_clusters=numberOfClusters).fit(features)
    imageClusters = defaultdict(list)
    for label, filename in zip(kmeans.labels_, imagePaths):
        imageClusters[label].append(filename)

    print("Plotting clusters")
    for label in imageClusters.keys():
        images = imageClusters[label]
        f, axes = plt.subplots(10, 10)
        plt.title(str(label))
        imageIndex = 0
        for row in axes:
            for col in row:
                if imageIndex < len(images):
                    filePath = images[imageIndex]
                    imageData = image.load_img(filePath)
                    col.imshow(imageData)
                imageIndex += 1
    plt.show()

if __name__ == "__main__":
    main()

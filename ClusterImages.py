from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, GlobalMaxPooling2D
from keras import applications
from sklearn.cluster import KMeans
from collections import defaultdict, namedtuple
from random import shuffle
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import glob
import math
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# dimensions of our images.
img_width, img_height = 400, 400
numberOfClusters = 4
top_model_weights_path = 'bottleneck_fc_model.h5'

epochs = 50

labeledPoint = namedtuple('features', 'label', 'distance')

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

def initializeClusterCenters(images, numberOfClusters):
    return random.choices(images, k=numberOfClusters)

def calculateFeatures(model, centerImages):
    #extract feature representation from fully connected layer
    predictions = model.predict(centerImages)
    fullyConnectedLayer = model.get_layer(name="fc9") 
    return fullyConnectedLayer.output

def distance(imageFeatures, center):
    diff = imageFeatures - centerFeatures
    return math.sqrt(np.dot(diff, diff))

def calculateDistance(imageFeatures, centerFeatures):
    allDistances = [(centerIndex, distance(imageFeatures, center.features)) for centerIndex, center in enumerate(centerFeatures)]
    centerIndex, minDistance = min(allDistances, key=lambda x: x[1])
    return labeledPoint(imageFeatures, centerFeatures[centerIndex].label, minDistance)

def fineTuneModel(model, selectedPoints, Km):
    pass

def updateCentroids(centerFeatures, learningRates):
    pass

def cluster(model, images, numberOfClusters, epochs, miniBatchSize, Km):
    C = initializeClusterCenters(images, numberOfClusters)
    centerFeatures = calculateFeatures(model, C)
    Mn = []
    Yn = []
    for epoch in range(epochs):
        M = random.sample(images, miniBatchSize)

        #assign each point to its closest center
        distances = []
        pointsPerLabel = defaultdict(int)
        for m in M:
            d = calculateDistance(m, centerFeatures)
            distances.append(d)
            pointsPerLabel[d.label] += 1
        learningRates = {key: 1 / pointsPerLabel[key] for key in pointsPerLabel.keys()}
        closestFeatures = sorted(distances, key = lambda x: x.distance)[0:Km] 

        Mn.append(closestFeatures)
        if len(Mn) == miniBatchSize:
            fineTuneModel(model, Mn, Km)
            for m in M:
                updateCentroids(centerFeatures, learningRates)
            Mn = []
            Yn = []



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

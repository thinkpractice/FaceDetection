from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import sys
import glob
import os

def loadData(testDirectory, batchSize, class_mode = "categorical"):
    test_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = test_datagen.flow_from_directory(
        testDirectory,
        target_size=(224, 224),
        batch_size=batchSize,
        class_mode=class_mode)

    return validation_generator

def loadModel(filename):
    return load_model(filename)

def getImagesInDirectory(directory):
    imageDir = directory + "/**/*.jp*"
    print(imageDir)
    for filename in glob.glob(imageDir, recursive=True):
        yield os.path.join(directory, filename)

def countImages(directory):
    return len([item for item in getImagesInDirectory(directory)])

def main(argv):
    if len(argv) <= 2:
        print("usage: ValidateModel.py <model file> <validation data directory>")
        exit(1)

    modelFilename = argv[1]
    validationDirectory = argv[2]
    batchSize = 16

    model = loadModel(modelFilename)
    validationData = loadData(validationDirectory, batchSize, "binary")

    numberOfImages = countImages(validationDirectory)
    scores = model.evaluate_generator(validationData, numberOfImages // batchSize)
    print(["{}={}".format(metric, value) for metric, value in zip(model.metrics_names, scores)])

if __name__ == "__main__":
    main(sys.argv)

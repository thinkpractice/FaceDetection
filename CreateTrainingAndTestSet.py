import sys
import os
import glob
import random
import shutil

def getFiles(directory):
    pngFiles = glob.glob1(directory, "*.png")
    jpgFiles = glob.glob1(directory, "*.jpg")
    jpegFiles = glob.glob1(directory, "*.jpeg")
    return [os.path.join(directory, filename) for filename in pngFiles + jpgFiles + jpegFiles]

def getTrainingAndTestSet(filenames,trainRatio=0.9):
    numberOfFiles = len(filenames)
    numberOfTrainFiles = int(trainRatio * numberOfFiles)
    numberOfTestFiles = numberOfFiles - numberOfTrainFiles
    return random.sample(filenames, numberOfTrainFiles), random.sample(filenames, numberOfTestFiles)

def createDirectory(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def copyFiles(filepaths, destinationDir):
    for filepath in filepaths:
        shutil.copy2(filepath, destinationDir)

def main(argv):
    if len(argv) < 3:
        print("usage: CreateTrainingAndTestSet.py <sourcedir> <destinationdir>")
        exit(1)

    sourceDir = argv[1]
    destinationDir = argv[2]

    allFiles = getFiles(sourceDir)
    train, test = getTrainingAndTestSet(allFiles)
    
    trainDir = os.path.join(destinationDir, "Train")
    createDirectory(trainDir)
    copyFiles(train, trainDir)

    testDir = os.path.join(destinationDir, "Test")
    createDirectory(testDir)
    copyFiles(test, testDir)

if __name__ == "__main__":
    main(sys.argv)

import csv
import sys
import os

def getProfileImageUrl(screenName):
    return "https://twitter.com/{}/profile_image?size=original".format(screenName)

def getImageFilename(destinationDirectory, profileId, extension):
    return os.path.join(destinationDirectory, "{}.{}".format(profileId, extension))

def downloadImage(url, destinationPath):
    urllib.request.urlretrieve(url, destinationPath)

def main(argv):
    if len(argv) <= 2:
        print("usage: DownloadProfileImages.py <csv filename> <destination directory>")
        exit(1)
    filename = argv[1]
    destinationDirectory = argv[2]
    with open(filename, "r") as csvFile:
        csvReader = csv.DictReade(csvFile, delimiter=";")
        for row in csvReader:
            imageUrl = getProfileImageUrl(row["screen_name"])
            destinationPath = getImageFilename(destinationDirectory, row["id"], "jpg")
            downloadImage(imageUrl, destinationDirectory)

if __name__ == "__main__":
    main(sys.argv)

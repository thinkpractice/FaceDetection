import csv
import sys
import os
import urllib.request

def getProfileImageUrl(screenName):
    #return "https://twitter.com/{}/profile_image?size=original".format(screenName)
    return screenName.replace("normal", "400x400")

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
        csvReader = csv.DictReader(csvFile, delimiter=";")
        for row in csvReader:
            imageUrl = getProfileImageUrl(row["profile_image_url"])
            extension = "png" if ".png" in imageUrl else "jpg"
            destinationPath = getImageFilename(destinationDirectory, row["id"], extension)
            downloadImage(imageUrl, destinationPath)

if __name__ == "__main__":
    main(sys.argv)

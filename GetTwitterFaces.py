import sys
import requests
import os
import urllib.request
import tweepy
import json
import urllib.request
import csv

class Twitter(object):
    def __init__(self, keysFilename):
        self.__keysFilename = keysFilename
        self.__api = None

    @property
    def keysFilename(self):
        return self.__keysFilename

    @property
    def api(self):
        if not self.__api:
            consumer_key, consumer_secret, access_token, access_token_secret = self.getKeysFromFile(self.keysFilename)
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            
            self.__api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, timeout=10)
        return self.__api

    def getKeysFromFile(self, filename):
        with open(filename, "r") as jsonFile:
            keysDict = json.load(jsonFile)
            return keysDict["api_key"],keysDict["api_secret"], keysDict["access_token"], keysDict["access_token_secret"]

    def limit_handled(self, cursor):
        while True:
            try:
                yield cursor.next()
            except tweepy.RateLimitError:
                time.sleep(15 * 60)
    
    def followersDo(self, userId):
        return self.cursorDo(userId, self.api.followers_ids)

    def friendsDo(self, userId):
        return self.cursorDo(userId, self.api.friends_ids)

    def cursorDo(self, userId, method):
        for item in self.limit_handled(tweepy.Cursor(method, id=userId).items()):
            yield item

class RetrieveUserDetails(object):
    def __init__(self, twitter):
        self.__twitter = twitter
        self.__userIds = set()
    
    @property
    def twitter(self):
        return self.__twitter

    @property
    def userIds(self):
        return self.__userIds

    @userIds.setter
    def userIds(self, value):
        self.__userIds = value

    @property
    def header(self):
        return ["id", "screen_name", "description", "url", "time_zone", "location", "followers_count", "friends_count", "statuses_count", "profile_image_url"]

    def start(self, userId):
        friends = self.twitter.friendsDo(userId)
        followers = self.twitter.followersDo(userId)

        newUserIds = set(friends)
        newUserIds = newUserIds.union(set(followers))
        print(newUserIds)
        
        notInUserIds = newUserIds - self.userIds
        self.userIds = self.userIds.union(newUserIds)
       
        newProfileInfos = []
        #breadth first enumeration of userIds
        for userId in notInUserIds:
            profileInfo = self.processProfile(userId)
            if not profileInfo:
                continue
            yield profileInfo

        for userId in notInUserIds:
            yield from self.start(userId)

    def processProfile(self, userId):
        try:
            user = self.twitter.api.get_user(id = userId)
            print ("retrieving data for: {}".format(user.screen_name))
            return [userId, user.screen_name, user.description, user.url, user.time_zone, user.location, user.followers_count, user.friends_count, user.statuses_count, user.profile_image_url]
        except Exception:
            return []

def getExtension(url):
    if ".png" in pictureUrl.lower():
        return "png"
    return ".jpg"

def downloadImage(url, destinationPath):
    urllib.request.urlretrieve(url, destinationPath)

def downloadImageAtUrl(profileId, pictureUrl, directory):
    fileExtension = getExtension(pictureUrl)
    destinationPath = os.path.join(directory, "{}.{}".format(profileId, fileExtension))
    self.downloadImage(pictureUrl, destinationPath)

def main(argv):
    if len(argv) <= 3:
        print ("usage: GetTwitterFaces.py <keys filename> <csvFilename> <numberOfUsers>")
        exit(1)
    
    keysFilename = argv[1]
    csvFilename = argv[2]
    numberOfUsers = int(argv[3])

    twitter = Twitter(keysFilename)
    retrieveUserDetails = RetrieveUserDetails(twitter)
    user = twitter.api.get_user(screen_name="tallyschmeits")
    
    with open(csvFilename, "w") as csvFile:
        csvWriter = csv.writer(csvFile, delimiter = ";")
        csvWriter.writerow(retrieveUserDetails.header)
        for index, userInfo in enumerate(retrieveUserDetails.start(user.id)):
            csvWriter.writerow(userInfo)
            if index >= numberOfUsers:
                break

if __name__ == "__main__":
    main(sys.argv)



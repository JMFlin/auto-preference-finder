import tensorflow as tf
import random
import sys
import os
import logging
import requests
import datetime
import json
from time import time
from geopy.geocoders import Nominatim

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

URL = "https://api.gotinder.com"
TOKEN = "INSERT_TOKEN"

if not os.path.exists("./images/"):
    os.mkdir("./images/")

if not os.path.exists("./images/tmp/"):
    os.mkdir("./images/tmp/")

geolocator = Nominatim(user_agent='auto-finder')

class Person(object):

    def __init__(self, data, api):
        self._api = api

        self.id = data["_id"]
        self.name = data.get("name", "Unknown")

        self.bio = data.get("bio", "")
        self.distance = data.get("distance_mi", 0) / 1.60934

        self.birth_date = datetime.datetime.strptime(data["birth_date"], "%Y-%m-%dT%H:%M:%S.%fZ") if data.get(
            "birth_date", False) else None
        self.gender = ["Male", "Female", "Unknown"][data.get("gender", 2)]

        self.images = list(map(lambda photo: photo["url"], data.get("photos", [])))

        self.jobs = list(
            map(lambda job: {"title": job.get("title", {}).get("name"), "company": job.get("company", {}).get("name")}, data.get("jobs", [])))

        self.schools = list(map(lambda school: school["name"], data.get("schools", [])))

        if data.get("pos", False):
            self.location = geolocator.reverse(f'{data["pos"]["lat"]}, {data["pos"]["lon"]}')

    def __repr__(self):
        return f'{self.id}  -  {self.name} ({self.birth_date.strftime("%d.%m.%Y")})'

    def like(self):
        return self._api.like(self.id)

    def dislike(self):
        return self._api.dislike(self.id)

    def predict_likeliness(self, model):
        ratings = []
        LOGGER.info(f'Person has {len(self.images)} images in profile')
        for image in self.images:
            req = requests.get(image, stream=True)
            tmp_filename = f'./images/tmp/run.jpg'
            if req.status_code == 200:
                with open(tmp_filename, "wb") as f:
                    f.write(req.content)

            #img = person_detector.get_person(tmp_filename, sess)
            #if img:
            #img = img.convert("L")
            #image.save(tmp_filename, "jpeg")

            img = tf.keras.preprocessing.image.load_img(
                f'./images/tmp/run.jpg', target_size=(518,518)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array)
            LOGGER.info(f'Image {image} has a score of {predictions[0]}')
            ratings.append(predictions[0])
        ratings.sort(reverse=True)
        if len(ratings) == 0:
            return 0.001
        elif len(ratings) == 1:
            return ratings[0]
        return(ratings[0]*0.6 + sum(ratings[1:])/len(ratings[1:])*0.4)


class Profile(Person):

    def __init__(self, data, api):

        super().__init__(data["user"], api)

        self.email = data["account"].get("email")
        self.phone_number = data["account"].get("account_phone_number")

        self.age_min = data["user"]["age_filter_min"]
        self.age_max = data["user"]["age_filter_max"]

        self.max_distance = data["user"]["distance_filter"]
        self.gender_filter = ["Male", "Female"][data["user"]["gender_filter"]]

class tinderAPI():

    def __init__(self, token):
        self._token = token

    def profile(self):
        data = requests.get(URL + "/v2/profile?include=account%2Cuser", headers={"X-Auth-Token": self._token}).json()
        return Profile(data["data"], self)

    def matches(self, limit=10):
        data = requests.get(URL + f"/v2/matches?count={limit}", headers={"X-Auth-Token": self._token}).json()
        return list(map(lambda match: Person(match["person"], self), data["data"]["matches"]))

    def like(self, user_id):
        data = requests.get(URL + f"/like/{user_id}", headers={"X-Auth-Token": self._token}).json()
        return {
            "is_match": data["match"],
            "liked_remaining": data["likes_remaining"]
        }

    def dislike(self, user_id):
        requests.get(URL + f"/pass/{user_id}", headers={"X-Auth-Token": self._token}).json()
        return True

    def nearby_persons(self):
        data = requests.get(URL + "/v2/recs/core", headers={"X-Auth-Token": self._token}).json()
        return list(map(lambda user: Person(user["user"], self), data["data"]["results"]))


def main():
    
    api = tinderAPI(TOKEN)
    session_matches = []
    total_session_likes = 2
    likes = 0
    end_time = time() + 60*60*2
    model = tf.keras.models.load_model('./saved_model/saved_model/model')
    while time() < end_time:
        
        persons = api.nearby_persons()
        #pos_schools = ["Universität Zürich", "University of Zurich", "UZH"]
        
        LOGGER.info(f'\nFound {len(persons)} persons nearby')
        for person in persons:
            LOGGER.info("#---------------------------------#")
            LOGGER.info(f'Analyzing {person}')
            score = person.predict_likeliness(model) #model
            LOGGER.info(f'Profile has a total score of {score}')
            #for school in pos_schools:
            #    if school in person.schools:
            #        print()
            #        score *= 1.2

            if score > 0.5:
                res = person.like()
                LOGGER.info('LIKE')
                LOGGER.info(f'Is match: {res["is_match"]}')
                if res["is_match"]:
                    session_matches.append(person.name)
            else:
                res = person.dislike()
                LOGGER.info('DISLIKE')

            likes = likes + 1
            
            LOGGER.info(f'Session likes + dislikes is {likes}')
            
            if likes >= total_session_likes:
                break

        if likes >= total_session_likes:
                break

    LOGGER.info(f'Total matches this sessions was {len(session_matches)}')

    if len(session_matches) > 0:
        LOGGER.info(f'These are {json.dumps(session_matches)}')

main()
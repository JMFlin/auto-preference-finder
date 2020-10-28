import tensorflow as tf
#import random
#import sys
#import os
import logging
import requests
import datetime
import json
import streamlit as st
import time
#from time import time
from geopy.geocoders import Nominatim

URL = "https://api.gotinder.com"

geolocator = Nominatim(user_agent='auto-finder')

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

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
        imageLocation = st.empty()
        totalScoreLocation = st.empty()

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
            imageLocation.image(image = f'./images/tmp/run.jpg', caption=f'Predicted score {predictions[0]}', width = 200)

            LOGGER.info(f'Image {image} has a score of {predictions[0]}')
            ratings.append(predictions[0])
            if len(ratings) == 0:
                totalScoreLocation.text(f'Profile total running score: 0')
            elif len(ratings) == 1:
                totalScoreLocation.text(f'Profile total running score: {ratings[0]}')
            else:
                totalScoreLocation.text(f'Profile total running score: {sum(ratings) / len(ratings)}')

            time.sleep(2)

        ratings.sort(reverse=True)
        if len(ratings) == 0:
            return 0.001
        elif len(ratings) == 1:
            return ratings[0]
        return(sum(ratings) / len(ratings))


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

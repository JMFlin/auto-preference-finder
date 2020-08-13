import sys
import cv2
import requests
import datetime
import argparse
import logging
from pathlib import Path
from shutil import copyfile
import tkinter as tk
import tkinter.ttk as ttk 
from ttkthemes import ThemedStyle
from geopy.geocoders import Nominatim
from time import sleep, time
from random import random, seed
from os import listdir, remove
from os.path import isfile, join
from PIL import ImageTk, Image, ImageDraw 
from google.cloud import storage
import numpy as np
from sklearn.model_selection import train_test_split
import face_recognition
# import dlib
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

import os
import tensorflow as tf
from tqdm import tqdm
import hashlib

#from mtcnn.mtcnn import MTCNN
#detector = MTCNN()


# URL = '{{ secrets.bot.url }}'
# TOKEN = '{{ secrets.bot.xauth_token }}'
# PROJECT = '{{ secrets.project }}'
# BUCKET = '{{ secrets.bot.bucket }}'
# DESTINATION = '{{ secrets.bot.destination }}'
# SEED = {{ seed }}

#  python3.7 fixmatch.py --filters=32 --dataset=fixmatch_train.1@1-1 --train_dir ./experiments/fixmatch

# https://github.com/serengil/deepface
# from deepface import DeepFace
# demography = DeepFace.analyze(image)
# print(f'{directory}/{filename}')
# print(demography)

VALIDATION_SAMPLES = 250
SEED = 1
seed(SEED)
np.random.seed(SEED)

PROCESSED_PROFILES = './images/unclassified/profiles.txt'
geolocator = Nominatim(user_agent='auto-finder')

# face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

try:
    path = Path("./images/classified/female_faces/negative")
    path.mkdir(parents = True, exist_ok = True)
    path = Path("./images/classified/female_faces/positive")
    path.mkdir(parents = True, exist_ok = True)
    path = Path("./images/classified/male_faces/negative")
    path.mkdir(parents = True, exist_ok = True)
    path = Path("./images/classified/male_faces/positive")
    path.mkdir(parents = True, exist_ok = True)

    path = Path("./images/classified/bio_female/negative")
    path.mkdir(parents = True, exist_ok = True)
    path = Path("./images/classified/bio_female/positive")
    path.mkdir(parents = True, exist_ok = True)
    path = Path("./images/classified/bio_male/negative")
    path.mkdir(parents = True, exist_ok = True)
    path = Path("./images/classified/bio_male/positive")
    path.mkdir(parents = True, exist_ok = True)

    path = Path("./images/unclassified/female")
    path.mkdir(parents = True, exist_ok = True)
    path = Path("./images/unclassified/female")
    path.mkdir(parents = True, exist_ok = True)
    path = Path("./images/unclassified/male")
    path.mkdir(parents = True, exist_ok = True)
    path = Path("./images/unclassified/male")
    path.mkdir(parents = True, exist_ok = True)

    path = Path("./images/unclassified/sanitized_female_faces")
    path.mkdir(parents = True, exist_ok = True)
    path = Path("./images/unclassified/sanitized_male_faces")
    path.mkdir(parents = True, exist_ok = True)


    path = Path("./images/unclassified/bio_female")
    path.mkdir(parents = True, exist_ok = True)
    path = Path("./images/unclassified/bio_male")
    path.mkdir(parents = True, exist_ok = True)
except FileExistsError:
    pass


def face_alignment(img, scale=0.9, face_size=(224,224)):
    '''
    face alignment API for single image, get the landmark of eyes and nose and do warpaffine transformation
    :param face_img: single image that including face, I recommend to use dlib frontal face detector
    :param scale: scale factor to judge the output image size
    :return: an aligned single face image
    '''
    h, w, c = img.shape
    output_img = list()
    face_loc_list = _face_locations_small(img)
    for face_loc in face_loc_list:
        face_img = _crop_face(img, face_loc, padding_size=int((face_loc[2] - face_loc[0])*0.5))
        face_loc_small_img = _face_locations_small(face_img)
        face_land = face_recognition.face_landmarks(face_img, face_loc_small_img)
        if len(face_land) == 0:
            return []
        left_eye_center = _find_center_pt(face_land[0]['left_eye'])
        right_eye_center = _find_center_pt(face_land[0]['right_eye'])
        nose_center = _find_center_pt(face_land[0]['nose_tip'])
        trotate = _get_rotation_matrix(left_eye_center, right_eye_center, nose_center, img, scale=scale)
        warped = cv2.warpAffine(face_img, trotate, (w, h))
        new_face_loc = face_recognition.face_locations(warped, model='hog')
        if len(new_face_loc) == 0:
            return []
        output_img.append(cv2.resize(_crop_face(warped, new_face_loc[0]), face_size))

    return output_img

def _find_center_pt(points):
    '''
    find centroid point by several points that given
    '''
    x = 0
    y = 0
    num = len(points)
    for pt in points:
        x += pt[0]
        y += pt[1]
    x //= num
    y //= num
    return (x,y)

def _angle_between_2_pt(p1, p2):
    '''
    to calculate the angle rad by two points
    '''
    x1, y1 = p1
    x2, y2 = p2
    tan_angle = (y2 - y1) / (x2 - x1)
    return (np.degrees(np.arctan(tan_angle)))

def _get_rotation_matrix(left_eye_pt, right_eye_pt, nose_center, face_img, scale):
    '''
    to get a rotation matrix by using skimage, including rotate angle, transformation distance and the scale factor
    '''
    eye_angle = _angle_between_2_pt(left_eye_pt, right_eye_pt)
    M = cv2.getRotationMatrix2D((nose_center[0]/2, nose_center[1]/2), eye_angle, scale )

    return M

def _dist_nose_tip_center_and_img_center(nose_pt, img_shape):
    '''
    find the distance between nose tip's centroid and the centroid of original image
    '''
    y_img, x_img, _ = img_shape
    img_center = (x_img//2, y_img//2)
    return ((img_center[0] - nose_pt[0]), -(img_center[1] - nose_pt[1]))

def _crop_face(img, face_loc, padding_size=0):
    '''
    crop face into small image, face only, but the size is not the same
    '''
    h, w, c = img.shape
    top = face_loc[0] - padding_size
    right = face_loc[1] + padding_size
    down = face_loc[2] + padding_size
    left = face_loc[3] - padding_size

    if top < 0:
        top = 0
    if right > w - 1:
        right = w - 1
    if down > h - 1:
        down = h - 1
    if left < 0:
        left = 0
    img_crop = img[top:down, left:right]
    return img_crop

def _face_locations_raw(img, scale):
#     img_scale = (tf.resize(img, (img.shape[0]//scale, img.shape[1]//scale)) * 255).astype(np.uint8)
    h, w, c = img.shape
    img_scale = cv2.resize(img, (int(img.shape[1]//scale), int(img.shape[0]//scale)))
    face_loc_small = face_recognition.face_locations(img_scale, model='hog')
    face_loc = []
    for ff in face_loc_small:
        tmp = [pt*scale for pt in ff]
        if tmp[1] >= w:
            tmp[1] = w
        if tmp[2] >= h:
            tmp[2] = h
        face_loc.append(tmp)
    return face_loc

def _face_locations_small(img):
    for scale in [16, 8, 4, 2, 1]:
        face_loc = _face_locations_raw(img, scale)
        if face_loc != []:
            return face_loc
    return []

def _add_to_file(filename):
    with open('./images/unclassified/sanitized_female_faces.txt', "a") as f:
        f.write(filename+"\r\n")

def _preprocess_images():
    
    female = "./images/unclassified/female"
    male = "./images/unclassified/male"
    destination_female_faces = './images/unclassified/sanitized_female_faces'
    destination_male_faces = './images/unclassified/sanitized_male_faces'

    def _copy(directory, destination):
        
        images = [f for f in listdir(f'{directory}') if not isfile(join(f'{destination}', f))]
        logger.info(f'{len(images)} images in original set')

        with open('./images/unclassified/sanitized_female_faces.txt', "r") as list_file:
            already_here = [l.strip() for l in list_file]

        images = [x for x in images if x not in already_here]
        logger.info(f'{len(images)} images remaining')
        
        # for idx in :
        for idx in tqdm(range(len(images))):

            image = Image.open(f'{directory}/{images[idx]}')
            original_width, original_height = image.size

            # Transform the jpg file into a numpy array
            image = np.array(image)
            
            # Find all the faces in the image using the default HOG-based model.
            # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
            # See also: find_faces_in_picture_cnn.py
            # face_locations = face_recognition.face_locations(image, model='hog')
            
            #if len(face_locations) != 1:
            #    logger.info(f'{directory}/{filename} has {len(face_locations)} faces')
            #    _add_to_file(filename)
            #    continue
            #    
            #top, right, bottom, left = face_locations[0]
            #height = bottom - top
            #width = right - left

            #if height/original_height < 0.10 or width/original_width < 0.10:
            #    logger.info(f'{directory}/{filename} face is too small with ratios heigth {height/original_height} and width {width/original_width}')
            #    _add_to_file(filename)
            #    continue

            image = face_alignment(image, scale = 1.05)
            
            if len(image) > 0:
                image = Image.fromarray(image[0])
                image.save(f'{destination}/{images[idx]}')
            else:
                _add_to_file(images[idx])

    _copy(female, destination_female_faces)
    _copy(male, destination_male_faces)


def hash_identifier(identifier):
    identifier = identifier.lower()
    return hashlib.sha256(identifier.encode()).hexdigest()

def create_unlabel_tfrecords(height, width):
    tfrecords = "./images/tfrecords/"
    unlabel = "./images/unclassified/sanitized_female_faces/"
    unlabel_images = os.listdir(unlabel)
    filename_unlabel = f'fixmatch_unlabel.{SEED}@{VALIDATION_SAMPLES}.tfrecord'
    filepath_unlabel = os.path.join(tfrecords, filename_unlabel)

    test_positive_female_faces = './images/classified/female_faces/positive'
    test_negative_female_faces = './images/classified/female_faces/negative'

    unlabel_images = [f for f in listdir(f'{unlabel}') if not isfile(join(f'{test_positive_female_faces}', f))]
    unlabel_images = [f for f in listdir(f'{unlabel}') if not isfile(join(f'{test_negative_female_faces}', f))]

    def _create(dataset, path, writer, label, height, width):

        for idx in tqdm(range(len(dataset))):
            
            image = np.array(Image.open(path + dataset[idx], "r"), dtype='uint8')
            image = cv2.resize(image, (height, width))
            image = cv2.imencode('.jpg', image)[1].tostring()

            feature = dict(image=_bytes_feature(image),
                        label=_int64_feature(0))

            record = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(record.SerializeToString())

    logger.info(f'Building dataset: {filename_unlabel}')
    with tf.io.TFRecordWriter(filepath_unlabel) as writer:

        _create(unlabel_images, unlabel, writer, 0, width, height)
        
        writer.close()

    return filename_unlabel

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_user_tfrecords(height, width):

    tfrecords = "./images/tfrecords/"
    female_negative = "./images/classified/female_faces/negative/"
    female_positive = "./images/classified/female_faces/positive/"

    negative_images = os.listdir(female_negative)
    positive_images = os.listdir(female_positive)

    filename_train = f'fixmatch_train.{SEED}@{VALIDATION_SAMPLES}.tfrecord'
    filename_test = f'fixmatch_test.{SEED}@{VALIDATION_SAMPLES}.tfrecord'
    
    filepath_train = os.path.join(tfrecords, filename_train)
    filepath_test = os.path.join(tfrecords, filename_test)

    def _create(dataset, path, writer, label, which, height, width):
        
        X_train, X_test, y_train, y_test = train_test_split(dataset, [label] * len(dataset), test_size=0.25)

        if "test" in which:
            x = X_test
            y = y_test
        elif "train" in which:
            x = X_train
            y = y_train

        for idx in tqdm(range(len(x))):
            
            image = np.array(Image.open(path + x[idx], "r"), dtype='uint8')
            image = cv2.resize(image, (height, width))
            image = cv2.imencode('.jpg', image)[1].tostring()

            feature = dict(image=_bytes_feature(image),
                        label=_int64_feature(y[idx]))

            record = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(record.SerializeToString())
    
    logger.info(f'Building dataset: {filename_train}')
    with tf.io.TFRecordWriter(filepath_train) as writer:

        _create(positive_images, female_positive, writer, 1, 'train', height, width)
        _create(negative_images, female_negative, writer, 0, 'train', height, width)
       
        writer.close()

    logger.info(f'Building dataset: {filename_test}')
    with tf.io.TFRecordWriter(filepath_test) as writer:

        _create(positive_images, female_positive, writer, 1, 'test')
        _create(negative_images, female_negative, writer, 0, 'test')
       
        writer.close()

    return list([filename_test, filename_train])


def _upload_files_to_storage(name, email, bucket, bucket_destination):

    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket)
    except Exception as e:
        logger.error("Failed to initialize storage client")
        logger.error(e, exc_info=True)
        sys.exit(1)

    logger.info(f'Uploading {email}/{name}')

    try:
        blob = bucket.blob(f'{bucket_destination}/{email}/{name}')
        blob.upload_from_filename(f'./images/tfrecords/{name}')
        logger.info(f'{email}/{name} uploaded to {bucket}/{bucket_destination}')
    except Exception as e:
        logger.error("Failed to upload files to Cloud Storage")
        logger.error(e, exc_info=True)
        sys.exit(1)


class API(object):

    def __init__(self, token):
        self._token = token

    def profile(self):
        data = requests.get(URL + "/v2/profile?include=account%2Cuser", headers={"X-Auth-Token": self._token}).json()
        return Profile(data["data"], self)

    def matches(self, limit=10):
        data = requests.get(URL + f'/v2/matches?count={limit}', headers={"X-Auth-Token": self._token}).json()
        return list(map(lambda match: Person(match["person"], self), data["data"]["matches"]))

    def like(self, user_id):
        data = requests.get(URL + f'/like/{user_id}', headers={"X-Auth-Token": self._token}).json()
        return {
            "is_match": data["match"],
            "liked_remaining": data["likes_remaining"]
        }

    def dislike(self, user_id):
        requests.get(URL + f'/pass/{user_id}', headers={"X-Auth-Token": self._token}).json()
        return True

    def nearby_persons(self):
        data = requests.get(URL + "/v2/recs/core", headers={"X-Auth-Token": self._token}).json()
        return list(map(lambda user: Person(user["user"], self), data["data"]["results"]))


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

    def download_images_and_bio(self, image_folder=".", bio_folder=".", sleep_max_for=0):
        with open(PROCESSED_PROFILES, "r") as f:
            lines = f.readlines()
            if self.id in lines:
                return
        with open(PROCESSED_PROFILES, "a") as f:
            f.write(self.id+"\r\n")

        index = -1
        for image_url in self.images:
            index += 1
            req = requests.get(image_url, stream=True)
            if req.status_code == 200:
                with open(f'{image_folder}/{self.id}_{self.name}_{index}.jpeg', "wb") as f:
                    f.write(req.content)
                if index == 0: 
                    logger.info(f'Saving: {self.name} {self.schools} {self.jobs}')
                    if len(self.bio) > 0:
                        with open(f'{bio_folder}/{self.id}_{self.name}.txt', "w+") as f:
                            f.write(self.bio+"\r\n")
            sleep(random()*sleep_max_for)

    def predict_likeliness(self, classifier, sess):
        ratings = []
        for image in self.images:
            req = requests.get(image, stream=True)
            tmp_filename = f'./images/tmp/run.jpg'
            if req.status_code == 200:
                with open(tmp_filename, "wb") as f:
                    f.write(req.content)
            img = person_detector.get_person(tmp_filename, sess)
            if img:
                img = img.convert("L")
                img.save(tmp_filename, "jpeg")
                certainty = classifier.classify(tmp_filename)
                pos = certainty["positive"]
                ratings.append(pos)
        ratings.sort(reverse=True)
        ratings = ratings[:5]
        if len(ratings) == 0:
            return 0.001
        return ratings[0]*0.6 + sum(ratings[1:])/len(ratings[1:])*0.4


class Profile(Person):

    def __init__(self, data, api):

        super().__init__(data["user"], api)

        self.email = data["account"].get("email")
        self.phone_number = data["account"].get("account_phone_number")

        self.age_min = data["user"]["age_filter_min"]
        self.age_max = data["user"]["age_filter_max"]

        self.max_distance = data["user"]["distance_filter"]
        self.gender_filter = ["Male", "Female"][data["user"]["gender_filter"]]


class Application(tk.Frame):

    def __init__(self, master, source_path, destination_path, images, originals):
        super().__init__(master)

        self.master = master
        self.master.title("Swiper")

        style = ThemedStyle(master)
        style.set_theme("aquativo")

        self.source_path = source_path
        self.destination_path = destination_path
        self.images = images
        self.originals = originals
        self.max_height = 500
        self.create_widgets()

    def create_widgets(self):
        self.text_label = tk.StringVar()

        #self.text_label = ttk.Label(
        #    self.master, text="Please only like or dislike images with visible faces", font="Helvetica 14 bold")
        #self.text_label.grid(row=0, column=0, sticky=tk.W)

        self.btn_like = ttk.Button(
            self.master, text="Add to likes", command=self.positive)
        self.btn_like.grid(row=2, column=1, sticky=tk.E)

        self.btn_dislike = ttk.Button(
            self.master, text="Add to dislikes", command=self.negative)
        self.btn_dislike.grid(row=2, column=1, sticky=tk.S)

        self.btn_delete = ttk.Button(
            self.master, text="DELETE", command=self.delete)
        self.btn_delete.grid(row=2, column=0, sticky=tk.W)

        self.btn_flip = ttk.Button(
            self.master, text="Flip", command=self.flip)
        self.btn_flip.grid(row=2, column=0, sticky=tk.S)

        self.next_image()

    def flip(self):
        self.image_label.destroy()
        self.image = self.image.transpose(Image.FLIP_TOP_BOTTOM)
        
        self.photo = ImageTk.PhotoImage(self.image)
        
        self.image_label = ttk.Label(
            image=self.photo)
        self.image_label.grid(row=1, column=0)

    ## Change copying to saving
    def positive(self):
        logger.info(f'Saving image to: {self.destination_path}/positive/{self.image_id}'.replace('sanitized_', ''))
        try:
            self.image.save(f'{self.destination_path}/positive/{self.image_id}'.replace('sanitized_', ''))
        except StopIteration as e:
            logger.error("Stopping iteration")
            logger.error(e, exc_info=True)
            sys.exit(1)

        self.image_label.destroy()
        self.image_label_2.destroy()
        self.next_image()

    def negative(self):
        logger.info(f'Saving image to: {self.destination_path}/negative/{self.image_id}'.replace('sanitized_', ''))
        try:
            self.image.save(f'{self.destination_path}/Negative/{self.image_id}'.replace('sanitized_', ''))
        except StopIteration as e:
            logger.error("Stopping iteration")
            logger.error(e, exc_info=True)
            sys.exit(1)
        
        self.image_label.destroy()
        self.image_label_2.destroy()
        self.next_image()

    def delete(self):
        logger.info(f'Deleting image: {self.source_path}/{self.image_id}')
        try:
            remove(self.source_path + "/" + self.image_id)
        except StopIteration as e:
            logger.error("Stopping iteration")
            logger.error(e, exc_info=True)
            sys.exit(1)
        
        self.image_label.destroy()
        self.next_image()

    def next_image(self):
        try:
            self.image_id = next(iter(self.images))
            self.original_id = next(iter(self.originals))
            self.images.pop(0)
            self.originals.pop(0)

            while os.path.exists(f'{self.destination_path}/negative/{self.image_id}'.replace('sanitized_', '')) or os.path.exists(f'{self.destination_path}/positive/{self.image_id}'.replace('sanitized_', '')):
                self.image_id = next(iter(self.images))
                self.original_id = next(iter(self.originals))
                self.images.pop(0)
                self.originals.pop(0)

            logger.info(self.image_id)
        except StopIteration:
            sys.exit(0)

        self.image = Image.open(f'{self.source_path}/{self.image_id}')
        self.original = Image.open(f'{self.source_path}/{self.original_id}'.replace('sanitized_', '').replace('_faces', ''))

        self.width, self.height = self.original.size
        
        if self.height > self.max_height:
            resize_factor = self.max_height / self.height
            self.original = self.original.resize((int(self.width*resize_factor), int(self.height*resize_factor)), resample=Image.LANCZOS)
            self.width, self.height = self.original.size

        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label = ttk.Label(
            image=self.photo)

        self.original = ImageTk.PhotoImage(self.original)
        self.image_label_2 = ttk.Label(
            image=self.original)

        self.image_label_2.grid(row=1, column=1)
        self.image_label.grid(row=1, column=0)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()


    # Create the parser
    parser = argparse.ArgumentParser(prog="bot",
                                    description="mode")
    parser.add_argument("--mode",
                       type=str,
                       required=True,
                       choices=["download", "set_preferences", "train", "play", "create_datasets", "process_images"], 
                       default="play",
                       help="mode of the bot")
    
    parser.add_argument('--num', 
			type=int, 
			default=100,
			required=False, 
            help="Number to download")

    parser.add_argument('--upload', 
			type=str2bool, 
			default=True,
			required=False, 
            help="Upload to Cloud Storage")

    parser.add_argument('--height', 
			type=int, 
			default=32,
			required=False, 
            help="Height of image")

    parser.add_argument('--width', 
			type=int, 
			default=32,
			required=False, 
            help="Width of image")

    args = parser.parse_args()
    mode = args.mode
    num = args.num
    upload = args.upload
    height = args.height
    width = args.width
    
    #if mode not in ["train"]:
    #    try:
    #        api = API(TOKEN)
    #        my_profile = api.profile()
    #    except Exception as e:
    #        logger.error("Failed to initialize API")
    #        logger.error(e, exc_info=True)
    #        sys.exit(1)
#

    if mode == "download":

        counter = 0
        logger.info(f'Retrieving {num} images of {my_profile.gender_filter.lower()}s in radius of {my_profile.max_distance}')

        while True:
            persons = api.nearby_persons()
            for person in persons:
                person.download_images_and_bio(
                    image_folder=f'./images/unclassified/{my_profile.gender_filter.lower()}', 
                    bio_folder=f'./images/unclassified/bio_{my_profile.gender_filter.lower()}', 
                    sleep_max_for=random()*3)
                sleep(random()*10)
                
                counter = counter + 1
                logger.info(f'{counter}/{num}')

                if counter == num:
                    sys.exit(0)
            sleep(random()*10)

        _preprocess_images()

    elif mode == "set_preferences":


    # my_profile.gender_filter.lower()

        #source_image_folder = f'./images/unclassified/sanitized_{my_profile.gender_filter.lower()}_faces'
        #destination_image_folder = f'./images/classified/sanitized_{my_profile.gender_filter.lower()}_faces'
        source_image_folder = f'./images/unclassified/sanitized_female_faces'
        destination_image_folder = f'./images/classified/sanitized_female_faces'
        # images = [f for f in listdir(fsource_image_folder) if isfile(join(source_image_folder, f))]
        images = [f for f in listdir(f'{source_image_folder}') if isfile(join(f'{source_image_folder}', f))]
        original_images = [f for f in listdir(f'{source_image_folder}') if isfile(join(f'{source_image_folder}', f))]

        root = tk.Tk()

        app = Application(
            master=root, 
            source_path=source_image_folder, 
            destination_path=destination_image_folder, 
            images=images,
            originals=original_images)

        app.mainloop()

    elif mode == "create_datasets":

        if my_profile.email is not None:
           identifier = my_profile.email
        elif my_profile.phone_number is not None:
           identifier = my_profile.phone_number
        else:
           logger.error("Failed to get an identifier")
           sys.exit(1)
       
        hashed_identifier = hash_identifier(identifier)
        # filenames = create_user_tfrecords(height, width)
        filename = create_unlabel_tfrecords(height, width)

        if upload:
            for name in filenames:
                _upload_files_to_storage(
                    name=name, 
                    email=hashed_identifier, 
                    bucket=BUCKET,
                    bucket_destination=DESTINATION)
 
                _upload_files_to_storage(
                    name=filename, 
                    email='unlabel', 
                    bucket=BUCKET,
                    bucket_destination=DESTINATION)
    
    elif mode == "process_images":
        _preprocess_images()

    logger.info('Done')

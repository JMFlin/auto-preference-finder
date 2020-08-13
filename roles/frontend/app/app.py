import hashlib
import tarfile
import streamlit as st
import logging
import os
import pandas as pd
from google.cloud import storage
import tensorflow as tf

# tar -zcvf archive.tar.gz female/ &&  split -b 250M archive.tar.gz "archive.part" && rm archive.tar.gz
# find . -type f | awk -v N=10 -F / 'match($0, /.*\//, m) && a[m[0]]++ < N' | xargs -r -d '\n' tar -rvf backup.tar
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

@st.cache
def download_blobs():
    """Downloads a blob from the bucket."""
    LOGGER.info(f'tf version: {tf.version.VERSION}')
    if not os.path.exists("./images/"):
        os.mkdir("./images/")

    if not os.path.exists("./images/unclassified/"):
        os.mkdir("./images/unclassified/")

    if not os.listdir('./images/unclassified/female') :

        _, bucket = initialize_gcloud()
        LOGGER.info(f'Beginning to download images from bucket')

        blob = bucket.blob('preferences/archive/archive.tar.gz')
        blob.download_to_filename('./images/unclassified/archive.tar.gz')

        LOGGER.info(f'Extracting files from archive')
        my_tar = tarfile.open('./images/unclassified/archive.tar.gz')
        my_tar.extractall('./images/unclassified/')
        my_tar.close()

        LOGGER.info(f'Blobs downloaded\n')

def initialize_gcloud():
    bucket_name = os.getenv('GCS_BUCKET')
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    return(storage_client, bucket)

def download_create_model(userid):
    
    """Downloads a blob from the bucket."""
    
    if not os.path.exists(f'./saved_model/saved_model/model'):
        _, bucket = initialize_gcloud()
        import time
        try:
            blob = bucket.blob(f'users/{userid}/saved_model/model.tar.gz')
            LOGGER.info(f'Beginning to download model from bucket with url users/{userid}/saved_model/model.tar.gz') 
            blob.download_to_filename(f'./saved_model/model.tar.gz') 
            LOGGER.info(f'Blob downloaded')
            time.sleep(2)
            tar = tarfile.open('./saved_model/model.tar.gz')
            tar.extractall(path="./saved_model/.")
            tar.close()
            LOGGER.info(f'Model extracted\n')
        except:
            pass

def model_predict(image, userid):
    download_create_model(userid)
    
    model = tf.keras.models.load_model('./saved_model/saved_model/model')
    
    img = tf.keras.preprocessing.image.load_img(
        f'./images/unclassified/female/{image}', target_size=(518,518)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)


    predictions = model.predict(img_array)

    score = predictions[0]
    if score > 0.5:
        st.write(f'Model predicted a {score} probability of you liking.')
        st.write(f'Swiping RIGHT')
    else:
        st.write(f'Model predicted a {score} probability of you liking.')
        st.write(f'Swiping LEFT')
    
    LOGGER.info(f'Model predicted {score} for image {image}')
    return score

def initialize():

    if not os.path.exists("./files"):
        os.mkdir("./files")

    if not os.path.exists("./saved_model"):
        os.mkdir("./saved_model")
    
    if not os.path.exists("./files/image_state_file.txt"):
        with open("./files/image_state_file.txt", "w") as text_file:
            print(0, file=text_file)
    
    if not os.path.exists("./files/like_temporary_db_file.txt"):
        open("./files/like_temporary_db_file.txt", "w+")

    if not os.path.exists("./files/dislike_temporary_db_file.txt"):
        open("./files/dislike_temporary_db_file.txt", "w+")

    if not os.path.exists("./files/pass_temporary_file.txt"):
        open("./files/pass_temporary_file.txt", "w+")

    
    source_image_folder = "./images/unclassified/female"
    images = [f for f in os.listdir(f'{source_image_folder}') if os.path.isfile(os.path.join(f'{source_image_folder}', f))]

    images, _, _, _ = sync_likes_dislikes(images)
    return(images)

def like_image(image):
    LOGGER.info(f'User liked image: {image}')
    with open("./files/like_temporary_db_file.txt", "a+") as file_object:
        file_object.write(str(image) + os.linesep)

def dislike_image(image):
    LOGGER.info(f'User disliked image: {image}')
    with open("./files/dislike_temporary_db_file.txt", "a+") as file_object:
        file_object.write(str(image) + os.linesep)

def pass_image(image):
    LOGGER.info(f'User passed image: {image}')
    with open("./files/pass_temporary_file.txt", "a+") as file_object:
        file_object.write(str(image) + os.linesep)

def show_image(result):
    st.image(image = f'./images/unclassified/female/{result}', use_column_width=True)

def image_state():
    if os.path.exists("./files/image_state_file.txt"):
        with open('./files/image_state_file.txt') as f:
            samples_count = int(f.readline())
            samples_count = samples_count + 1
        with open("./files/image_state_file.txt", "w") as text_file:
            print(samples_count, file=text_file)
    else:
        samples_count = 0
    return(samples_count)

def restart():
    LOGGER.info(f'\nUser restarted image labeling\n')
    if os.path.exists("./files/image_state_file.txt"):
        os.remove("./files/image_state_file.txt")
    if os.path.exists("./files/like_temporary_db_file.txt"):
        os.remove("./files/like_temporary_db_file.txt")
    if os.path.exists("./files/dislike_temporary_db_file.txt"):
        os.remove("./files/dislike_temporary_db_file.txt")
    if os.path.exists("./files/pass_temporary_file.txt"):
        os.remove("./files/pass_temporary_file.txt")
    if os.path.exists("./files/model_load_state.txt"):
        os.remove("./files/model_load_state.txt")
    if os.path.exists("./files/preferences.csv"):
        os.remove("./files/preferences.csv")

    source_image_folder = "./images/unclassified/female"
    images = [f for f in os.listdir(f'{source_image_folder}') if os.path.isfile(os.path.join(f'{source_image_folder}', f))]
    return(images)

def prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)

def merge_files(userid):

    prepend_line("./files/like_temporary_db_file.txt", "likes")
    prepend_line("./files/dislike_temporary_db_file.txt", "dislikes")

    likes = pd.read_csv('./files/like_temporary_db_file.txt')
    dislikes = pd.read_csv('./files/dislike_temporary_db_file.txt')
    merged = pd.concat([likes.reset_index(drop=True), dislikes], axis=1)
    merged['userid'] = userid
    merged.to_csv('./files/preferences.csv', sep = ",", index = False)

def upload_blob(userid):
    
    client = storage.Client()
    bucket_name = os.getenv('GCS_BUCKET')
    
    LOGGER.info(f'\nUser uploading files to {bucket_name}\n')

    bucket = client.get_bucket(bucket_name)

    blob = bucket.blob(f'users/{userid}/preferences.csv')
    blob.upload_from_filename('./files/preferences.csv')
    LOGGER.info(f'File preferences.csv uploaded to {bucket_name}/users/{userid}/preferences.csv')


def hash_identifier(identifier):
    identifier = identifier.lower()
    return hashlib.sha256(identifier.encode()).hexdigest()

def sync_likes_dislikes(images):

    if os.path.exists("./files/dislike_temporary_db_file.txt"):
        with open('./files/dislike_temporary_db_file.txt') as f:
            dislikes = [line.rstrip() for line in f]
    else:
        dislikes = []

    if os.path.exists("./files/like_temporary_db_file.txt"):
        with open('./files/like_temporary_db_file.txt') as f:
            likes = [line.rstrip() for line in f]
    else:
        likes = []

    if os.path.exists("./files/pass_temporary_file.txt"):
        with open('./files/pass_temporary_file.txt') as f:
            passes = [line.rstrip() for line in f]
    else:
        passes = []

    images = [x for x in images if x not in dislikes]
    images = [x for x in images if x not in likes]
    images = [x for x in images if x not in passes]

    return(images, likes, dislikes, passes)

def main():

    st.title(body = 'Preference creator')

    st.sidebar.markdown(body = """
    ## Information about the app
    This web app is for creating a training set of your preferences for Tinder. These samples will then be used to train either ResNet50 + FixMatch or VGG16 transfer learning. This is a hobby application and in no way related to Match Groups official Tinder app.
    """)

    userid = st.text_input("Please enter your user id", "")
    userid = hash_identifier(userid)

    download_blobs()
    download_create_model(userid)

    images = initialize()

    if os.path.exists("./files/image_state_file.txt"):
        with open('./files/image_state_file.txt') as f:
            samples_count = int(f.readline())
    else:
        samples_count = 0

    if st.sidebar.button(label = 'Like'):

        like_image(images[samples_count])
        samples_count = image_state()
        
    if st.sidebar.button(label = 'Dislike'):

        dislike_image(images[samples_count])
        samples_count = image_state()

    if st.sidebar.button(label = 'Pass'):
        pass_image(images[samples_count])

    if st.sidebar.button(label = 'Done'):
        if len(userid) > 0:
            #hashed_identifier = hash_identifier(userid)
            merge_files(userid)
            upload_blob(userid)
            images = restart()
            samples_count = 0
            st.write("The sample data has been recorded and the training process will begin shortly!")
        else:
            st.write("Please enter a user id")

    if st.sidebar.button(label = 'Restart'):
        images = restart()
        samples_count = 0

    images, likes, dislikes, passes = sync_likes_dislikes(images)
    st.write(f'Images: {len(images)} Likes: {len(likes)} Dislikes: {len(dislikes)} Passes: {len(passes)}')
    st.write(f'{round(samples_count / 250 * 100, 1)}% complete to minimum suggested amount')
    show_image(images[samples_count])
    model_predict(images[samples_count], userid)

main()
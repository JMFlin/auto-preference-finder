# https://www.tensorflow.org/tutorials/load_data/images
# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
# https://www.tensorflow.org/tutorials/keras/save_and_load
import tensorflow as tf
import logging
import os
import os.path
from tensorflow.keras import layers
from google.cloud import storage
import tarfile
import time
import pandas as pd
import shutil

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

LOGGER.info(f'tf version: {tf.version.VERSION}')

#import pathlib
#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file(origin=dataset_url, 
#                                   fname='flower_photos', 
#                                   untar=True)
#data_dir = pathlib.Path(data_dir)

if not os.path.exists("./data/"):
    os.mkdir("./data/")
if not os.path.exists("./data/negative/"):
    os.mkdir("./data/negative/")
if not os.path.exists("./data/positive/"):
    os.mkdir("./data/positive/")
if not os.path.exists("./training_1/"):
    os.mkdir("./training_1/")
if not os.path.exists("./saved_model/"):
    os.mkdir("./saved_model/")

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

bucket_name = os.getenv('GCS_BUCKET')
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
userid_list = []
trainer_list = []

BATCH_SIZE = 16
IMG_HEIGHT = 518
IMG_WIDTH = 518
NUM_CLASSES = 2
EPOCHS = 50
VALIDATION_SPLIT = 0.2

def watcher():
  
  def download_blobs(userid):
      """Downloads a blob from the bucket."""

      LOGGER.info(f'Beginning to download images from bucket {bucket_name} with url users/{userid}/preferences.csv')

      blob = bucket.blob(f'users/{userid}/preferences.csv')
      blob.download_to_filename(f'./data/preferences.csv')

      LOGGER.info(f'Blobs downloaded\n')

  def my_list_bucket(bucket_name):
    resource_list = []
    trainer_list = []
    a_bucket = storage_client.lookup_bucket(bucket_name)
    bucket_iterator = a_bucket.list_blobs()
    
    for resource in bucket_iterator:
      if 'users' in resource.name:
        resource_list.append(resource.name)
        userid = resource.name.split('/')[1]
        if userid not in userid_list:
          userid_list.append(userid)

    for userid in userid_list:
      if sum(1 for s in resource_list if userid in s) == 1:
        trainer_list.append(userid)

    return(trainer_list)

  while True:
    if len(trainer_list) > 0:
      LOGGER.info(f'userid(s) {trainer_list} found!')
      for userid in trainer_list:
        download_blobs(userid)
        create_datasets()
        train(userid)
        upload_blob(userid)
        cleanup()
        LOGGER.info(f'Process complete for userid: {userid}')
        watcher()
    else:
      LOGGER.info(f'Watching {bucket_name}/users/')
      time.sleep(60*30)
      trainer_list = my_list_bucket(bucket)

def create_datasets():
  source_image_folder = '../images/unclassified/female'
  images = [f for f in os.listdir(f'{source_image_folder}') if os.path.isfile(os.path.join(f'{source_image_folder}', f))]
  user_preferences = pd.read_csv(f'./data/preferences.csv')
  likes = user_preferences['likes'].tolist()
  dislikes = user_preferences['dislikes'].tolist()
  for image in images:
    if image in likes:
      LOGGER.info(f'copying {image} to positive')
      shutil.copyfile(f'../images/unclassified/female/{image}', f'./data/positive/{image}')
    if image in dislikes:
      LOGGER.info(f'copying {image} to negative')
      shutil.copyfile(f'../images/unclassified/female/{image}', f'./data/negative/{image}')

def train(userid):

  base_source_image = './data/'
  positive_source_image = "./data/positive/"
  negative_source_image = "./data/negative/"

  positive_samples = [f for f in os.listdir(f'{positive_source_image}') if os.path.isfile(os.path.join(f'{positive_source_image}', f))]
  negative_samples = [f for f in os.listdir(f'{negative_source_image}') if os.path.isfile(os.path.join(f'{negative_source_image}', f))]

  LOGGER.info(f'positive samples: {len(positive_samples)}')
  LOGGER.info(f'positive samples: {len(negative_samples)}')
  
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_source_image,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_source_image,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

  class_names = train_ds.class_names
  LOGGER.info(f'class names: {class_names}')

  for image_batch, labels_batch in train_ds:
    LOGGER.info(f'image batch shape: {image_batch.shape}')
    LOGGER.info(f'labels batch shape:{labels_batch.shape}')
    break

  AUTOTUNE = tf.data.experimental.AUTOTUNE
  train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomRotation(0.1),
  ])

  # Define a simple sequential model
  def make_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return tf.keras.Model(inputs, outputs)


  model = make_model(input_shape=(IMG_HEIGHT, IMG_WIDTH) + (3,), num_classes=NUM_CLASSES)  

  #model = create_model()
    # Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

  model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
  )

  model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[cp_callback]
  )

  model.save('saved_model/model') 

  def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
      tar.add(source_dir, arcname=os.path.basename(source_dir))

  #LOGGER.info(f'\nCreating archive of model weights\n')
  #make_tarfile('model-weights.archive.tar.gz', './training_1')
  LOGGER.info(f'\nCreating tar.gz of saved_model\n')
  make_tarfile('model.tar.gz', './saved_model')

def upload_blob(userid):
  LOGGER.info(f'\nUploading files to storage\n')

  client = storage.Client()
  bucket_name = os.getenv('GCS_BUCKET')
  
  bucket = client.get_bucket(bucket_name)

  blob = bucket.blob(f'users/{userid}/saved_model/model.tar.gz')
  blob.upload_from_filename('model.tar.gz')
  LOGGER.info(f'File model.tar.gz uploaded to {bucket_name}/users/{userid}/saved_model/model.tar.gz')

def cleanup():
  if os.path.exists("model-weights.archive.tar.gz"):
    LOGGER.info("Clean up: removing archive file")
    os.remove('model-weights.archive.tar.gz')

  def _cleanup(name):
    time.sleep(5)
    if os.path.exists(name):
      LOGGER.info("Clean up: removing model checkpoints")
      folder = name
      for filename in os.listdir(folder):
          file_path = os.path.join(folder, filename)
          try:
              if os.path.isfile(file_path) or os.path.islink(file_path):
                  os.unlink(file_path)
              elif os.path.isdir(file_path):
                  shutil.rmtree(file_path)
          except Exception as e:
              LOGGER.info('Failed to delete %s. Reason: %s' % (file_path, e))
  _cleanup("./training_1")
  _cleanup("./saved_model")

watcher()
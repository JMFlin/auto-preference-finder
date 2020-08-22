import logging
import os
import sys
import tensorflow as tf
import tarfile
import shutil
import math
from tqdm import tqdm
from google.cloud import storage

version = 'v1'
height = 518
width = 518
files_per_tar = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
      tar.add(source_dir, arcname=os.path.basename(source_dir))

def create_common_archives(height, width):

    source_image_folder = "./images/unclassified/female"
    temporary_archive_folder = './images/unclassified/tmp'
    filename = f'archive_partition_{height}_{width}'

    images = [f for f in os.listdir(f'{source_image_folder}') if os.path.isfile(os.path.join(f'{source_image_folder}', f))]
    rounds = math.ceil(len(images)/(files_per_tar))
    
    logger.info(f'Starting to build {rounds} datasets out of {len(images)} files')
    
    for i in tqdm(range(0, rounds)):
        logger.info(f'Building dataset: {filename}_{i}')
        subset = images[:files_per_tar]

        if not os.path.exists(f'{temporary_archive_folder}{i}'):
            os.mkdir(f'{temporary_archive_folder}{i}')
        
        for image in subset:
            shutil.copyfile(f'{source_image_folder}/{image}', f'{temporary_archive_folder}{i}/{image}')

        make_tarfile(f'{filename}_{i}.tar.gz', f'{temporary_archive_folder}{i}') 
        upload_files_to_storage(f'data_partitions/{version}/{filename}_{i}.tar.gz', f'{filename}_{i}.tar.gz')
        shutil.rmtree(f'{temporary_archive_folder}{i}')
        os.remove(f'{filename}_{i}.tar.gz')
        del images[:files_per_tar]
        

def upload_files_to_storage(bucket_destination, local_file):

    try:
        bucket_name = os.getenv('GCS_BUCKET')
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        logger.error("Failed to initialize storage client")
        logger.error(e, exc_info=True)
        sys.exit(1)

    try:
        blob = bucket.blob(f'{bucket_destination}')
        blob.upload_from_filename(f'{local_file}')
    except Exception as e:
        logger.error("Failed to upload files to Cloud Storage")
        logger.error(e, exc_info=True)
        sys.exit(1)

create_common_archives(height, width)
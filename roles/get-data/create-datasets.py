import logging
import os
import sys
import tarfile
import shutil
import math
from PIL import Image
from tqdm import tqdm
from google.cloud import storage

version = 'v1'
files_per_tar = 250

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
      tar.add(source_dir, arcname=os.path.basename(source_dir))

def create_common_archives():

    source_image_folder = "./images/unclassified/female"
    temporary_archive_folder = './images/unclassified/tmp'
    filename = f'archive_partition'

    images = [f for f in os.listdir(f'{source_image_folder}') if os.path.isfile(os.path.join(f'{source_image_folder}', f)) and os.path.join(f'{source_image_folder}', f).endswith(('.jpeg'))]
    rounds = math.ceil(len(images)/(files_per_tar))
    
    logger.info(f'Starting to build {rounds} datasets out of {len(images)} files with {files_per_tar} files per archive')
    
    for i in tqdm(range(0, rounds)):
        logger.info(f'Building dataset: {filename}_{i}')
        subset = images[:files_per_tar]

        if not os.path.exists(f'{temporary_archive_folder}{i}'):
            os.mkdir(f'{temporary_archive_folder}{i}')
        
        for image in subset:
            try:
                im = Image.open(f'{source_image_folder}/{image}')
                im.verify()
                im.close()
                shutil.copyfile(f'{source_image_folder}/{image}', f'{temporary_archive_folder}{i}/{image}')
            except Exception:
                logger.info(f'Skipping {image}')
                pass

        make_tarfile(f'{filename}_{i}.tar.gz', f'{temporary_archive_folder}{i}') 
        upload_files_to_storage(f'data_partitions/{version}/{filename}_{i}.tar.gz', f'{filename}_{i}.tar.gz')
        
        try:
            if i > 1:
                shutil.rmtree(f'{temporary_archive_folder}{i-1}')
        except Exception:
            logger.info(f'Unable to delete {temporary_archive_folder}{i}. Please delete it manualy')
            pass

        try:
            os.remove(f'{filename}_{i}.tar.gz')
        except Exception:
            logger.info(f'Unable to delete {filename}_{i}.tar.gz. Trying again later..')
            pass
        
        del images[:files_per_tar]

    for i in range(0, rounds):
        try:
            shutil.rmtree(f'{temporary_archive_folder}{i}')
        except Exception:
            logger.info(f'Unable to delete {temporary_archive_folder}{i}. Please delete it manualy')
            pass
        

def upload_files_to_storage(bucket_destination, local_file):

    try:
        bucket_name = os.getenv('GCS_BUCKET')
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        logger.error("Failed to initialize storage client")
        logger.error(e, exc_info=True)
        sys.exit(1)

    logger.info(f'Uploading local file {local_file} to Cloud Storage {bucket_destination}')

    try:
        blob = bucket.blob(f'{bucket_destination}')
        blob.upload_from_filename(f'{local_file}')
    except Exception as e:
        logger.error("Failed to upload files to Cloud Storage")
        logger.error(e, exc_info=True)
        sys.exit(1)

create_common_archives()
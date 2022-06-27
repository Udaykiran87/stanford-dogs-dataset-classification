import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils import read_yaml, create_directories
import zipfile
import tarfile
import requests


STAGE = "LoadData" ## <<< change stage name
# ARCHIVE EXTENSIONS
ZIP_EXTENSION = ".zip"
TAR_EXTENSION = ".tar"
TAR_GZ_EXTENSION = ".tar.gz"
TGZ_EXTENSION = ".tgz"
GZ_EXTENSION = ".gz"

EMPTY_URL_ERROR = "ERROR: URL should not be empty."
FILENAME_ERROR = "ERROR: Filename should not be empty."
UNKNOWN_FORMAT = "ERROR: Unknown file format. Can't extract."

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def download_dataset(url, target_path="data/", keep_download=True, overwrite_download=False):
    """Downloads dataset from a url.
    url: string, a dataset path
    target_path: string, path where data will be downloaded
    keep_download: boolean, keeps the original file after extraction
    overwrite_download: boolean, stops download if dataset already exists
    """
    if url == "" or url is None:
        raise Exception(EMPTY_URL_ERROR)

    filename = get_filename(url)
    file_location = get_file_location(target_path, filename)
    create_directories([target_path])
    if os.path.exists(file_location) and not overwrite_download:
        logging.info(f"File already exists at {file_location}. Use: 'overwrite_download=True' to \
        overwrite download")
        extract_file(target_path, filename)
        return
    logging.info(f"Downloading file from {url} to {file_location}.")
    # Download
    with open(file_location, 'wb') as f:
        with requests.get(url,allow_redirects=True, stream=True) as resp:
            for chunk in resp.iter_content(chunk_size = 512):  #chunk_size in bytes
                if chunk:
                    f.write(chunk)
    logging.info("Finished downloading.")
    logging.info("Extracting the file now ...")
    extract_file(target_path, filename)
    if not keep_download:
        os.remove(file_location)

def extract_file(target_path, filename):
	"""Extract file based on file extension
	target_path: string, location where data will be extracted
	filename: string, name of the file along with extension
	"""
	if filename == "" or filename is None:
		raise Exception(FILENAME_ERROR)

	file_location = get_file_location(target_path, filename)

	if filename.endswith(ZIP_EXTENSION):
		logging.info("Extracting zip file...")
		zipf = zipfile.ZipFile(file_location, 'r')
		zipf.extractall(target_path)
		zipf.close()
	elif filename.endswith(TAR_EXTENSION) or \
		 filename.endswith(TAR_GZ_EXTENSION) or \
		 filename.endswith(TGZ_EXTENSION):
		logging.info("Extracting tar file")
		tarf = tarfile.open(file_location, 'r')
		tarf.extractall(target_path)
		tarf.close()
	elif filename.endswith(GZ_EXTENSION):
		logging.info("Extracting gz file")
		out_file = file_location[:-3]
		with open(file_location, "rb") as f_in:
			with open(out_file, "wb") as f_out:
				shutil.copyfileobj(f_in, f_out)
	else:
		logging.info(UNKNOWN_FORMAT)

def get_filename(url):
	"""Extract filename from file url"""
	filename = os.path.basename(url)
	return filename

def get_file_location(target_path, filename):
	""" Concatenate download directory and filename"""
	return os.path.join(target_path, filename)

def copy_file(source_download_dir, local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    N = len(list_of_files)
    for file in tqdm(list_of_files, total=N, desc=f'copying file from {source_download_dir} to {local_data_dir}', colour="green"):
        src = os.path.join(source_download_dir, file)
        dest = os.path.join(local_data_dir, file)
        shutil.copy(src, dest)

def get_data(config_path):
    ## read config files
    config = read_yaml(config_path)
    image_download_url = config["source_download_url"]["image_download_url"]
    annotation_download_url = config["source_download_url"]["annotation_download_url"]
    target_path = config["local_data_dirs"]
    # Download Images
    download_dataset(url=image_download_url, target_path=target_path)
    # Download Annotation
    download_dataset(url=annotation_download_url, target_path=target_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        get_data(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
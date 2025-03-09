import os
import tarfile
import shutil
from tqdm import tqdm

SOURCE_DIR = "./Origin/"
TARGET_DIR = "./DATA/"


def un_tgz(filename, target):
    tar = tarfile.open(filename)
    os.makedirs(target, exist_ok=True)
    tar.extractall(target)
    tar.close()


if __name__ == "__main__":
    filelist = os.listdir(SOURCE_DIR)
    for item in tqdm(filelist):
        filename, extension = os.path.splitext(item)
        if (extension == ".tgz"):
            un_tgz(os.path.join(SOURCE_DIR, item), os.path.join(TARGET_DIR, filename))
        else:
            shutil.copy(os.path.join(SOURCE_DIR, item), os.path.join(TARGET_DIR, item))

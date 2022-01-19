import os
from pathlib import Path
import shutil
import subprocess

from src.config.settings import settings


DOWNLOAD_URL = "https://zenodo.org/record/5875246/files/ae-nlb-2021-model-checkpoints.zip?download=1"
OUTPUT_FILENAME = Path("ae-nlb-2021-model-checkpoints.zip")
TARGET_DIR = settings.SUBMISSION_VALIDATION_ROOT


def main():    
    print(f"This script will download and extract checkpoints for AESMTE3 into: {TARGET_DIR}")
    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
    os.chdir(TARGET_DIR)
    print()

    print("Downloading zip file from Zenodo")
    subprocess.run(["wget", f"--output-document={OUTPUT_FILENAME}", DOWNLOAD_URL])
    print()

    print(f"Extracting checkpoints into {TARGET_DIR}")
    subprocess.run(["unzip", OUTPUT_FILENAME])
    print()

    print(f"Organizing files in {TARGET_DIR}")
    for obj in Path(OUTPUT_FILENAME.stem).glob("*"):
        print(obj)
        shutil.move(str(obj), ".")
    Path(OUTPUT_FILENAME.stem).rmdir()
    print()

    print("Success!")


if __name__=="__main__":
    main()

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import torch
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

import warnings
import subprocess
warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)


import matplotlib.pyplot as plt
import numpy as np
import zarr
import pandas as pd
from tqdm import tqdm
from gamma_model_simulator import GammaModelSimulator
import os

import os
import numpy as np
import cupy as cp
import gamma.interface as rs
from multiprocessing import Process
import time

num = 1
os.environ['CUDA_VISIBLE_DEVICES'] = f'{num}'

class GammaModelSimulator:
    def __init__(self, input_data_dir, sim_dir_name, laser_file, VtkOutputStep=1., ZarrOutputStep=0.02, outputVtkFiles=False, verbose=True):
        self.input_data_dir = input_data_dir
        self.sim_dir_name = sim_dir_name
        self.laser_file = laser_file
        self.VtkOutputStep = VtkOutputStep
        self.ZarrOutputStep = ZarrOutputStep
        self.outputVtkFiles = outputVtkFiles
        self.verbose = verbose

        self.sim_itr = None

    def setup_simulation(self):
        self.sim_itr = rs.FeaModel(
                    input_data_dir=self.input_data_dir,
                    geom_dir=self.sim_dir_name,
                    laserpowerfile=self.laser_file,
                    VtkOutputStep=self.VtkOutputStep,
                    ZarrOutputStep=self.ZarrOutputStep,
                    outputVtkFiles=self.outputVtkFiles,
                    verbose=self.verbose)

    def run_simulation(self):
        if self.sim_itr:
            self.sim_itr.run()
        else:
            raise ValueError("Simulation is not setup yet. Call setup_simulation() first.")



class GammaSimulatorRunner:
    def __init__(self, input_data_dir, sim_dir_name, laser_file):
        self.simulator = GammaModelSimulator(
            input_data_dir=input_data_dir,
            sim_dir_name=sim_dir_name,
            laser_file=laser_file
        )

    def run(self):
        # Set up the simulation
        self.simulator.setup_simulation()

        # Execute the simulation
        self.simulator.run_simulation()

import os
import subprocess
import shutil

def compress_and_upload_with_progress(source_path, target_path):
    # Compress the file (assuming tar.gz compression)
    compressed_file_path = f"{source_path}.tar.gz"
    subprocess.run(['tar', '-czf', compressed_file_path, '-C', os.path.dirname(source_path), os.path.basename(source_path)], check=True)
    print(f"Compressed {source_path} to {compressed_file_path}")

    # Upload the compressed file
    subprocess.run(['rclone', 'copy', compressed_file_path, target_path], check=True)
    print(f"Uploaded {compressed_file_path} to {target_path}")
    
    # Delete the compressed file after uploading
    os.remove(compressed_file_path)
    print(f"Deleted local compressed file: {compressed_file_path}")

def delete_local_file(file_path):
    # Check if the path is a file or directory and delete accordingly
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
        print(f"Deleted directory: {file_path}")
    else:
        print(f"Path does not exist: {file_path}")

start_profile_id = 1
end_profile_id = 100
profile_ids = list(range(start_profile_id, end_profile_id + 1))

SIM_DIR_NAME = "custom_thinwall"
CLOUD_TARGET_BASE_PATH = "DED_DT:/custom_thinwall_174PH"

if __name__ == "__main__":
    INPUT_DATA_DIR = "/home/vnk3019/DT_DED/ME441_Projects/data"
    SIM_DIR_NAME = SIM_DIR_NAME
    BASE_LASER_FILE_DIR = "/home/vnk3019/DT_DED/ME441_Projects/laser_power_profiles/csv"
    CLOUD_TARGET_BASE_PATH = CLOUD_TARGET_BASE_PATH

    for i in profile_ids: ## Resuming from previous 
        laser_profile_filename_simulation = f"laser_profile_{i}"
        laser_profile_filename = f"laser_profile_{i}.zarr"
        LASER_FILE = os.path.join(BASE_LASER_FILE_DIR, laser_profile_filename_simulation)

        # Assuming the simulation process creates the LASER_FILE
        runner = GammaSimulatorRunner(INPUT_DATA_DIR, SIM_DIR_NAME, LASER_FILE)
        runner.run()

        # Define the cloud target path for this specific file
        #target_path_with_compressed = os.path.join(CLOUD_TARGET_BASE_PATH, f"compressed_{laser_profile_filename}.tar.gz")

        # Upload the file immediately after it's created and then delete it
        #compress_and_upload_with_progress(LASER_FILE, target_path_with_compressed)

        # Once uploaded, delete the original file and its compressed version from local storage
        #compressed_file_path = f"{LASER_FILE}.tar.gz"  # Assuming this naming convention
        #delete_local_file(LASER_FILE)  # Delete the original laser profile file
        # This assumes LASER_FILE without the .zarr extension, which may not be the case here




        # Define the cloud target path for this specific file for .zarr format
        target_path_with_compressed = os.path.join(CLOUD_TARGET_BASE_PATH, f"compressed_{laser_profile_filename}.zarr.tar.gz")

        LASER_FILE_ZARR = os.path.join(BASE_LASER_FILE_DIR, laser_profile_filename)

        # Upload the file immediately after it's created and then delete it for .zarr format
        compress_and_upload_with_progress(LASER_FILE_ZARR, target_path_with_compressed)
        compressed_file_path = f"{LASER_FILE}.tar.gz"  

        delete_local_file(compressed_file_path)  # Delete the compressed file

        zarr_file_path = f"{LASER_FILE}.zarr"  # Construct the .zarr file path
        delete_local_file(zarr_file_path)  # Delete the .zarr file

#CUDA_VISIBLE_DEVICES=1 python /home/vnk3019/DT_DED/ME441_Projects/GAMMA_data_generator_cube_GPU0.py

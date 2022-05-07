import os
import argparse
from slurm import ConfigParser
from slurm import Schedueler 

PYTHON_VERSION = "/shared/renard/conda/envs/CTNN/bin/python"
PROJECT_FOLDER = "/home/renard/Documents/CTNN_Policies_DERL"


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='', type=str)
args = parser.parse_args()


################ Train parameters
config = ConfigParser.load_config(args.config)
Schedueler.run_slurm(config, PYTHON_VERSION, PROJECT_FOLDER)

################ Running the script
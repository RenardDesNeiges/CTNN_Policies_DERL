import os
from slurm import ConfigParser
from slurm import Schedueler 
PYTHON_VERSION = "/shared/renard/conda/envs/CTNN/bin/python"
PROJECT_FOLDER = "/home/renard/Documents/CTNN_Policies_DERL"


################ Train parameters

NODE = "dev"
JOBNAME = "LTC_PPO"
MAX_TIME = 2
SCRIPT = "run-mujoco.py"
SBATCH_NAME = "run_cluster.sh"

# TODO : replace that by processing of the yaml file

argdict = { '--num-train-steps': '70000',
            '--policy-net': 'ltc',
            '--value-net': 'ltc',
            '--recurrent-policy': 'True',
            '--recurrent-value': 'True'}

env_dicts = [ {**argdict,
              '--env-id': 'InvertedPendulum-v2',
              '--logdir': 'logdir/ltc/InvertedPendulum',}, 
              {**argdict,
              '--env-id': 'InvertedDoublePendulum-v2',
              '--logdir': 'logdir/ltc/InvertedDoublePendulum',}, 
              {**argdict,
              '--env-id': 'Swimmer-v3',
              '--logdir': 'logdir/ltc/Swimmer',}, ]



foldername = Schedueler.create_output_folder(JOBNAME)


# TODO:  wrap those lines in a Schedueler method
print("Copying the script file for tracability")
os.system('cp {} {}/archive_{}'.format(SCRIPT,foldername,SCRIPT)) 
arguments = Schedueler._def_args(argdict)
header = Schedueler._slurm_head(foldername,cpu,mem,JOBNAME,MAX_TIME)
start_runs = ["{} -u {}/{} {} > {} \n".format(PYTHON_VERSION,PROJECT_FOLDER,SCRIPT,Schedueler._def_args(agd),foldername+'/'+agd['--env-id']+'.txt')  for agd in env_dicts]
script_path = Schedueler._sbatch(header, start_runs, foldername, SBATCH_NAME)
print("running sbatch")
os.system("sbatch {}".format(script_path))
################ Running the script
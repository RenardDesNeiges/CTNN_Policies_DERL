import os
import stat
from datetime import datetime
PYTHON_VERSION = "/shared/renard/conda/envs/nodeRL/bin/python"
PROJECT_FOLDER = "/home/renard/Documents/CTNN_Policies_DERL"

################ Utility functions

""" Generating the output folder """
def create_output_folder(_jobname):
  datestr = datetime.now().strftime("%Y-%m-%d-%H-%M")
  foldername = "/home/renard/Documents/experiments/R_{}_{}".format(datestr,_jobname)
  print("Created {} folder".format(foldername))
  os.mkdir(foldername)
  return foldername

""" Generating the sbatch script """
def _slurm_head(_foldername,cpu,mem,_jobname,_max_time):

  logpath = "{}/{}.out".format(_foldername,_jobname)

  header =  \
  "#!/bin/bash \n\
  \n#SBATCH --job-name={}\
  \n#SBATCH --output={}\
  \n#SBATCH --partition={}\
  \n#SBATCH --cpus-per-task={}\
  \n#SBATCH --mem={}GB\
  \n#SBATCH --nodes=1\
  \n#SBATCH --tasks-per-node=1\
  \n#SBATCH --time={}:00:00 \n\n".format(_jobname,logpath,NODE,cpu,mem,_max_time)

  return header

""" Writing the sbatch script """
def write_sbatch_shell(header, start_runs, foldername, _sbatch_name):
  print("writing sbatch script")
  # create the shell script
  sbatch_script = header
  for run in start_runs:
    sbatch_script += run

  script_path = "{}/{}".format(foldername,_sbatch_name)
  sbatch_script_file = open(script_path,"x")
  sbatch_script_file.write(sbatch_script)
  sbatch_script_file.close()

  return script_path

def def_arguments():

  arguments = ''

  return arguments

################ Train parameters

NODE = "dev"
JOBNAME = "LTC_PPO"
MAX_TIME = 2
SCRIPT = "test_script.py"
SBATCH_NAME = "run_cluster.sh"

if NODE == "dev":
    cpu = 24
    mem = 22
    if MAX_TIME > 2:
      raise Exception("MAX TIME = {} > 2h, which is the max time limit for the dev node".format(NODE))
elif NODE == "prod":
    cpu = 24
    mem = 22
    if MAX_TIME > 10:
      raise Exception("MAX TIME = {} > 10h, which is the max time limit for the prod node".format(NODE))
else:   
    raise Exception("Invalid node type : {}".format(NODE))

foldername = create_output_folder(JOBNAME)

print("Copying the script file for tracability")
os.system('cp {} {}/archive_{}'.format(SCRIPT,foldername,SCRIPT)) 

arguments = ''

header = _slurm_head(foldername,cpu,mem,JOBNAME,MAX_TIME)
scriptout = "{}/script_out.txt".format(foldername)
start_runs = ["{} -u {}/{} {} > {} \n".format(PYTHON_VERSION,PROJECT_FOLDER,SCRIPT,arguments,scriptout)]
script_path = write_sbatch_shell(header, start_runs, foldername, SBATCH_NAME)

################ Running the script
print("running sbatch")
# os.system("sbatch {}".format(script_path))
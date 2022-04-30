import os
import stat
from datetime import datetime
PYTHON_VERSION = "/shared/renard/conda/envs/nodeRL/bin/python"
PROJECT_FOLDER = "/home/renard/Documents/CTNN_Policies"

################ Train parameters

NODE = "dev"
JOBNAME = "LTC_PPO"
MAX_TIME = 10
SCRIPT = "tune_wrapper.py"
SCRIPT_NAME = "run_cluster.sh"

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

################ Generating the output folder

datestr = datetime.now().strftime("%Y-%m-%d-%H-%M")
foldername = "/home/renard/Documents/train_archives/R_{}_{}".format(datestr,JOBNAME)
print("Created {} folder".format(foldername))
os.mkdir(foldername)


################ Generating the sbatch script

scriptout = "{}/script_out.txt".format(foldername)
logpath = "{}/{}.out".format(foldername,JOBNAME)

header =  \
"#!/bin/bash \n\
\n\
#SBATCH --job-name={} \n\
#SBATCH --output={} \n\
\n\
#SBATCH --partition={} \n\
#SBATCH --cpus-per-task={} \n\
#SBATCH --mem={}GB \n\
#SBATCH --nodes=1 \n\
#SBATCH --tasks-per-node=1 \n\
#SBATCH --time={}:00:00 \n".format(JOBNAME,logpath,NODE,cpu,mem,MAX_TIME)


start_run = "{} -u {}/{} > {} \n".format(PYTHON_VERSION,PROJECT_FOLDER,SCRIPT,scriptout)

################ Writing the sbatch script

print("writing sbatch script")
# create the shell script
sbatch_script = header + start_run
script_path = "{}/{}".format(foldername,SCRIPT_NAME)
sbatch_script_file = open(script_path,"x")
sbatch_script_file.write(sbatch_script)
sbatch_script_file.close()

################ Archiving

print("Copying the script file for tracability")
os.system('cp {} {}/archive_{}'.format(SCRIPT,foldername,SCRIPT)) 

################ Running the script

print("running sbatch")
os.system("sbatch {}".format(script_path))
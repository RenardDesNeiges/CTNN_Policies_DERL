from yaml import load
import os
from datetime import datetime

class ConfigParser():
    @staticmethod
    def load(address):
        
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
            
        stream = open(address, 'rt')

        return load(stream.read(), Loader)

_config = ConfigParser.load('experiments/inverted_pendulum_LQR.yaml')
print(_config)


""" Write a slurm shell script and run it, save logs in an archive folder """
class Schedueler():
    """ Generating the output folder """
    @staticmethod
    def _create_output_folder(_jobname):
        datestr = datetime.now().strftime("%m%d%H%M")
        foldername = "/home/renard/Documents/experiments/R_{}_{}".format(_jobname,datestr)
        print("Created {} folder".format(foldername))
        os.mkdir(foldername)
        return foldername

    """ Generating the sbatch script """
    @staticmethod
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
    @staticmethod
    def _sbatch(header, start_runs, foldername, _sbatch_name):
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

    """ Write script arguments dict to string """
    @staticmethod
    def _def_args(argdict):
        arguments = ''
        for key in argdict:
            arguments += (key + ' ' + argdict[key] + ' ')

        return arguments
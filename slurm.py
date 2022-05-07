from yaml import load
import os
from datetime import datetime
from types import SimpleNamespace

class ConfigParser():
    @staticmethod
    def _load_yaml(address):
        # _config = ConfigParser.load('experiments/inverted_pendulum_LQR.yaml')
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
            
        stream = open(address, 'rt')

        return load(stream.read(), Loader)
    
    @staticmethod
    def load_config(address):
        yaml_dict = ConfigParser._load_yaml(address)
        args = SimpleNamespace(**yaml_dict)
        for key in args.run_args.keys():
            args.run_args[key] = {'--'+k:v for (k,v) in args.run_args[key].items()}
        
        if args.node == "dev":
            args.cpu = 24
            args.mem = 22
            if args.max_time > 2:
                raise Exception("MAX TIME = {} > 2h, which is the max time limit for the dev node".format(args.nodes))
        elif args.node == "prod":
            args.cpu = 40
            args.mem = 30
            if args.max_time > 10:
                raise Exception("MAX TIME = {} > 10h, which is the max time limit for the prod node".format(args.nodes))
        else:   
            raise Exception("Invalid node type : {}".format(args.nodes))
        
        return args


""" Write a slurm shell script and run it, save logs in an archive folder """
class Schedueler():
    """ Generating the output folder """
    @staticmethod
    def _create_output_folder(_jobname):
        datestr = datetime.now().strftime("%m%d%H%M")
        foldername = "/home/renard/Documents/experiments/{}_{}".format(_jobname,datestr)
        print("Created {} folder".format(foldername))
        os.mkdir(foldername)
        return foldername

    """ Generating the sbatch script """
    @staticmethod
    def _slurm_head(_foldername,args):

        head_args = (
            args.jobname,
            "{}/{}.out".format(_foldername,args.jobname),
            args.node,
            args.cpu,
            args.mem,
            args.max_time
        )

        header =  \
        "#!/bin/bash \n\
        \n#SBATCH --job-name={}\
        \n#SBATCH --output={}\
        \n#SBATCH --partition={}\
        \n#SBATCH --cpus-per-task={}\
        \n#SBATCH --mem={}GB\
        \n#SBATCH --nodes=1\
        \n#SBATCH --tasks-per-node=1\
        \n#SBATCH --time={}:00:00 \n\n".format(*head_args)
        return header

    """ Writing the sbatch script """
    @staticmethod
    def _sbatch(header, start_run, foldername, _sbatch_name):
        print("writing sbatch script")
        # create the shell script
        sbatch_script = header
        sbatch_script += start_run

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
            arguments += (key + ' ' + str(argdict[key]) + ' ')

        return arguments

    """ Write script arguments dict to string """
    @staticmethod
    def run_slurm(config, _py_path, _local_folder):

        foldername = Schedueler._create_output_folder(config.jobname)

        print("Copying the script file for tracability")
        os.system('cp {} {}/archive_{}'.format(config.script,foldername,config.script)) 

        header = Schedueler._slurm_head(foldername,config)
        
        start_run = ''
        for keys in config.run_args:
            arguments = Schedueler._def_args(config.run_args[keys])
            _buffer = foldername+'/'+keys+'.txt'
            start_run += "{} -u {}/{} {} > {} \n".format(_py_path,_local_folder,config.script,arguments,_buffer)

        script_path = Schedueler._sbatch(header, start_run, foldername, config.sbatch_name)

        print("running slurm")
        os.system("sbatch {}".format(script_path))
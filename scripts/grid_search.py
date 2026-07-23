import argparse 
import itertools 
import os 
import random
import string
import subprocess
import toml

from easydict import EasyDict


''' 
example script submission. only use the auto-submit flag to submit each job to a slurm cluster

python grid_search.py --config-path configs/config_default_3d.toml
python grid_search.py --config-path configs/config_default_3d.toml --auto-submit
'''

# list of parameter combinations to test - each must be the same length
# SET MAX_RES
PARAM_DICT = {'model.hiera_requires_grad': [True, True, True, False, False, False, False],
              'model.latent_dim': [64, 64, 64, 128, 64, 64, 64], 
              'model.corr_hidden_dim': [384, 384, 384, 384, 384, 1024, 1024], 
              'model.corr_output_dim': [256, 256, 256, 256, 256, 512, 512],
              'dataset.train.cams_to_sample': [[2, 6], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6]],
              'dataset.train.kpts_to_sample': [[64, 128], [128, 256], [256, 512], [256, 512], [128, 256], [256, 512], [512, 1024]],
              'training.optimizer.learning_rate': [3e-4, 3e-4, 3e-4, 3e-4, 3e-4, 3e-4, 3e-4], 
              'training.scheduler.milestones': [[], [], [], [], [], [], []], 
              'training.scheduler.gamma': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}

PARAM_DICT = {'model.hiera_requires_grad': [True, True, True, False, False, False, False],
              'model.latent_dim': [64, 64, 64, 128, 64, 64, 64], 
              'model.corr_hidden_dim': [384, 384, 384, 384, 384, 1024, 1024], 
              'model.corr_output_dim': [256, 256, 256, 256, 256, 512, 512],
              'dataset.train.cams_to_sample': [[2, 6], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6], [2, 6]],
              'dataset.train.kpts_to_sample': [[64, 128], [128, 256], [256, 512], [256, 512], [128, 256], [256, 512], [512, 1024]],
              'training.optimizer.learning_rate': [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4], 
              'training.scheduler.milestones': [[10000, 15000, 20000], [10000, 15000, 20000], [10000, 15000, 20000], [10000, 15000, 20000], [10000, 15000, 20000], [10000, 15000, 20000], [10000, 15000, 20000]], 
              'training.scheduler.gamma': [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]}

# # FOR TESTING 
# PARAM_DICT = {'training.n_iterations': [25], 
#               'wandb.project_name': ['posetail-test']}

def parse_args(): 
    '''
    parse command line arguments
    ''' 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', default = './configs/config_defaul_3d.toml')
    args = parser.parse_args()

    return args


def load_config(config_path, easy = True): 
    ''' 
    loads and returns the toml configuration file in which
    keys can be accessed.like.this
    '''
    with open(config_path, 'r') as toml_file:
        config = toml.load(toml_file)

    if easy: 
        config = EasyDict(config)

    return config


def save_config(config, outpath):
    '''
    save config to the provided outpath
    '''
    with open(outpath, 'w') as toml_file:
        toml.dump(config, toml_file)


def generate_uuid(n = 24):
    ''' 
    generates a unique id of the given length
    '''
    alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits
    uuid = [random.choice(alphabet) for _ in range(n)]
    uuid = ''.join(uuid)

    return uuid


def get_combinations(param_dict): 
    ''' 
    gets a list of dictionaries of all possible parameter combinations
    '''
    combinations = list(itertools.product(*param_dict.values()))
    param_dicts = [dict(zip(param_dict.keys(), c)) for c in combinations]
    print(len(param_dicts))

    return param_dicts


def get_combinations_simple(param_dict): 

    param_dicts = []
    keys = list(param_dict.keys())
    combinations = len(param_dict[keys[0]])
    
    for i in range(combinations):
        params = {k: param_dict[k][i] for k in keys}
        param_dicts.append(params)

    return param_dicts

def update_config(default_config, param_dict):
    ''' 
    given a default configuration file, updates 
    the parameters with the new parameters provided
    in param dict
    '''
    for ks, v in param_dict.items(): 

        d = default_config
        keys = ks.split('.')

        for k in keys[:-1]: 
            d = d[k]
        
        d[keys[-1]] = v

    return default_config


def main(args): 

    default_config = load_config(args.config_path)
    prefix =  os.path.dirname(args.config_path)

    if PARAM_DICT is not None:

        # update default config with new params
        config_paths = []
        param_dicts = get_combinations_simple(PARAM_DICT)

        for i, param_dict in enumerate(param_dicts): 

            new_config = update_config(default_config, param_dict)
            outpath = os.path.join(prefix, f'gs2_config{i}.toml')
            save_config(new_config, outpath)
            config_paths.append(outpath)
            print(f'creating config {outpath}')

    else: 
        config_paths = [args.config_path]

    return config_paths


if __name__ == '__main__':

    args = parse_args()
    config_paths = main(args)

    print('generated configs: ')
    print(config_paths)
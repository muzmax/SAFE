from pipeline.utils import run_train_contrastive,load_config
from pipeline.logger import setup_logger, LOGGER

import os
import yaml
import pprint

def mainSAFE(path_file):
    
    # load script params
    params = None
    with open(path_file, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    # create save dir and logger
    model_save_path = params['data']['save_dir']
    os.makedirs(model_save_path, exist_ok=True)
    dump = os.path.join(model_save_path, 'params-train.yaml')
    with open(dump, 'w') as f:
        yaml.dump(params, f)
    logger_path = os.path.join(model_save_path, "log.txt")
    setup_logger(out_file=logger_path)
    LOGGER.info('Configuration of training ...')
    # run configuration
    config = load_config('configs/train_safe','Config_Safe')
    conf = config(params)
    run_train_contrastive(conf)
    
if __name__ == "__main__":

    path_file = './configs/yaml/example_train_1ch.yaml'
    mainSAFE(path_file)
    # model is saved every 25 epochs and at the last one
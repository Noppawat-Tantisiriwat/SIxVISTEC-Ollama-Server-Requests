import yaml
import json
import csv


## Read config ymal

def read_yaml_config(config_path:str):

    """
    Read Config path for the experiment:

    Args:

        config_path: str 
            path of config to be read
    
    Return:

        config: dict
            configuration for the experiment

    Note: 
    YAML file sholud have the data fields in the following:
    
    `server`: server for the ollama api
    
    `model`: model of the experiment
    
    `data_config`:
        `path`: path to csv
        `start`: starting index
        `end`: ending index
    
    `prompt`: prompt of llms

    """

    try:

        with open(config_path, "r") as f:

            data = yaml.safe_load(f)

        return data

    except:
        print(f"Configuration Error at {config_path}")




if __name__ == "__main__":

    experiment_config = read_yaml_config("./config/llama3.1_8b_TH.yaml")

    print(f"API at server : {experiment_config['server']}")
    print(f"Model : {experiment_config['model']}")
    print(f"Using Data Configuration : {experiment_config['data_config']}")
    print(f"Using Prompt : {experiment_config['prompt']}")
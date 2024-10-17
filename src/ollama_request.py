import pandas as pd
import numpy as np
import re
import json

import matplotlib.pyplot as plt
import seaborn as sns

import requests
import ast

import os
import os.path as op

from tqdm import tqdm

import time


from config import read_yaml_config

# -------- Functions --------

### define function -> get llm generated output

def get_llm_generation(
    server,
    text,
    instruction=None,
    model=None
):
    
    """
    Get the LLMs text generation result form ollama deployed server
    """

    # spinner = Halo(
    #     text='Generating ...',
    #     spinner={
    #     'interval': 100,
    #     'frames': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    #     }
    # )

    url = "http://localhost:11434/api/generate"

    headers = {
        "Content-Type" : "application/json"
    }
    if instruction:

        # text = instruction + "\n\n" + text +  "[\INST]"
        text = instruction + "\n\n" + text
    
    
    if model:

        data = {
            "model" : f"{model}",
            # "model" : "taozhiyuai/openbiollm-llama-3:70b-q6_k",
            # "model" : "llava:34b",
            "prompt" : f"{text}",
            "stream" : False,
            "options":{"seed":0, "temperature":0}
        }

    else:

        data = {
            "model" : "taozhiyuai/openbiollm-llama-3:70b-q6_k",
            # "model" : "llava:34b",
            "prompt" : f"{text}",
            "stream" : False,
            "options":{"seed":0, "temperature":0}
        }

    
    

    ## Getting Response
    # spinner.start()
    response = requests.post(server, headers=headers, data=json.dumps(data))
    # spinner.stop()
    # Status Code Check

    if response.status_code == 200:

        response_text = response.text
        data = json.loads(response_text)
        actual_response = data["response"]
        # print(actual_response)

        return actual_response

    else:

        print("Error:" , response.status_code, response.text)


def extract_icd_list(text_from_request):

    try:
        
        str_dict = str(re.findall(r"(\{[\s\S]*\})", text_from_request)[-1])

        result_list = re.findall(r"([^\[,]*),", str_dict)

        if len(result_list) == 0:

            result_list = re.findall(r"\[([^\[]*)\]", str_dict)

        # icd_list = ast.literal_eval(str_dict)


        return result_list
    
    except Exception as e:

        print(e)

        return list()

## Selecting TAG: <output> ... </output>

def reflection_output_extract(full_reflection_output):

    ## tag detection <output> </output>
    try:
        output = re.findall(pattern=r"<output>([\S\s]*)$", string=full_reflection_output)

        result_list = extract_icd_list(output[0])

        return result_list

    except Exception as e:
        
        print(e)

        return list()

    
def run_llm_prompt_experiment(
        data_df: pd.DataFrame,
        server: str="http://localhost:11434/api/generate",
        model: str="llama3.1:8b",
        instruction: str=None,
        root: str="./",
        result_path: str="res",
):

    storage = {}

    for i, row in tqdm(data_df.iterrows(), total=len(data_df)):

        # Get Medical Record

        record = row["translate_patient_history_eng"]

        # Get Label

        label_list = ast.literal_eval(row["target"].replace(" ", ", "))

        # Get Request

        text_result = get_llm_generation(server=server, text=record, model=model, instruction=instruction)

        # Extract Request

        if "reflection" in model:

            reflection_result = reflection_output_extract(text_result)
            icd_list = extract_icd_list(reflection_result)
        
        else:

            icd_list = extract_icd_list(text_result)


        # Save Label + Request

        storage[i] = {
            
            "Input" : {
                
                "Record" : record,
                
                "Instruction" : instruction,
                
                "Model" : model
            },
            
            "Output" : {
                
                "Results" : text_result,
                
                "Evaluate" : {
            
                    "True" : label_list,
                
                    "Pred" : icd_list
                },
            }
        }

        with open(os.path.join(root, result_path, f"test_result-{model.split('/')[0]}_icd_test_Reflection_TH_sample.json"), "w") as f:
            
            json.dump(storage, f, indent=5)

    
    
class OllamaAPIPromptExperiment():

    def __init__(self, config_path):

        self.config = read_yaml_config(config_path)

        # --- Get Config ---

        self.server = self.config["server"]
        self.model = self.config["model"]
        self.root = self.config["root"]
        self.data_config = self.config["data_config"]
        self.result_config = self.config["result_config"]
        self.instruction = self.config["prompt"]

    def run_experiment(self):

        result_path = self.result_config["path"]
        data_path = self.data_config["path"]

        text_field = self.data_config["text"]
        label_field = self.data_config["label"]

        # --- Load Data with data field ---
        if ".json" in data_path:

            with open(data_path, "r") as f:

                testing_data = json.load(f)

            self.test_texts = testing_data[text_field]

            self.test_labels = testing_data[label_field]



        elif ".csv" in data_path:
            
            testing_data = pd.read_csv(data_path)

            self.test_texts = testing_data[text_field].to_numpy()

            self.test_labels = testing_data[label_field].to_numpy()

        elif ".xlsx" in data_path:

            testing_data = pd.read_excel(data_path)

            self.test_texts = testing_data[text_field].to_numpy()

            self.test_labels = testing_data[label_field].to_numpy()


        assert len(self.test_texts) == len(self.test_labels)


        # --- Run Experiment ---

        # Saving dict

        self.result_storage = {}

        # start inference
        for i, (text, label) in tqdm(enumerate(zip(self.test_texts, self.test_labels)), total=len(self.test_texts)):

            text_result, icd_list, label = self.inference_loop(text, label)

            self.result_storage[i] = {
        
                                        "Input" : {
                                            
                                            "Record" : text,
                                            
                                            "Instruction" : self.instruction,
                                            
                                            "Model" : self.model
                                        },
                                        
                                        "Output" : {
                                            
                                            "Results" : text_result,
                                            
                                            "Evaluate" : {
                                        
                                                "True" : label,
                                            
                                                "Pred" : icd_list
                                            },
                                        }
                                    }
            with open(os.path.join(result_path, f"test_result-{self.model.split('/')[0]}_icd_test.json"), "w") as f:
                json.dump(self.result_storage, f, indent=5)
        
        # --- Load JSON result after experiment is run ---

        with open(os.path.join(result_path, f"test_result-{self.model.split('/')[0]}_icd_test.json")) as f:

            experiment_result = json.load(f)

        return experiment_result
    

    def inference_loop(self, text, label):

        # Get request
        text_result = get_llm_generation(
            server=self.server,
            text=text,
            model=self.model,
            instruction=self.instruction
        )

        # Extract Request

        icd_list = extract_icd_list(text_result)

        
        return text_result, icd_list, label




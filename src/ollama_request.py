import requests
import re
import ast
import json

from halo import Halo

import csv

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

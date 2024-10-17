from ollama_request import OllamaAPIPromptExperiment
import os
import argparse



 # Getting parser
parser = argparse.ArgumentParser(
    description="Experimental Code for Ollama Requests for ICD codes labeling"
)

# Adding `config` argument
parser.add_argument(
    "--config", help='''YAML Configuration of the Experiment.
The config file must be stored in  `config` folder.
The structure of the file:




`server: "http://localhost:11434/api/generate"

model: "llama3.1:8b"

root: "."

data_config:
    path: "data/test_data/test_dataset.csv"

result_config:
    path: "res"

prompt: | 
    Instructions:

    1. Read the following medical record carefully.

    2. Extract all relevant ICD codes (including ICD-10 for diagnosis and ICD-9 procedural codes for procedures) that correspond to the medical conditions, diagnoses, symptoms, findings, and procedures mentioned in the text.

    3. Ensure patient confidentiality by not including any personally identifiable information in your response.

    4. Output only the list of ICD codes in a single combined list within a JSON object, using the following exact format:

    json
    Copy code
    {
    "ICD": ["Code1", "Code2", "Code3", ..., "CodeN"]
    }
    5. Do not include any additional text, explanations, or descriptions. Only provide the JSON output as specified.

    Medical Record:
    `
                    
                        '''
)

if __name__ == "__main__":

    selected_config = parser.parse_args().config


    CONFIG = os.path.join("config", selected_config)
    


    experiment_agent = OllamaAPIPromptExperiment(
        config_path=CONFIG
    )


    # RUN EXPERIMENT
    print("\n\n")
    print("-"*100)
    print("\n\n")

    print("Starting Experiment ... ")
    print("\n\n")
    print("-"*100)
    print("\n\n")

    result = experiment_agent.run_experiment()

    print("Experiment Complete ...")

    print("\n\n")
    print("-"*100)
    print("\n\n")

    print("Example Results ...")
    print("\n\n")

    print(result["1"]["Output"]["Results"])
# ollama-server-requests
This semi-private repo is for ollama LLM experiment (used under Siriraj VISTEC MOU)

# Script

1. Transfer the dataset to `data`  folder (create subfolder inside `data` is recommended)


2. Create YAML configuration consisting of fields as example
```yaml
server: "http://localhost:11434/api/generate"

model: "OLLAMA_MODEL"

root: "."

data_config:
    path: "data/DATA_FOLDER/DATA_FILE"
    
    text: "TEXT"

    label: "TARGET"

result_config:
    path: "res"

prompt: | 
    ...
    
```

3. Run experiment using `main.py`

```bash
python src/main.py --config "llama3.1_70b_TH.yaml"
```

4. Retrieve the result from `res` folder 

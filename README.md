## Project: Personalized Real Estate Agent

We use `gemini` for the real estate data and use the data to build a personalized real estate agent. The agent will provide the user with a list of properties that are most likely to be of interest to them based on their preferences.

- Use the `gemini` models to generate real estate data.
- Integrate with COMET ML for experiment tracking and LLM Ops.
- Integrate the kubeflow pipeline to automate the data processing and model training process.
- Apply `gradio` library to quickly create user interfaces for machine learning models.


## Project Setup

### 1. Create a new virtual environment

```bash
python3 -m venv env
```

### 2. Activate the virtual environment

```bash
source env/bin/activate
```

### 3. Install the required packages (can operate in the jupyter notebook code cell)

```bash
pip3 install -r requirements.txt
```

### 4. Save the current environment packages to requirements.txt

After installing all the necessary packages, you can save the current environment's packages to `requirements.txt` by running:

```bash
pip freeze > requirements.txt
```

This will generate a `requirements.txt` file with all the installed packages and their versions, which can be used to recreate the environment in the future.

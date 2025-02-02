## Project: Personalized Real Estate Agent

We use `gpt-3.5-turbo-instruct` for the real estate data and use the data to build a personalized real estate agent. The agent will provide the user with a list of properties that are most likely to be of interest to them based on their preferences.

- Use the `gpt-3.5-turbo-instruct` models to generate real estate data.
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

## Software Design Strategies:
 - The `create_listing_prompt` function generates the prompt for a given `Listing` object, making it reusable and clear.
 - The `Listing` class is defined using `Pydantic`, which provides validation and type checking, ensuring that the data is structured correctly.
 - The `few-shot examples` are directly linked to the Listing model, making it easier to manage and modify.
 - Comebine the `few_shot_prompt` and the `example_prompt` into the `chain_of_thoughts` (CoT) concept that processes the questions and generates responses in a straightforward manner. This structure allows for efficient generation of real estate listings while maintaining clarity and ease of use.

- The `generate_images` function uses the `CompVis/stable-diffusion-v1-4` model run in `Apple Silicon (M1/M2)` to efficiently generate images in `batches` from generated real estate listings based on the provided prompts which assembled from the dataframes `df` row by row in batches defined by `batch_size` and saved to the `generated_images` directory.

## Kubeflow Pipeline Deployment:

## Gradio Interactive Interface Demo:
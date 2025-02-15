## Project: Personalized Real Estate Agent

We use `gpt-3.5-turbo-instruct` for the real estate data and use the data to build a personalized real estate agent. The agent will provide the user with a list of properties that are most likely to be of interest to them based on their preferences.

- Use the `gpt-3.5-turbo-instruct` models to generate real estate data.
- Integrate with `COMET ML` for experiment tracking and LLM Ops.
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

### 5. Run the main script

```bash
python3 HomeMatch.py 
```

- You can also run the jupyter notebook script `HomeMatch.ipynb` to interact with the code cell by cell.

## Software Design Strategies

- The `create_listing_prompt` function generates the prompt for a given `Listing` object, making it reusable and clear.
- The `Listing` class is defined using `Pydantic`, which provides validation and type checking, ensuring that the data is structured correctly.
- The `few-shot examples` are directly linked to the Listing model, making it easier to manage and modify.
- Comebine the `few_shot_prompt` and the `example_prompt` into the `chain_of_thoughts` (CoT) concept that processes the questions and generates responses in a straightforward manner. This structure allows for efficient generation of real estate listings while maintaining clarity and ease of use.

- The `evaluate_generated_listings` function automates the process of calculating `BLEU` and `ROUGE` scores, making it easy to assess the quality of generated text against reference listings.

- The `generate_images` function uses the `CompVis/stable-diffusion-v1-4` model run in `Apple Silicon (M1/M2)` to efficiently generate images in `batches` from generated real estate listings based on the provided prompts which assembled from the dataframes `df` row by row in batches defined by `batch_size` and saved to the `generated_images` directory.

- We use `OpenCLIPEmbeddings`to perform `Multimodal Embeddings` generate embeddings for both text and images using the CLIP model. By leveraging a multimodal approach, we can effectively represent and compare different types of data (textual descriptions and visual content) in a unified manner.

- We apply `Chroma` which is designed to store embeddings efficiently, allowing for fast retrieval and `similarity searches`. By storing both text and image embeddings in the same database, we can easily perform queries that consider both modalities.  

With the ability to search and retrieve listings based on both text and images, users can have a more intuitive and satisfying experience. They can find properties that match their preferences more easily, whether they are browsing through images or reading descriptions.

## Gradio Interactive Interface Demo

![Gradio Interface](demo_images/MLWebAppDemo.gif)

## Experiment Tracking with COMET ML
![COMET ML](demo_images/COMET.gif)

## Future Work

- Carefully design our prompts to guide the model through the reasoning process. Such as, we might prompt the model to consider each attribute of the listing step-by-step.
- Test the outputs generated using CoT reasoning and iterate on our prompts and examples to improve the quality of the generated data.

- Applying other performance metrics such as `METEOR` and `BERTScore` to evaluate the quality of the generated listings.
- If we need the ability to efficiently look up neighborhoods then we refactor the code efficiently uses [`tf.lookup.StaticHashTable`](https://www.tensorflow.org/api_docs/python/tf/lookup/StaticHashTable) to manage neighborhood lookups, which can improve performance, especially when dealing with large datasets.

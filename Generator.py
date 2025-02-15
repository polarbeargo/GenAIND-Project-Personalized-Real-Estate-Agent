from langchain_community.llms import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field, NonNegativeInt
from comet_ml import Experiment
import torch
from diffusers import StableDiffusionPipeline
from langchain.output_parsers import PydanticOutputParser
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import os
from dotenv import load_dotenv
import pandas as pd
from typing import List

# Load environment variables
load_dotenv('my_config.env')

# API configuration
API_KEY = os.getenv('API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")
COMET_API_KEY = os.getenv("COMET_API_KEY")


# Initialize the experiment
experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name="real-estate-agent",
    workspace="polarbeargo",
    log_code=True,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

instruction = """
Generate five realistic real estate listings from a wide range of neighborhoods.
"""

template = \
"""
Here is the template of real estate listing:

Neighborhood: Mountain View
Price: $650,000
Bedrooms: 5
Bathrooms: 4
House Size: 3000 sqft
Description: Spacious family home with breathtaking views of the mountains and a large backyard for outdoor entertaining.
Neighborhood Description: Mountain View is known for its scenic landscape and outdoor activities, making it the ideal location for nature lovers and adventure seekers.
"""
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7, api_key=openai_api_key, max_tokens = 500)
image_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# For Apple Silicon (M1/M2) replace mps to device if executing on other devices
image_model.to("mps")

image_dir = "generated_images"
os.makedirs(image_dir, exist_ok=True)

# Define the Listing data model
class Listing(BaseModel):
    neighborhood: str = Field(description="The neighborhood where the property is located.")
    price: NonNegativeInt = Field(description="The price of the property in USD.")
    bedrooms: NonNegativeInt = Field(description="The number of bedrooms in the property.")
    bathrooms: NonNegativeInt = Field(description="The number of bathrooms in the property.")
    house_size: NonNegativeInt = Field(description="The size of the property in square feet.")
    description: str = Field(description="A brief description of the property.")
    neighborhood_description: str = Field(description="A description of the neighborhood where the property is located.")

class Listings(BaseModel):
    listing: List[Listing] = Field(description="List of available real estate listings.")
    

def create_listing_prompt(listing: Listing) -> str:
    return f"""
    Neighborhood: {listing.neighborhood}
    Price: ${listing.price}
    Bedrooms: {listing.bedrooms}
    Bathrooms: {listing.bathrooms}
    House Size: {listing.house_size} sqft
    Description: {listing.description}
    Neighborhood Description: {listing.neighborhood_description}
    """

# Define few-shot examples
examples = [
    {
        "question": "Generate a listing for a 3-bedroom house in downtown.",
        "answer": Listing(
            neighborhood="Downtown",
            price=500000,
            bedrooms=3,
            bathrooms=2,
            house_size=1500,
            description="A beautiful 3-bedroom house located in the heart of downtown with modern amenities.",
            neighborhood_description="Downtown is vibrant and bustling, with plenty of restaurants, shops, and parks."
        )
    },
    {
        "question": "Create a listing for a luxury apartment in the suburbs.",
        "answer": Listing(
            neighborhood="Suburbia",
            price=750000,
            bedrooms=2,
            bathrooms=2,
            house_size=1200,
            description="A luxurious apartment featuring high-end finishes and spacious living areas.",
            neighborhood_description="Suburbia offers a peaceful environment with great schools and family-friendly parks."
        )
    }
]

parser = PydanticOutputParser(pydantic_object=Listings)

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="{question}\n{answer}",
    partial_variables={"format_instructions": parser.get_format_instructions},
)

few_shot_prompt = FewShotPromptTemplate(
    examples=[{"question": ex["question"], "answer": create_listing_prompt(ex["answer"])} for ex in examples],
    example_prompt=example_prompt,
    suffix="Use these examples to generate a listing for the following question: {input}",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions},
)

full_prompt = few_shot_prompt.format(sample=template, input=instruction)
response = llm(full_prompt)
print(f"Raw Response: {response}")

# Split the string into individual listings
listings = response.strip().split('\n\n')
print(listings)
data = []

for listing in listings:

    # Remove the integer before 'Neighborhood'
    listing = listing.split('. ', 1)[1]

    # Split the listing into lines
    lines = listing.split('\n')
    listing_data = {}
    
    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)  
            listing_data[key.strip()] = value.strip()
    
    data.append(listing_data)

df = pd.DataFrame(data)

df.rename(columns={
    'Neighborhood': 'neighborhood',
    'Price': 'price',
    'Bedrooms': 'bedrooms',
    'Bathrooms': 'bathrooms',
    'House Size': 'house_size',
    'Description': 'description',
    'Neighborhood Description': 'neighborhood_description'
}, inplace=True)

# Convert price to integer and house_size to integer (removing ' sqft')
df['price'] = df['price'].replace({'\$': '', ',': ''}, regex=True).astype(int)
df['house_size'] = df['house_size'].replace({' sqft': ''}, regex=True).fillna(0).astype(int)
df.to_csv('generated_real_estate_data.csv', index_label = 'id')

# Batch generation of images based on the DataFrame
def generate_images(df, batch_size=2):
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        for idx, row in batch.iterrows():
            prompt = f"A {row['bedrooms']}-bedroom house in {row['neighborhood']}. {row['neighborhood_description']}"
            image = image_model(prompt, num_inference_steps=50).images[0]
            image_path = os.path.join(image_dir, f"{row['neighborhood']}_{row['bedrooms']}_bedroom.png")
            image.save(image_path)
            print(f"Generated image saved at: {image_path}")

generate_images(df)

def evaluate_generated_listings(generated: str, reference: str) -> dict:
    
    # Calculate BLEU score
    smoothing_function = SmoothingFunction()
    bleu_score = sentence_bleu([reference.split()], generated.split(), smoothing_function=smoothing_function.method1)
    
    # Calculate ROUGE score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    
    return {
        "bleu": bleu_score,
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }

reference_listings = [
    "Neighborhood: Beachfront\nPrice: $1,200,000\nBedrooms: 4\nBathrooms: 3\nHouse Size: 3000 sqft\nDescription: A stunning beachfront property with ocean views.\nNeighborhood Description: Beachfront is known for its luxurious homes and vibrant community.",
    "Neighborhood: City Center\nPrice: $800,000\nBedrooms: 2\nBathrooms: 1\nHouse Size: 900 sqft\nDescription: A modern apartment located in the heart of the city.\nNeighborhood Description: City Center is bustling with activity and offers a variety of amenities.",
    "Neighborhood: Mountain View\nPrice: $650,000\nBedrooms: 5\nBathrooms: 4\nHouse Size: 3000 sqft\nDescription: Spacious family home with breathtaking views of the mountains and a large backyard for outdoor entertaining.\nNeighborhood Description: Mountain View is known for its scenic landscape and outdoor activities, making it the ideal location for nature lovers and adventure seekers.",
]

# Evaluate the generated listings
for generated, reference in zip(response, reference_listings):
    metrics = evaluate_generated_listings(generated, reference)
    print(f"Generated Listing: {generated}")
    print(f"Evaluation Metrics: {metrics}")
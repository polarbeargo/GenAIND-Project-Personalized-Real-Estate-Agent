import os
import PIL
import pandas as pd
from Generator import image_dir
from langchain.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain.chains import RetrievalQA
from Generator import llm

df = pd.read_csv('generated_real_estate_data.csv')
idx = [{'id':i} for i in range(len(df.index))]
image_paths = []
images = []
texts = []
text_template = """
Neighborhood: {}
Price: {}
Bedrooms: {}
Bathrooms: {}
House Size: {}

Description: {}
Neighborhood Description: {}
"""

for i, row in df.iterrows():
    image_path = os.path.join(image_dir, f"{row['neighborhood']}_{row['bedrooms']}_bedroom.png")
    image_paths.append(image_path)
    images.append(PIL.Image.open(image_path))
    texts.append(text_template.format(row['neighborhood'], row['price'], row['bedrooms'], row['bathrooms'], row['house_size'], row['description'], row['neighborhood_description']))
    
embeddings = OpenCLIPEmbeddings()
clip_db = Chroma(collection_name="real_estate_listings", embedding_function=embeddings)

clip_db.add_texts(
    texts=texts,
    metadatas=idx
)

clip_db.add_images(
    uris=image_paths,
    metadatas=idx
)

questions = [   
                "How big do you want your house to be?",
                "What are 3 most important things for you in choosing this property?", 
                "Which amenities would you like?", 
                "Which transportation options are important to you?",
                "How urban do you want your neighborhood to be?",   
            ]
answers = [
    "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and convenient shopping options.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters.",
]

query = f"""
{questions[0]} {answers[0]}
{questions[1]} {answers[1]}
{questions[2]} {answers[2]}
{questions[3]} {answers[3]}
"""
print("Constructed Query:", query)
use_chain_helper = False

if use_chain_helper:
    rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=clip_db.as_retriever())
    print(rag.run(query))
else:
    similar_docs = clip_db.similarity_search(query, k=3)
    print("Similar Documents:", similar_docs)

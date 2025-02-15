import gradio as gr 
from PIL import Image
from CLIP import clip_db, image_paths

def retrieve_data(query):
    """Retrieve similar texts and images based on the query."""
    similar_texts = clip_db.similarity_search(query, k=3)
    print("Similar Documents:", similar_texts)
   
    similar_images_ids = [result.metadata.get('id', 'id not found') for result in similar_texts] 
    similar_images = [Image.open(image_paths[int(id)]) for id in similar_images_ids]
    print("Similar Images:", similar_images)

    output_texts = "\n".join([result.page_content for result in similar_texts])
    return output_texts, similar_images

def generate_app():
    with gr.Blocks() as demo:
        gr.Markdown("""
        # Your top 3 Real Estate Listings
        1. Fill out your query based on customer preferences.
        2. Click on the "Search" button to fetch the top 3 real estate listings based on your input preferences.
        3. Review the listings displayed below. Each listing includes an image, price, location, and a brief description.
        4. If you want to refine your search, adjust your preferences and click "Search" again.
        """)

        query_input = gr.Textbox(label="Enter your query about real estate:")
        search_button = gr.Button("Search")
    
        output_text = gr.Textbox(label="Retrieved Listings", interactive=False)
        output_image = gr.Gallery(label="Retrieved Images", show_label=True)

        search_button.click(fn=retrieve_data, inputs=query_input, outputs=[output_text, output_image])

    demo.launch()
    return demo

if __name__ == "__main__":
    interface = generate_app()
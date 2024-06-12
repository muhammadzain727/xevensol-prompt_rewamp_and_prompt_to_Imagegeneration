import os
import requests
from PIL import Image
import io
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import openai
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate image using OpenAI DALL-E
def generate_image_openai(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
        quality="standard",
    )
    image_url = response.data[0].url
    image = Image.open(io.BytesIO(requests.get(image_url).content))
    return image

# Streamlit application
def main():
    # Set page title
    st.title("Your Text to Image Generator")

    # Brief detail about the system
    st.write("""
        Welcome to the Text to Image Generator based on DALL-E 3! This tool helps you generate high-quality prompts using GPT-3.5 API and QA chain for image generation.
        Simply enter your initial prompt, and our system will provide three improved prompts to enhance the quality of generated images.
    """)

    if st.button("How to use ?"):
        st.write("## User Manual ")
        st.write("""
            1. **Enter Prompt:** Describe the image you want.
            2. **Generate Improved Prompts:** Copy suggestions.
            3. **Generate Image:** Use a suggestion or your own prompt.
        """)
        st.button("Close")

    # Input chat field to take user prompt
    user_prompt = st.text_input("Your Initial Prompt:")

    # Output text field for improved prompts
    if st.button("Generate Improved Prompts"):
        if user_prompt:
            results = prompt_rewamper(user_prompt)
            st.write("Improved Prompts:")
            st.write(results)

    # Button to generate image based on user input
    img_prompt = st.text_input("Text to Image Prompt:")
    if st.button("Generate Image"):
        if img_prompt:
            with st.spinner("Generating image..."):
                try:
                    image = generate_image_openai(img_prompt)
                    st.image(image, caption=f"Generated Image for: {img_prompt} using OpenAI DALL-E 3")
                except Exception as e:
                    st.error(f"Error generating image: {e}")
        else:
            st.error("Please enter a prompt.")

# Function to rewarp prompts
def prompt_rewamper(user_prompt):
    try:
        template = """
        As an AI prompt modifier for better image generation, your task is to generate 3 improved prompts based on the following user prompt to enhance image generation quality:
        
        User Prompt: "{user_prompt}"
        
        Improved Prompts:
        1. 
        2. 
        3. 
        """

        prompt = ChatPromptTemplate.from_template(template)
        setup_and_retrieval = RunnableParallel(
            {"user_prompt": RunnablePassthrough()}
        )
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        model = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=OPENAI_API_KEY)
        output_parser = StrOutputParser()
        rag_chain = (
            setup_and_retrieval
            | prompt
            | model
            | output_parser
        )

        response = rag_chain.invoke(user_prompt)

        return response
    except Exception as ex:
        return str(ex)

if __name__ == "__main__":
    main()
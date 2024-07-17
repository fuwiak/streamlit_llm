import streamlit as st
from transformers import pipeline

# Load the model and tokenizer
@st.cache_resource
def load_pipeline():
    model_name = "rubra-ai/Mistral-7B-Instruct-v0.3"
    pipe = pipeline("text-generation", model=model_name)
    return pipe

pipe = load_pipeline()

st.title("Mistral-7B-Instruct Model Demo")
st.write("Enter text and get a response from the Mistral-7B-Instruct model.")

# User input
user_input = st.text_area("Input Text", height=200)

if st.button("Generate Response"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Generate response
        messages = [
            {"role": "user", "content": user_input},
        ]
        response = pipe(messages, max_new_tokens=150, do_sample=True, top_p=0.95, top_k=60)[0]['generated_text']
        st.text_area("Response", response, height=200)

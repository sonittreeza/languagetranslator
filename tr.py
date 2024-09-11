import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer from MarianMT
@st.cache_resource
def load_model_and_tokenizer(src_lang, tgt_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

# Function to translate text
def translate_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Streamlit app layout
st.title("Language Translator")

# Define the source and target languages (English and another language)
languages = {
    'French': 'fr',
    'German': 'de',
    'Spanish': 'es'
}

# Input form for translation
option = st.selectbox('Choose the translation direction', ['To English', 'From English'])

# Select a language other than English
lang_choice = st.selectbox('Select language', list(languages.keys()))

input_text = st.text_area("Enter text")

# Determine the source and target language
src_lang, tgt_lang = ('en', languages[lang_choice]) if option == 'From English' else (languages[lang_choice], 'en')

# Load the model and tokenizer based on selected languages
if st.button("Translate"):
    model, tokenizer = load_model_and_tokenizer(src_lang, tgt_lang)
    translated_text = translate_text(model, tokenizer, input_text)
    st.write("Translated Text:")
    st.write(translated_text)

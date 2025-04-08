import os
os.environ["THINC_BACKEND"] = "cpu"  # ✅ Force CPU backend for spaCy

import streamlit as st
import spacy
from spacy.pipeline import EntityRuler
from spacy import displacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to load lexicon entries as lowercase patterns
def load_lexicon(file_path, label):
    with open(file_path, 'r', encoding='utf-8') as f:
        entries = f.read().splitlines()
    return [{"label": label, "pattern": [{"LOWER": entry.lower()}]} for entry in entries if entry.strip()]

# Create EntityRuler and add patterns before spaCy's NER
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Load lexicon patterns (case-insensitive)
patterns = []
patterns += load_lexicon("nlp/lexicons/indian_cities.txt", "GPE")
patterns += load_lexicon("nlp/lexicons/indian_names.txt", "PERSON")
patterns += load_lexicon("nlp/lexicons/indian_organizations.txt", "ORG")
patterns += load_lexicon("nlp/lexicons/computer_languages.txt", "PROG_LANG")
patterns += load_lexicon("nlp/lexicons/music_instruments.txt", "INSTRUMENT")
patterns += load_lexicon("nlp/lexicons/sports.txt", "sport")
patterns += load_lexicon("nlp/lexicons/automobiles.txt", "automobile")
patterns += load_lexicon("nlp/lexicons/food.txt", "food")

ruler.add_patterns(patterns)

# Streamlit App UI
st.set_page_config(page_title="NER BY ABHINAV RANA", layout="centered")
st.title("NER")
st.markdown("Enter a news article or sentence to extract named entities based on Indian lexicons and spaCy's model (case-insensitive).")

# Text input
user_input = st.text_area("📝 Enter news text below:", height=200)

# Process input
if user_input:
    doc = nlp(user_input)

    # Show named entities
    st.subheader("📌 Named Entities")
    if doc.ents:
        for ent in doc.ents:
            st.markdown(f"- **{ent.text}** → `{ent.label_}`")
    else:
        st.write("No entities found.")

    # Visualize with displacy
    st.subheader("🔍 Entity Visualization")
    html = displacy.render(doc, style="ent", page=True)
    st.components.v1.html(html, scrolling=True, height=300)

import streamlit as st
# Theme toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
if dark_mode:
    st.markdown("""
        <style>
        body {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .stApp {
            background-color: #1e1e1e;
        }
        .title {
            color: #f1c40f;
        }
        .tab-content {
            background-color: #2c2c2c;
            color: #ffffff;
        }
        .stTextInput>div>div>input {
            background-color: #333;
            color: white;
        }
        .stButton>button {
            background-color: #f39c12;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {
            background-color: #f6fafd;
        }
        .stApp {
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
            padding-bottom: 10px;
        }
        .tab-content {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .stTextInput>div>div>input {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #0f5c96;
        }
        .stSuccess {
            border-left: 5px solid #2ecc71;
            background-color: #ecf9f1;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    body {
        background-color: #f6fafd;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        padding-bottom: 10px;
    }
    .tab-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .stTextInput>div>div>input {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0f5c96;
    }
    .stSuccess {
        border-left: 5px solid #2ecc71;
        background-color: #ecf9f1;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Improved disease prediction from symptoms
def identify_disease(symptoms):
    prompt = (
        f"You are a medical assistant AI. Your task is to analyze symptoms and predict the most likely disease.\n\n"
        f"Example 1:\n"
        f"Symptoms: fever, cough, sore throat\n"
        f"Disease: Common Cold\n\n"
        f"Example 2:\n"
        f"Symptoms: high fever, joint pain, rash, headache\n"
        f"Disease: Dengue\n\n"
        f"Example 3:\n"
        f"Symptoms: dry cough, body pain, loss of smell\n"
        f"Disease:"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-process result
        cleaned = response.strip()
        if ":" in cleaned:
            cleaned = cleaned.split(":")[-1].strip()
        if not cleaned.endswith("."):
            cleaned += "."
        return cleaned

    except Exception:
        return "Sorry, could not identify the disease."



# Improved natural remedy generator
def suggest_remedy(disease):
    prompt = (
        f"The user has a health issue: {disease}. "
        f"Suggest only natural, home-based remedies. "
        f"Do not recommend any medications or drugs. "
        f"Write 1-2 helpful sentences in simple language. "
        f"Don't repeat the disease name in the answer."
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150)  # Increased tokens
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean the output
        cleaned = response.replace(prompt, "").strip()
        if not cleaned.endswith((".", "!", "?")):
            cleaned += "."

        # Optional: check for blocked words
        blocked_words = ["ibuprofen", "tablet", "pill", "paracetamol", "drug", "medicine"]
        if any(bad_word in cleaned.lower() for bad_word in blocked_words):
            cleaned = "Please avoid medication. Try drinking warm fluids like honey-lemon tea and rest well."

        if cleaned == "":
            cleaned = "Try drinking warm water, turmeric milk, or honey tea. Use steam to relieve coughing."

        return cleaned

    except Exception as e:
        return "Sorry, I couldn’t generate a remedy at the moment."




# Streamlit app layout
st.markdown("<div class='title'>🩺 Citizen AI – Intelligent Health Assistant</div>", unsafe_allow_html=True)


tab1, tab2 = st.tabs(["🧪 Symptoms Identifier", "🌿 Home Remedies"])

with tab1:
    with st.container():
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        symptoms = st.text_input("Enter your symptoms (comma-separated):")
        if st.button("Predict Disease"):
            if symptoms:
                result = identify_disease(symptoms)
                st.success(f"Predicted Disease: {result}")
            else:
                st.warning("Please enter symptoms.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    with st.container():
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        disease = st.text_input("Enter the disease name:")
        if st.button("Suggest Remedy"):
            if disease:
                result = suggest_remedy(disease)
                st.success(f"Suggested Remedy: {result}")
            else:
                st.warning("Please enter a disease.")
        st.markdown("</div>", unsafe_allow_html=True)

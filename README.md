# Citizen AI – Intelligent Health Assistant 🤖🌿

This project is a Generative AI-powered health assistant built using **Google Colab**, **Gradio**, and **Hugging Face Transformers**.

## 🔍 Features
- **Symptoms Identifier**: Predict diseases from user-reported symptoms.
- **Home Remedies Generator**: Suggest natural remedies based on diseases.

## 🚀 Built With
- Python
- Gradio
- Hugging Face Transformers (`flan-t5-base`)
- Google Colab (no local setup needed)

## 📁 Run the App
1. Clone the repo or open the `.ipynb` in Google Colab
2. Install required libraries:
```python
!pip install gradio transformers

Run the last cell to launch the Gradio UI

💬 Sample Inputs
Symptoms: fever, cold, body pain

Output: Flu or Covid

Disease: Acidity

Remedy: Drink cold milk in the morning

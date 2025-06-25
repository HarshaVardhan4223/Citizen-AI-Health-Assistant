# 🩺 Citizen AI – Intelligent Health Assistant

A Generative AI-powered health assistant that:
- ✅ Predicts diseases based on user symptoms
- 🌿 Suggests home remedies for common illnesses
- Built with 🤗 Hugging Face models + Streamlit UI
- Deployed and tested on Google Colab & locally via VS Code

---

## 🚀 Features

### 🧪 1. Symptoms Identifier
- Enter symptoms like: `fever, cough, body pain`
- Get AI-predicted disease like: `Flu` or `COVID-19`

### 🌱 2. Home Remedies Generator
- Enter disease: `Acidity` or `Cough`
- Get natural remedies like:  
  _"Drink cold milk, avoid spicy food"_ or _"Honey-lemon tea with turmeric"_

---

## 🛠 Tech Stack

| Tool       | Purpose                            |
|------------|------------------------------------|
| Python     | Core logic                         |
| Streamlit  | UI framework                       |
| Transformers (FLAN-T5) | Hugging Face model for prompt response |
| Google Colab | Model testing + experimentation  |
| VS Code    | Local development & deployment     |

---

## 📷 Screenshots

![image](https://github.com/user-attachments/assets/3898da57-dd4b-4da0-94c9-ee63ff911e4f)


---

## 💡 Prompt Engineering

We used few-shot examples and medical context to guide the model:
```text
Symptoms: dry cough, body pain, loss of smell  
Expected Output: COVID-19

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the model and tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Function 1: Predict disease from symptoms
def identify_disease(symptoms: str) -> str:
    prompt = f"User has the following symptoms: {symptoms}. What could be the possible disease?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# Function 2: Suggest remedies
def suggest_home_remedy(disease: str) -> str:
    prompt = f"What are some natural home remedies for {disease}?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
async def process(request: Request, mode: str = Form(...), user_input: str = Form(...)):
    if mode == "symptoms":
        result = identify_disease(user_input)
    else:
        result = suggest_home_remedy(user_input)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "user_input": user_input,
        "mode": mode
    })

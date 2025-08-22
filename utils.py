import os, json, joblib, pathlib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def model_path(name: str) -> str:
    return os.path.join(MODEL_DIR, name)

def load_json(name: str):
    with open(model_path(name), "r", encoding="utf-8") as f:
        return json.load(f)

def load_model(name: str):
    return joblib.load(model_path(name))
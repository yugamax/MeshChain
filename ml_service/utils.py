import os, json, joblib, pathlib

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))

def model_path(name: str) -> str:
    return os.path.join(MODEL_DIR, name)

def save_json(obj, name: str):
    p = model_path(name)
    pathlib.Path(os.path.dirname(p)).mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(name: str):
    with open(model_path(name), "r", encoding="utf-8") as f:
        return json.load(f)

def save_model(model, name: str):
    p = model_path(name)
    pathlib.Path(os.path.dirname(p)).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)

def load_model(name: str):
    return joblib.load(model_path(name))
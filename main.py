from fastapi import FastAPI
import joblib

app = FastAPI(title="ThreatSenseAI")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.get("/")
def home():
    return {"message": "ThreatSenseAI is running"}

@app.post("/scan")
def scan_text(text: str):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    if pred == 1:
        return {"result": "Malicious / Phishing", "safe": False}
    else:
        return {"result": "Safe message", "safe": True}

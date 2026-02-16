import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Small starter dataset (we will replace with real dataset later)
data = {
    "text": [
        "Your bank account is locked click here",
        "Win free iphone now",
        "Verify your paypal account immediately",
        "Claim lottery prize urgent",
        "Meeting at 5pm tomorrow",
        "Let's have lunch",
        "Project submission deadline extended",
        "Your OTP is 458921 do not share"
    ],
    "label": [1,1,1,1,0,0,0,0]  # 1 = malicious, 0 = safe
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# Train model
model = LogisticRegression()
model.fit(X, df["label"])

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("ThreatSenseAI model trained successfully!")

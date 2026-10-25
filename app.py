
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {
    "text": [
        "Share this secret cure immediately",
        "Guaranteed money if you forward this message",
        "This shocking claim has no evidence",
        "The official website reports a road closure",
        "The article links to an original research report",
        "The local library closes at 6 PM"
    ],
    "label": [
        "Potentially Misleading",
        "Potentially Misleading",
        "Potentially Misleading",
        "Needs Verification",
        "Needs Verification",
        "Needs Verification"
    ]
}

df = pd.DataFrame(data)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

model.fit(df["text"], df["label"])

print("FAKE NEWS CLAIM CLASSIFICATION")
print("--------------------------------")
print("Educational prototype only")
print("Type 'exit' to stop")

while True:
    claim = input("\nEnter a news claim: ")

    if claim.lower() == "exit":
        print("Program ended.")
        break

    if not claim.strip():
        print("Please enter a claim.")
        continue

    result = model.predict([claim])[0]

    print("Predicted category:", result)
    print("Warning: This is not a fact-check.")
    print("Verify the claim using reliable sources.")
  

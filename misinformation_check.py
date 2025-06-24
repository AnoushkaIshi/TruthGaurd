import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import pipeline

def classify_claims(claims):
    classifier = pipeline("text-classification", model="bert-base-uncased")

    results = []
    for claim in claims:
        classification = classifier(claim)[0]
        label = classification['label']
        confidence = f"{classification['score'] * 100:.2f}%"

        # Convert HuggingFace model's label to human-readable "Yes/No"
        if "POSITIVE" in label:
            verdict = "Yes"
        else:
            verdict = "No"

        result = {
            "claim": claim,
            "classification": {
                "label": verdict,
                "confidence": confidence
            }
        }
        results.append(result)

    return results

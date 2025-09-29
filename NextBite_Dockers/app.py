from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from transformers import pipeline
import json 

# Create new FASTAPI api
app = FastAPI()

pipe = pipeline("ner", model="review-model-updated", tokenizer="review-model-updated")


@app.get('/')
def home():
    return {
    "message":"Hello world"
    }

@app.get("/generate")
def generate(text: str):
    if pipe is None:
        return {"error": "Model failed to load."}
    try:
        output = pipe(text)

        res = reconstruct_entities(output, text)
            
        return json.dumps(res)
    except Exception as e:
        return {"error": str(e)}

def reconstruct_entities(tokens, original_text):
    entities = []
    if not tokens:
        return entities

    current = {
        "entity": tokens[0]["entity"],
        "start": tokens[0]["start"],
        "end": tokens[0]["end"]
    }

    for token in tokens[1:]:
        if token["entity"].startswith("I") and token["entity"][2:] == current["entity"][2:]:
            # Continuation of current entity
            current["end"] = token["end"]
        else:
            # New entity starts
            entity_text = original_text[current["start"]:current["end"]]
            entities.append(entity_text)
            current = {
                "entity": token["entity"],
                "start": token["start"],
                "end": token["end"]
            }

    # Add final entity
    entity_text = original_text[current["start"]:current["end"]]
    entities.append(entity_text)

    return entities
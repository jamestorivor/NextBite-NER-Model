from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset
import json


dataset = load_dataset("csv", data_files={"unlabeled": "output_data.csv"})["unlabeled"]
dataset = dataset.select(range(1000))

tokenizer = AutoTokenizer.from_pretrained("../review-model-v1")
model = AutoModelForTokenClassification.from_pretrained("../review-model-v1")

ner_pipe = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
)


results = []

# 4. Collect all texts at once
texts = list(dataset["review_text"])

# 5. Run batched inference
all_preds = ner_pipe(texts, batch_size=32)

# 6. Build Label Studio pre-annotation JSON
results = []

for text, preds in zip(texts, all_preds):
    formatted_preds = []
    prev_label = None
    prev_end = -1

    for i, p in enumerate(preds):
        current_label = p["entity_group"]
        current_start = p["start"]

        # If this is a new entity span (non-contiguous or different entity)
        if current_label != prev_label or current_start > prev_end + 1:
            tag = f"B-{current_label}"
        else:
            tag = f"I-{current_label}"

        formatted_preds.append({
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": p["start"],
                "end": p["end"],
                "text": p["word"],
                "labels": [tag]
            }
        })

        prev_label = current_label
        prev_end = p["end"]

    results.append({
        "data": {"text": text},
        "predictions": [{"result": formatted_preds}]
    })

# 7. Save to JSON
with open("preannotations.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

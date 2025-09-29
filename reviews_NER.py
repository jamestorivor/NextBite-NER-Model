import json
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, create_optimizer, DefaultDataCollator, Trainer, TrainingArguments
from datasets import *

label2id = {
    "O" : 0,
    "I-Food" : 1,
    "B-Food" : 2,
    "B-Drink" : 3,
    "I-Drink" : 4,
    "B-Cuisine" : 5,
    "I-Cuisine" : 6
}

id2label = {
    0 : "O",
    1 : "I-Food",
    2 : "B-Food",
    3 : "B-Drink",
    4 : "I-Drink",
    5: "B-Cuisine",
    6: "I-Cuisine"
}

with open("./Annotated_Data/cuisine_generated_data1.json", "r") as file:
    data = json.load(file)

processed_data = []
for item in data:
    # Extract relevant fields and format them for your Hugging Face task
    processed_data.append({"text": item["data"]["text"], "labels": item["annotations"][0]["result"]})

ds = Dataset.from_list(processed_data)

train_testvalid = ds.train_test_split(test_size=0.15)
# test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
dataset = DatasetDict({
'train': train_testvalid['train'],
'validation': train_testvalid['test']}
)

tokenizer = AutoTokenizer.from_pretrained("./checkpoints/review-model-v2")
model = AutoModelForTokenClassification.from_pretrained("./checkpoints/review-model-v2", num_labels=7, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)


# converts the "ner_tags" / output/ labels from the train dataset into tokens, then matches the sequence of the output to the input
def tokenize_and_align_labels(example):
    tokenized_input = tokenizer(
        example["text"],
        truncation=True,
        padding=False,  # Don't pad here; pad later in data collator
        return_offsets_mapping=True
    )


    labels = []

    for i, offsets in enumerate(tokenized_input["offset_mapping"]):
        entities = example["labels"][i]
        ner_tags = []

        for (start_offset, end_offset) in offsets:
            if start_offset == end_offset == 0:
                ner_tags.append(-100)
                continue


            # Default tag
            tag = "O"

            # Check if this token falls inside any entity span
            for entity in entities:
                ent_start, ent_end = entity["value"]["start"], entity["value"]["end"]
                label = entity["value"]["labels"][0]

                if start_offset >= ent_start and end_offset <= ent_end:
                    if label == "Cuisine":
                        # Assign B-/I- for Cuisine
                        tag = "B-Cuisine" if start_offset == ent_start else "I-Cuisine"
                    else:
                        tag = label if start_offset == ent_start else "I" + label[1:]
                    break

            ner_tags.append(tag)

        label_ids = []
        for tag in ner_tags:
            if tag == -100:
                label_ids.append(tag)
            else:
                label_ids.append(label2id[tag])
        labels.append(label_ids)
    
    tokenized_input["labels"] = labels
    tokenized_input.pop("offset_mapping")
    return tokenized_input

tokenized_ds = dataset.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

args = TrainingArguments(
    "bert-finetuned-ner",
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)
trainer.train()

model.save_pretrained("checkpoints/review-model-v1")
tokenizer.save_pretrained("checkpoints/review-model-v1")

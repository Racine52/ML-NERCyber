from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import classification_report
import numpy as np
import json
import os
import torch

DATA_DIR = "./data" 

torch.cuda.empty_cache()
device = torch.device("cpu")


train_file = os.path.join(DATA_DIR, "NER-TRAINING.jsonlines")
val_file = os.path.join(DATA_DIR, "NER-VALIDATION.jsonlines")

def load_data(file_path):
    """
    Load data from a JSON lines file and return it as a list of dictionaries.

    Parameters:
        file_path (str): The path to the JSON lines file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a JSON object from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

train_data = load_data(train_file)
val_data = load_data(val_file)

# Create Hugging Face dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print(train_dataset)

label_list = list(set(tag for sample in train_data for tag in sample["ner_tags"]))
label_list.sort()
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

def add_label_ids(example):
    """
    Add label IDs to the input example dictionary based on the label-to-ID mapping.

    Parameters:
        example (dict): A dictionary representing a single sample from the dataset.
        It should contain the following keys:
        - "ner_tags" (list of str): A list of named entity recognition (NER) tags.

    Returns:
        dict: The input example dictionary with an additional key "ner_ids".
            The "ner_ids" key contains a list of corresponding label IDs for each NER tag.
    """
    example["ner_ids"] = [label_to_id[tag] for tag in example["ner_tags"]]
    return example

train_dataset = train_dataset.map(add_label_ids)
val_dataset = val_dataset.map(add_label_ids)

print("Exemple après transformation :", train_dataset[0])  

model_name = "jackaduma/SecBERT" # Use SECBERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
model.to(device)

def tokenize_and_align_labels(examples):
    """
    Tokenize input examples and align their labels with the corresponding tokens.

    Parameters:
        examples (dict): A dictionary containing a list of input examples.
            The dictionary should have the following keys:
            - "tokens" (list of str): A list of input tokens.
            - "ner_ids" (list of list of int): A list of corresponding named entity recognition (NER) label IDs.

    Returns:
        dict: A dictionary containing tokenized inputs and aligned labels.
            The dictionary will have the following keys:
            - "input_ids" (list of int): A list of token IDs to be fed to the model.
            - "attention_mask" (list of int): A list of attention masks to ignore padding tokens.
            - "labels" (list of list of int): A list of aligned NER label IDs for each token.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        max_length=512,  # Maximum length for BERT
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:  # No corresponding word
                aligned_labels.append(-100)  # Ignore this sub-token
            else:
                aligned_labels.append(label[word_id])  # Align with ner_ids
        labels.append(aligned_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(pred):
    """
    Compute metrics for Named Entity Recognition (NER) model evaluation.

    Parameters:
        pred (tuple): A tuple containing two numpy arrays:
            - predictions (numpy.ndarray): A 3D array of shape (batch_size, sequence_length, num_labels)
                containing model predictions for each token.
            - labels (numpy.ndarray): A 2D array of shape (batch_size, sequence_length) containing
                true labels for each token.

    Returns:
        dict: A dictionary containing the following metrics:
            - "precision": Weighted average precision score.
            - "recall": Weighted average recall score.
            - "f1": Weighted average F1 score.
    """
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = classification_report(
        [label for sublist in true_labels for label in sublist],
        [pred for sublist in true_predictions for pred in sublist],
        output_dict=True
    )
    return {
        "precision": results["weighted avg"]["precision"],
        "recall": results["weighted avg"]["recall"],
        "f1": results["weighted avg"]["f1-score"],
    }

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print("Résultats d'évaluation :", results)

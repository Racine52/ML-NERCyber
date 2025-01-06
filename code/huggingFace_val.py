from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer
from datasets import Dataset
from sklearn.metrics import classification_report
import numpy as np
import json
import os

DATA_DIR = "./data" 

def load_data(file_path):
    """
    Load data from a JSON lines file.

    This function reads a JSON lines file and returns a list of dictionaries,
    where each dictionary represents a JSON object.

    Parameters:
        file_path (str): The path to the JSON lines file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a JSON object.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


val_file = os.path.join(DATA_DIR, "NER-VALIDATION.jsonlines")
val_data = load_data(val_file)
val_dataset = Dataset.from_list(val_data)

checkpoint_dir = './results/checkpoint-1830'  
model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)


label_list = list(set(tag for sample in val_data for tag in sample["ner_tags"]))
label_list.sort()
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

def add_label_ids(example):
    """
    Add numerical label IDs to the input example based on the provided label-to-ID mapping.

    This function takes an input example, which is a dictionary containing a list of NER tags,
    and adds a new key-value pair to the dictionary. The new key is 'ner_ids', and the value is a list
    of numerical label IDs corresponding to the NER tags. The label-to-ID mapping is provided as a global
    variable 'label_to_id'.

    Parameters:
        example (dict): A dictionary representing an input example. It should contain a key 'ner_tags'
                        with a list of NER tags as its value.

    Returns:
        dict: The input example with an additional key-value pair 'ner_ids' containing the numerical label IDs.
    """
    example["ner_ids"] = [label_to_id[tag] for tag in example["ner_tags"]]
    return example

val_dataset = val_dataset.map(add_label_ids)

def tokenize_and_align_labels(examples):
    """
    Tokenize input examples and align their labels with padding and truncation.

    This function takes a dictionary of input examples, tokenizes the 'tokens' field using the provided tokenizer,
    and aligns the corresponding 'ner_ids' labels with the tokenized sequences. The tokenization process includes
    padding and truncation to ensure all sequences have the same length.

    Parameters:
    examples (dict): A dictionary containing input examples. It should have the following keys:
                    - 'tokens': A list of lists, where each inner list represents a sequence of tokens.
                    - 'ner_ids': A list of lists, where each inner list represents the corresponding NER labels.

    Returns:
        dict: A dictionary containing the tokenized inputs and aligned labels. The dictionary will have the following keys:
          - 'input_ids': A list of tokenized input IDs.
          - 'attention_mask': A list of attention masks.
          - 'labels': A list of aligned NER labels.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",  
        max_length=512,        
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  
            else:
                aligned_labels.append(label[word_id])
        labels.append(aligned_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

def compute_metrics(pred):
    """
    Compute precision, recall, and F1 score for a sequence labeling task.

    This function takes the model predictions and true labels as input, processes them, and computes the
    weighted average precision, recall, and F1 score using the sklearn.metrics.classification_report function.

    Parameters:
        pred (tuple): A tuple containing two numpy arrays. The first array represents the model predictions,
                  and the second array represents the true labels. The shape of both arrays is (batch_size, seq_len, num_labels).

    Returns:
        dict: A dictionary containing the weighted average precision, recall, and F1 score. The keys are 'precision', 'recall', and 'f1'.
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

trainer = Trainer(
    model=model,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

results = trainer.evaluate()
print("Résultats d'évaluation :", results)

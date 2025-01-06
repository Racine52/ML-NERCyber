import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import tqdm

# Définir les chemins
DATA_DIR = "./data" 
INPUT_FILE = os.path.join(DATA_DIR, "NER-TESTING.jsonlines")
OUTPUT_FILE = os.path.join(DATA_DIR, "NER-PREDICTIONS-RESULT.jsonlines")
CHECKPOINT_DIR = "./results/checkpoint-1830"  

# Mapping from label index to label name
label_mapping = {
    'LABEL_0': 'B-Entity',       
    'LABEL_1': 'B-Action', 
    'LABEL_2': 'B-Modifier',
    'LABEL_3': 'I-Entity', 
    'LABEL_4': 'I-Action', 
    'LABEL_5': 'I-Modifier', 
    'LABEL_6': 'O'  
}

# Load the last model from the checkpoint and the tokenizer
checkpoint_dir = './results/checkpoint-1830' 
model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)


def load_prediction_data(file_path):
    """Load data from a JSONLines file.

    Args:
        file_path (str): path to the file

    Returns:
        json: content of the file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def predict_labels(test_data):
    """Predict NER tags for a list of examples

    Args:
        test_data (json): list of examples

    Returns:
        dict: list of predictions
    """
    results = []
    for example in tqdm.tqdm(test_data):  
        tokens = example["tokens"]

        # Tokeniser l'exemple avec alignement des mots
        inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        word_ids = inputs.word_ids()  # Récupérer les indices des mots pour chaque sous-token

        # Effectuer les prédictions
        with torch.no_grad():
            outputs = model(**{key: val.to(model.device) for key, val in inputs.items()})
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

        # Aligner les prédictions sur les mots d'origine
        aligned_labels = []
        previous_word_id = None
        for word_id, label_id in zip(word_ids, predictions):
            if word_id is None:
                continue  # Ignorer les sous-tokens spéciaux ([CLS], [SEP], etc.)
            if word_id != previous_word_id: 
                aligned_labels.append(label_mapping[f'LABEL_{label_id}'])
                previous_word_id = word_id

        # Ajouter les résultats au dictionnaire de sortie
        results.append({
            "unique_id": example["unique_id"],
            "tokens": tokens,
            "ner_tags": aligned_labels  
        })

    return results


def save_predictions(predictions, output_file):
    """Save predictions to a JSONLines file

    Args:
        predictions (dict): list of predictions
        output_file (str): path to the output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + '\n')

# load test data
test_file = os.path.join(DATA_DIR, "NER-TESTING.jsonlines")
test_data = load_prediction_data(test_file)

# predict labels
predictions = predict_labels(test_data)

output_file = 'predictions.jsonlines'  
save_predictions(predictions, output_file)

print(f"Prédictions sauvegardées dans {output_file}")
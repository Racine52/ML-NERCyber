# ML-NERCyber: Named Entity Recognition for Cybersecurity

## Overview

**ML-NERCyber** is a machine learning project designed to perform Named Entity Recognition (NER) in the cybersecurity domain. The project leverages state-of-the-art models like **SecBERT** and **Hugging Face Transformers** to identify and classify named entities such as malware, vulnerabilities, indicators, and more from unstructured textual data. This repository includes tools for training, evaluating, and predicting NER models, as well as utilities for analyzing datasets and assessing model predictions.

---

## Features

- **Training**: Fine-tune a pre-trained transformer model (e.g., SecBERT) on cybersecurity-specific data.
- **Prediction**: Use the trained model to predict entities in unseen data.
- **Validation**: Evaluate the model's performance on labeled validation datasets.
- **Data Analysis**: Tools for dataset exploration and preprocessing to ensure quality and balance in the labeled data.
- **Evaluation**: Functions to calculate metrics such as precision, recall, and F1-score, both globally and per entity class.

---

## Requirements

Install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Ensure that Python 3.8+ is installed on your system.

---

## File Structure

```
ML-NERCyber/
├── code/
│   ├── huggingFace_train.py      # Training script
│   ├── huggingFace_test.py       # Prediction script
│   ├── huggingFace_val.py        # Validation script
├── data/
│   ├── NER-TESTING.jsonlines     # Testing dataset   
│   ├── NER-TRAINNING.jsonlines   # Training dataset  
│   ├── NER-VALIDATION.jsonlines  # Validation dataset
├── outils/
│   ├── compute_seqeval.py     	  # Analyse prediction  
│   ├── data_processing.py     	  # Process data  
│   ├── explore_data.py     	  # Analyse dataset
├── config.py                     # Configuration file
├── requirements.txt              # Required Python libraries
├── main.py                       # Entry point script
├── predictions.jsonlines         # Output prediction

├── main.py                       # Entry point script
└── README.md                     # Project documentation
```

---

## Usage

The project is controlled via the `main.py` script, which allows you to perform training, prediction, and validation. Use the following commands to execute different actions:

### Train the Model

To train the NER model on the dataset:

```bash
python main.py train
```

### Predict Entities

To predict entities in unseen data:

```bash
python main.py predict
```

### Validate the Model

To evaluate the model's performance on the validation dataset:

```bash
python main.py validate
```

---

## Dataset

The project includes tools for dataset analysis to ensure quality and balance across different entity labels. Before training, make sure the datasets are correctly structured and follow the **BIO schema** for entity tagging.

---

## Evaluation

Evaluation metrics include:

- **Precision**: Percentage of correctly identified entities among all predicted entities.
- **Recall**: Percentage of correctly identified entities among all actual entities.
- **F1-Score**: Harmonic mean of precision and recall, used as the primary evaluation metric.

Additionally, per-class metrics (e.g., `B-Entity`, `I-Action`, etc.) are calculated to assess performance for each entity type.

---

## Contributions

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

---

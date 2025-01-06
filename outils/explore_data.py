import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

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
    
def remove_empty_documents(dataset):
    """
    Remove empty documents from the dataset.

    This function filters out documents from the given dataset where all the NER tags are 'O'.
    An empty document is considered one where none of the NER tags are different from 'O'.

    Parameters:
        dataset (list): A list of dictionaries, where each dictionary represents a document.
                    Each document should have a 'ner_tags' key, which is a list of NER tags.

    Returns:
        list: A filtered list of dictionaries, where each dictionary represents a non-empty document.
    """
    filtered_dataset = [
        document for document in dataset
        if any(tag != "O" for tag in document["ner_tags"])
    ]
    return filtered_dataset

def explore_dataset(dataset):
    """
    Explore a dataset by calculating various statistics and creating a DataFrame for token-tag counts.

    Parameters:
        dataset (list): A list of dictionaries, where each dictionary represents a document.
                        Each document should have 'tokens' and 'ner_tags' keys, which are lists of tokens and NER tags respectively.

    Returns:
        dict: A dictionary containing the following statistics and the token-tag DataFrame:
            - num_entries: The number of entries in the dataset.
            - total_tokens: The total number of tokens in the dataset.
            - unique_tokens: The number of unique tokens in the dataset.
            - tag_distribution: A Counter object representing the distribution of NER tags in the dataset.
            - avg_tokens_per_entry: The average number of tokens per entry in the dataset.
            - token_tag_df: A pandas DataFrame containing token-tag pairs.
    """
    num_entries = len(dataset)

    all_tokens = [token for entry in dataset for token in entry["tokens"]]
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))

    all_tags = [tag for entry in dataset for tag in entry["ner_tags"]]
    tag_distribution = Counter(all_tags)

    avg_tokens = total_tokens / num_entries

    token_tag_pairs = [(token, tag) for entry in dataset for token, tag in zip(entry["tokens"], entry["ner_tags"])]
    token_tag_df = pd.DataFrame(token_tag_pairs, columns=["Token", "Tag"])

    return {
        "num_entries": num_entries,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "tag_distribution": tag_distribution,
        "avg_tokens_per_entry": avg_tokens,
        "token_tag_df": token_tag_df
    }

if __name__ == '__main__':

    train_data = load_data("./data/NER-TRAINING.jsonlines")
    filtered_train_data = remove_empty_documents(train_data)
    stats = explore_dataset(filtered_train_data)

    print(f"Number of entries: {stats['num_entries']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Unique tokens: {stats['unique_tokens']}")
    print(f"Tag distribution: {stats['tag_distribution']}")
    print(f"Average tokens per entry: {stats['avg_tokens_per_entry']:.2f}")


    stats['token_tag_df'].head()


    tags = list(stats['tag_distribution'].keys())
    counts = list(stats['tag_distribution'].values())

    plt.figure(figsize=(8, 6))
    plt.bar(tags, counts, color='skyblue')
    plt.xlabel("NER Tags", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Distribution of NER Tags in the filtered Dataset", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()

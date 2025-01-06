import json 


def load_sample_data(file_path):
    """
    Load a sample of data from a JSON file.

    This function reads a JSON file and extracts a specified number of data samples.
    The number of samples is limited to 5 by default.

    Parameters:
    f   ile_path (str): The path to the JSON file containing the data.

    Returns:
        list: A list of dictionaries, where each dictionary represents a data sample.
    """
    data_samples = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if len(data_samples) < 5:
                data_samples.append(json.loads(line))
            else:
                break
    return data_samples

def load_data(file_path):
    """
    Load data from a JSON file.

    This function reads a JSON file and extracts all data samples present in the file.
    Each data sample is represented as a dictionary.

    Parameters:
        file_path (str): The path to the JSON file containing the data.

    Returns:
        list: A list of dictionaries, where each dictionary represents a data sample.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def analyze_sample_data(data_samples, index=0):
    """
    Analyze and print a specific data sample from a list of data samples.

    This function takes a list of dictionaries representing data samples and an optional index.
    It prints the unique ID of the selected data sample along with the tokens and their corresponding NER tags.

    Parameters:
        data_samples (list): A list of dictionaries, where each dictionary represents a data sample.
                         Each data sample should have the following keys: 'unique_id', 'tokens', 'ner_tags'.
        index (int, optional): The index of the data sample to analyze. Default is 0.

    Returns:
        None
    """
    col_width = 20
    print("id : " + str(data_samples[index]['unique_id']))
    for i in range(len(data_samples[index]['tokens'])):
        token = data_samples[index]['tokens'][i]
        tag = data_samples[index]['ner_tags'][i]
        print(token.ljust(col_width) + "|" + tag.ljust(col_width))
        
def count_token_labels(data_samples, id=None):
    token = []
    label = []
    if id is not None:
        for data in data_samples:
            if data['unique_id'] == id:
                token.append(len(data['tokens']))
def count_token_labels(data_samples, id=None):
    """
    Count the number of tokens and labels in a list of data samples.

    This function takes a list of dictionaries representing data samples and an optional unique ID.
    It counts the number of tokens and labels for each data sample, and returns the counts as two separate lists.
    If a unique ID is provided, the function will only count the tokens and labels for the data sample with that ID.

    Parameters:
        data_samples (list): A list of dictionaries, where each dictionary represents a data sample.
                             Each data sample should have the following keys: 'unique_id', 'tokens', 'ner_tags'.
        id (str, optional): The unique ID of the data sample to count. If not provided, the function will count all data samples.

    Returns:
        tuple: A tuple containing two lists:
               - A list of integers representing the number of tokens for each data sample.
               - A list of integers representing the number of labels for each data sample.
    """
    token = []
    label = []
    if id is not None:
        for data in data_samples:
            if data['unique_id'] == id:
                token.append(len(data['tokens']))
                label.append(len(data['ner_tags']))
                break
    else:
        for data in data_samples:
            token.append(len(data['tokens']))
            label.append(len(data['ner_tags']))
            
    return token, label
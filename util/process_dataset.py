import pandas as pd

def load_dataset(file_path: str):
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pandas.DataFrame: The loaded dataset.
    """
    
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
def process_benign_dataset(benign_data):
    X = benign_data.drop(columns=['label'])
    y = benign_data['label']
    
    # dropping unnecessary columns
    X = X.drop(columns=['category'])
    X = X.drop(columns=['specific_class'])
    
    # convert benign labels to 0 and DoS labels to 1
    y = y.apply(lambda x: 0 if x == 'BENIGN' else 1)
        
    return X, y

def process_dos_dataset(dos_data):
    X = dos_data.drop(columns=['label'])
    y = dos_data['label']
    
    # dropping unnecessary columns
    X = X.drop(columns=['category'])
    X = X.drop(columns=['specific_class'])
    
    # convert DoS labels to 1 and benign labels to 0
    y = y.apply(lambda x: 1 if x == 'ATTACK' else 0)
    
    return X, y

def combine_datasets(benign_X, benign_y, dos_X, dos_y):
    """
    Combine benign and DoS datasets into a single dataset.
    
    Args:
        benign_X: Features from benign dataset
        benign_y: Labels from benign dataset
        dos_X: Features from DoS dataset
        dos_y: Labels from DoS dataset
    
    Returns:
        Tuple of combined features (X) and labels (y)
    """
    X = pd.concat([benign_X, dos_X], ignore_index=True)
    y = pd.concat([benign_y, dos_y], ignore_index=True)
    return X, y

def random_combine_datasets(benign_X, benign_y, dos_X, dos_y):
    """
    Combine benign and DoS datasets into a single dataset with random shuffling.
    
    Args:
        benign_X: Features from benign dataset
        benign_y: Labels from benign dataset
        dos_X: Features from DoS dataset
        dos_y: Labels from DoS dataset
    
    Returns:
        Tuple of combined features (X) and labels (y)
    """
    combined_X = pd.concat([benign_X, dos_X], ignore_index=True)
    combined_y = pd.concat([benign_y, dos_y], ignore_index=True)
    
    # Shuffle the combined dataset
    shuffled_indices = combined_X.sample(frac=1, random_state=42).index
    X = combined_X.loc[shuffled_indices].reset_index(drop=True)
    y = combined_y.loc[shuffled_indices].reset_index(drop=True)
    
    return X, y

def get_dataset() -> None | tuple[pd.DataFrame, pd.Series]:
    """
    Load and process the benign and DoS datasets, 
    then combine them into a single dataset.
    
    Return:
        the combined features and labels as a tuple (X, y) if successful, or None if there was an error.
    """
    benign_data = load_dataset('dataset/decimal_benign.csv')
    dos_data = load_dataset('dataset/decimal_DoS.csv')
    
    if benign_data is None:
        print("Failed to load benign dataset.")
        return None
    
    if dos_data is None:
        print("Failed to load DoS dataset.")
        return None
    
    benign_X, benign_y = process_benign_dataset(benign_data)
    dos_X, dos_y = process_dos_dataset(dos_data)
    
    X, y = random_combine_datasets(benign_X, benign_y, dos_X, dos_y)
    return X, y

# For testing the dataset processing
if __name__ == "__main__":
    dataset = get_dataset()
    if dataset is not None:
        X, y = dataset
        print("Dataset processed successfully.")
        print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        
        print("Sample features:")
        print(X.head())
        print("Sample labels:")
        print(y.head())
    else:
        print("Failed to process dataset.")
# project_root/scripts/risk_ngrams.py
# run 'python -m scripts.risk_ngrams' on terminal

# gets raw ngrams (n=1..6) from datasets for each risk label and outputs to ../data/risk_labels/ngrams_*.json
# also outputs risk-only ngrams to ../data/risk_labels/ngrams_risk_only.json
# the file ../data/risk_labels/risk_ngrams_reviewed.json contains the final manually reviewed ngrams used in inference
import pandas as pd
import re
from collections import Counter
import json
from typing import Dict, List, Tuple
from nltk.corpus import stopwords
import nltk
import os

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

ENGLISH_STOPWORDS = set(stopwords.words('english'))

DATASET_PATHS = {
    "depression": "../data/processed/dataset_depression_clean.csv",
    "selfharm": "../data/processed/dataset_selfharm_clean.csv",
    "suicidal": "../data/processed/dataset_suicidal_clean.csv",
    "grief": "../data/processed/dataset_grief_clean.csv"
}

def extract_top_ngrams_by_label(
    csv_path: str,
    label: str,
    text_column: str = 'text',
    label_column: str = 'label',
    n_range: tuple = (1, 2, 3, 4, 5, 6),
    top_k: int = 50,
    preprocess: bool = True,
    remove_stopwords: bool = True
) -> Dict[str, List[str]]:
    """
    Extract top n-grams from a CSV dataset for a specific label.
    
    Args:
        csv_path: Path to the CSV file
        label: The label to filter by (value in label_column)
        text_column: Name of the text column
        label_column: Name of the label column
        n_range: Tuple of n values for n-grams
        top_k: Number of top n-grams to extract for each n
        preprocess: Whether to preprocess text (lowercase, remove punctuation)
        remove_stopwords: Whether to filter out n-grams containing only stopwords
    
    Returns:
        Dictionary with keys as n-gram types and values as lists of top n-grams
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")
    
    # Check if required columns exist
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in CSV")
    
    # Filter data for the specified label
    label_data = df[df[label_column] == label]
    
    if label_data.empty:
        available_labels = df[label_column].unique()
        raise ValueError(
            f"Label '{label}' not found. Available labels: {list(available_labels)}"
        )
    
    # Extract all text for the label
    all_texts = label_data[text_column].astype(str).tolist()
    
    def preprocess_text(text: str) -> str:
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and special characters (keep alphanumeric, whitespace, and apostrophes)
        text = re.sub(r'[^\w\s\']', ' ', text)
        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_ngrams(text: str, n: int) -> List[str]:
        """Extract n-grams from a single text."""
        if preprocess:
            text = preprocess_text(text)
        
        words = text.split()
        ngrams = []
        
        # Extract n-grams
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    # Collect all n-grams
    ngram_counter = Counter()
    
    # Process each text and extract n-grams
    for text in all_texts:
        if pd.isna(text) or text.strip() == "":
            continue
        
        for n in n_range:
            ngrams = extract_ngrams(text, n)
            ngram_counter.update(ngrams)
    
    # Organize results by n-gram type
    results = {}
    for n in n_range:
        # Filter n-grams of length n
        ngram_type = f"{n}-grams"
        ngrams_of_n = {}
        
        for ngram, count in ngram_counter.items():
            if len(ngram.split()) == n:
                # Skip if all words are stopwords (optional)
                if remove_stopwords:
                    words_in_ngram = ngram.split()
                    if all(word in ENGLISH_STOPWORDS for word in words_in_ngram):
                        continue
                ngrams_of_n[ngram] = count
        
        # Get top k n-grams
        top_ngrams = sorted(
            ngrams_of_n.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Store just the n-gram strings (not counts)
        results[ngram_type] = [ngram for ngram, count in top_ngrams]
    
    return results

def get_dataset_labels(csv_path: str) -> List[str]:
    """Get unique labels from a dataset."""
    df = pd.read_csv(csv_path)
    return df['label'].unique().tolist()

def extract_ngrams_for_all_datasets():
    """Extract n-grams for all datasets and save them separately."""
    
    all_results = {}
    
    for risk_type, csv_path in DATASET_PATHS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {risk_type}")
        print(f"File: {os.path.basename(csv_path)}")
        
        # Get labels for this dataset
        labels = get_dataset_labels(csv_path)
        print(f"Available labels: {labels}")
        
        dataset_results = {}
        
        for label in labels:
            print(f"\n  Extracting n-grams for label: '{label}'")
            
            try:
                ngrams = extract_top_ngrams_by_label(
                    csv_path=csv_path,
                    label=label,
                    text_column='text',
                    label_column='label',
                    n_range=(1, 2, 3, 4, 5, 6),
                    top_k=50,
                    remove_stopwords=True
                )
                
                dataset_results[label] = ngrams
                print(f"    Successfully extracted n-grams")
                
            except Exception as e:
                print(f"    Error: {e}")
                dataset_results[label] = {}
        
        # Save this dataset's results to a separate JSON file
        output_file = f"../data/risk_labels/ngrams_{risk_type}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n  Saved to: {output_file}")
        
        # Add to all results
        all_results[risk_type] = dataset_results
    
    # Save combined results
    combined_file = "../data/risk_labels/ngrams_all_datasets_combined.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Combined results saved to: {combined_file}")
    
    return all_results

def extract_risk_only_ngrams():
    """Extract n-grams only for the risk labels (excluding 'safe')."""
    
    risk_ngrams = {}
    
    for risk_type, csv_path in DATASET_PATHS.items():
        print(f"\nProcessing: {risk_type}")
        
        # Get labels and find the risk label (not 'safe')
        labels = get_dataset_labels(csv_path)
        risk_label = [label for label in labels if label != 'safe'][0]
        
        print(f"  Risk label: '{risk_label}'")
        
        try:
            ngrams = extract_top_ngrams_by_label(
                csv_path=csv_path,
                label=risk_label,
                text_column='text',
                label_column='label',
                n_range=(1, 2, 3, 4, 5, 6),
                top_k=50,
                remove_stopwords=True
            )
            
            risk_ngrams[risk_label] = ngrams
            print(f"  Successfully extracted {sum(len(v) for v in ngrams.values())} n-grams")
            
        except Exception as e:
            print(f"  Error: {e}")
            risk_ngrams[risk_label] = {}
    
    # Save risk-only n-grams
    output_file = "../data/risk_labels/ngrams_risk_only.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(risk_ngrams, f, indent=2, ensure_ascii=False)
    
    print(f"\nRisk-only n-grams saved to: {output_file}")
    
    return risk_ngrams

if __name__ == "__main__":
    print("Starting n-gram extraction...")
    
    # Option 1: Extract n-grams for all datasets (all labels)
    print("\nOption 1: Extracting n-grams for all datasets (all labels)")
    all_results = extract_ngrams_for_all_datasets()
    
    # Option 2: Extract only risk labels (excluding 'safe')
    print("\n" + "="*60)
    print("Option 2: Extracting only risk labels (excluding 'safe')")
    risk_results = extract_risk_only_ngrams()
    
    # Print summary of risk n-grams
    print("\n" + "="*60)
    print("RISK N-GRAMS SUMMARY:")
    print("="*60)
    
    for risk_label, ngrams_dict in risk_results.items():
        print(f"\n{risk_label.upper()}:")
        print("-" * 30)
        for ngram_type, ngrams in ngrams_dict.items():
            print(f"  {ngram_type}:")
            for i, ngram in enumerate(ngrams[:5], 1):  # Show top 5
                print(f"    {i}. {ngram}")
            if len(ngrams) > 5:
                print(f"    ... and {len(ngrams) - 5} more")
        print()
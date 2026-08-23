"""
GPU-Optimized Paraphrasing Augmentation for Disgust Dataset
Using t5-small for better compatibility
"""
# dependencies:
# pip install pandas torch transformers tqdm sentencepiece numpy sys

# run this to create the paraphrased dataset:
# python disgust_paraphrase.py

import pandas as pd
import numpy as np
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
from pathlib import Path
sys.path.append('../')
from scripts.merge_to_6labels_7labels import create_ids

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def disgust_paraphrase():
    """
    Main function to load disgust data, paraphrase it using T5,
    and save the augmented dataset, tracking original sample IDs.
    """
    # ============================================
    # 1. LOAD DATA
    # ============================================
    print("\n📂 Loading disgust data...")
    df = create_ids(mode="disgust")  # Load disgust data using create_ids function
    
    # Ensure 'sample_id' is numeric and find the max value
    df['sample_id'] = pd.to_numeric(df['sample_id'], errors='coerce')
    df.dropna(subset=['sample_id'], inplace=True)
    df['sample_id'] = df['sample_id'].astype(int)
    
    original_texts = df['text'].tolist()
    original_ids = df['sample_id'].tolist() # Track original IDs
    
    print(f"Loaded {len(original_texts)} original disgust instances")

    print(f"\n🔍 Sample original texts:")
    print(df.head(5))
    
    # ============================================
    # 2. T5 PARAPHRASER (Using t5-small for compatibility)
    # ============================================
    print("\n📝 Loading T5 model...")
    
    # Use t5-small instead (more compatible with Python 3.13)
    model_name = "t5-small"  # ~250MB, works on 4GB GPU
    # Alternative: "t5-base" if you want better quality (~850MB)
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    
    # Enable FP16 for memory efficiency
    if device.type == 'cuda':
        model = model.half()
        print("✅ Using FP16 for memory efficiency")
    
    BATCH_SIZE = 32  # Can use larger batches with t5-small
    NUM_PARAPHRASES = 2
    
    def paraphrase_batch(texts, original_sample_ids, batch_size=BATCH_SIZE, num_sequences=NUM_PARAPHRASES):
        """Paraphrase using T5's text generation capability and track original IDs"""
        paraphrased_records = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Paraphrasing"):
            batch_texts = texts[i:i+batch_size]
            batch_ids = original_sample_ids[i:i+batch_size]
            
            # T5 prompt for paraphrasing
            input_texts = [f"paraphrase: {t}" for t in batch_texts]
            
            # Tokenize
            encoding = tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **encoding,
                    max_length=128,
                    num_beams=4,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    num_return_sequences=num_sequences,
                    no_repeat_ngram_size=3
                )
            
            # Decode
            for j in range(len(batch_texts)):
                original_id = batch_ids[j]
                for k in range(num_sequences):
                    idx = j * num_sequences + k
                    paraphrased_text = tokenizer.decode(outputs[idx], skip_special_tokens=True)
                    paraphrased_text = ' '.join(paraphrased_text.split())
                    
                    if paraphrased_text and paraphrased_text != batch_texts[j] and len(paraphrased_text) > 10:
                        paraphrased_records.append({
                            "text": paraphrased_text,
                            "original_id": original_id
                        })
            
            # Clear cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        return paraphrased_records

    # ============================================
    # 3. GENERATE PARAPHRASES
    # ============================================
    print("\n🔄 Generating paraphrases...")
    paraphrased_data = paraphrase_batch(original_texts, original_ids)
    
    # Convert to DataFrame
    paraphrased_df = pd.DataFrame(paraphrased_data)
    print(f"Generated {len(paraphrased_df)} valid paraphrases.")
    
    # ============================================
    # 4. MERGE AND SAVE
    # ============================================
    print("\n💾 Merging and saving data...")
    
    # Assign new sample_id to paraphrased data
    max_existing_id = df['sample_id'].max()
    paraphrased_df['sample_id'] = range(max_existing_id + 1, max_existing_id + 1 + len(paraphrased_df))
    
    # Set metadata for paraphrased samples
    paraphrased_df['method'] = 'paraphrased'
    paraphrased_df['label'] = 'disgust' # All are disgust
    
    # Add original_id to the original dataframe for schema consistency
    df['original_id'] = np.nan

    # Combine original and paraphrased dataframes
    # Ensure columns match before concatenation
    final_df = pd.concat([df, paraphrased_df], ignore_index=True)
    
    # Reorder columns to a clean format
    final_df = final_df[['sample_id', 'original_id' , 'text', 'label', 'method']]
    
    # Convert original_id to a nullable integer type to handle NaNs
    final_df['original_id'] = final_df['original_id'].astype(pd.Int64Dtype())

    # Save to CSV
    output_path = "../data/raw/disgust_paraphrased.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Successfully saved {len(final_df)} total samples to {output_path}")
    print("\n📊 Final DataFrame sample:")
    print(final_df.tail(10))

if __name__ == "__main__":
    disgust_paraphrase()
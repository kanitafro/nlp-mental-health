"""
GPU-Optimized Paraphrasing Augmentation for Disgust Dataset
Using t5-small for better compatibility
"""

# run this to create the paraphrased dataset:
# python paraphrase_disgust.py

import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def disgust_paraphrase():
    # Set device
    print("🚀 Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ============================================
    # 1. LOAD DATA
    # ============================================
    print("\n📂 Loading disgust_original.csv...")
    df = pd.read_csv('../data/raw/disgust_original.csv')
    original_texts = df['text'].tolist()
    print(f"Loaded {len(original_texts)} original disgust instances")
    
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
    
    def paraphrase_batch(texts, batch_size=BATCH_SIZE, num_sequences=NUM_PARAPHRASES):
        """Paraphrase using T5's text generation capability"""
        all_paraphrased = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Paraphrasing"):
            batch = texts[i:i+batch_size]
            
            # T5 prompt for paraphrasing
            input_texts = [f"paraphrase: {t}" for t in batch]
            
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
            for j in range(len(batch)):
                for k in range(num_sequences):
                    idx = j * num_sequences + k
                    paraphrased = tokenizer.decode(outputs[idx], skip_special_tokens=True)
                    paraphrased = ' '.join(paraphrased.split())
                    
                    if paraphrased and paraphrased != batch[j] and len(paraphrased) > 10:
                        all_paraphrased.append(paraphrased)
            
            # Clear cache
            if i % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
        
        return all_paraphrased
    
    # Run paraphrasing
    print(f"\n🔄 Generating {NUM_PARAPHRASES} paraphrases per text...")
    paraphrased_texts = paraphrase_batch(original_texts)
    print(f"✅ Paraphrasing complete: {len(paraphrased_texts)} examples")
    
    # ============================================
    # 3. COMBINE AND SAVE
    # ============================================
    print("\n📊 Combining all examples...")
    
    augmented_data = []
    
    # Add original
    for text in original_texts:
        augmented_data.append({'text': text, 'label': 'disgust', 'method': 'original'})
    
    # Add paraphrased
    for text in paraphrased_texts:
        if text and text.strip():
            augmented_data.append({'text': text, 'label': 'disgust', 'method': 't5_paraphrase'})
    
    df_augmented = pd.DataFrame(augmented_data)
    df_augmented = df_augmented.drop_duplicates(subset=['text'], keep='first')
    
    # Save
    output_file = '../data/raw/disgust_paraphrased.csv'
    df_augmented.to_csv(output_file, index=False)
    
    # Summary
    print("\n" + "="*50)
    print("✅ AUGMENTATION COMPLETE!")
    print("="*50)
    print(f"\n📁 Saved to: {output_file}")
    print(f"\n📊 Summary:")
    print(f"   Original:           {len(original_texts):,}")
    print(f"   Paraphrased:        {len(paraphrased_texts):,}")
    print(f"   Total unique:       {len(df_augmented):,}")
    print(f"\n📈 Augmentation factor: {len(df_augmented)/len(original_texts):.2f}x")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🧹 GPU cache cleared")

if __name__ == "__main__":
    disgust_paraphrase()
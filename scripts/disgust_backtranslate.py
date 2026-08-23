"""
Server-Optimized Back-Translation for Disgust Dataset
Designed for multi-GPU university server
NOW WITH PER-LANGUAGE TRACKING
"""

# dependencies:
# pip install pandas torch transformers tqdm sentencepiece numpy sys argparse

# run this to recreate the back-translated dataset:
# python disgust_backtranslate.py --model_type m2m100 --languages fr es de

import pandas as pd
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse
import numpy as np

# Add project root to path
import sys
from pathlib import Path
sys.path.append('../')
from scripts.merge_to_6labels_7labels import create_ids

path_paraphrased = "../data/raw/disgust_paraphrased.csv"

# ============================================
# CONFIGURATION
# ============================================
BATCH_SIZE = 128  # L40S can handle this
INTERMEDIATE_LANGUAGES = ['fr', 'de', 'es', 'it', 'zh']  # Default languages
NUM_WORKERS = 4

def setup_gpu():
    """Configure GPU for optimal performance"""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available on server!")
    
    num_gpus = torch.cuda.device_count()
    print(f"🎮 Found {num_gpus} GPU(s):")
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    device = torch.device("cuda:0")
    return device, num_gpus

class FastBackTranslator:
    """Optimized back-translation using M2M100 (best quality)"""
    
    def __init__(self, device, model_name='facebook/m2m100_418M'):
        print(f"\n🔧 Loading back-translation model: {model_name}")
        self.device = device
        
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()
        self.model = self.model.half()  # FP16
        print("✅ Model loaded with FP16 optimization")
    
    def translate_batch(self, texts, src_lang='en', tgt_lang='fr'):
        """Batch translation with M2M100"""
        self.tokenizer.src_lang = src_lang
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang),
                max_length=128,
                num_beams=4,
                temperature=1.0,
                no_repeat_ngram_size=3
            )
        
        translations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translations
    
    def backtranslate_with_tracking(self, texts, original_ids, intermediate_langs=['fr'], src_lang='en'):
        """
        Back-translate with language and original ID tracking.
        Returns list of (text, language, original_id) tuples.
        """
        all_results = []  # Each element will be (text, lang, original_id)
        
        for lang in intermediate_langs:
            print(f"   → {src_lang} to {lang} to {src_lang}")
            
            # Process in batches to avoid OOM on large datasets
            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Translating to {lang}"):
                batch_texts = texts[i:i+BATCH_SIZE]
                batch_ids = original_ids[i:i+BATCH_SIZE]

                # Forward translation
                forward = self.translate_batch(batch_texts, src_lang=src_lang, tgt_lang=lang)
                
                # Back translation
                backward = self.translate_batch(forward, src_lang=lang, tgt_lang=src_lang)
                
                # Add to results with language tag and original ID
                for j, text in enumerate(backward):
                    if text and text.strip() and len(text) > 10 and text.lower() != batch_texts[j].lower():
                        all_results.append((text, lang, batch_ids[j]))

        return all_results

class LightweightBackTranslator:
    """Alternative using OPUS-MT (faster, slightly lower quality)"""
    
    def __init__(self, device):
        print(f"\n🔧 Loading lightweight back-translation models...")
        self.device = device
        self.models = {}
        self.reverse_models = {}
        
        # Language pairs (forward models)
        language_pairs = {
            'fr': 'Helsinki-NLP/opus-mt-en-fr',
            'de': 'Helsinki-NLP/opus-mt-en-de',
            'es': 'Helsinki-NLP/opus-mt-en-es',
            'ru': 'Helsinki-NLP/opus-mt-en-ru',
            'zh': 'Helsinki-NLP/opus-mt-en-zh',
        }
        
        # Reverse models
        reverse_models = {
            'fr': 'Helsinki-NLP/opus-mt-fr-en',
            'de': 'Helsinki-NLP/opus-mt-de-en',
            'es': 'Helsinki-NLP/opus-mt-es-en',
            'ru': 'Helsinki-NLP/opus-mt-ru-en',
            'zh': 'Helsinki-NLP/opus-mt-zh-en',
        }
        
        # Load only requested languages (will be filtered in backtranslate)
        self.language_pairs = language_pairs
        self.reverse_models_dict = reverse_models
        
        print("✅ Models will be loaded on-demand")
    
    def load_model(self, lang, direction='forward'):
        """Lazy load models only when needed"""
        try:
            if direction == 'forward':
                if lang not in self.models:
                    model_name = self.language_pairs.get(lang)
                    if not model_name:
                        return None
                    print(f"   Loading {lang} model...")
                    self.models[lang] = {
                        'tokenizer': AutoTokenizer.from_pretrained(model_name),
                        'model': AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
                    }
                    self.models[lang]['model'].eval()
                    self.models[lang]['model'] = self.models[lang]['model'].half()
                return self.models[lang]
            else:  # backward
                if 'reverse' not in self.models:
                    self.models['reverse'] = {}
                if lang not in self.models['reverse']:
                    model_name = self.reverse_models_dict.get(lang)
                    if not model_name:
                        return None
                    print(f"   Loading {lang}→EN reverse model...")
                    self.models['reverse'][lang] = {
                        'tokenizer': AutoTokenizer.from_pretrained(model_name),
                        'model': AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
                    }
                    self.models['reverse'][lang]['model'].eval()
                    self.models['reverse'][lang]['model'] = self.models['reverse'][lang]['model'].half()
                return self.models['reverse'][lang]
        except Exception as e:
            print(f"   Warning: Could not load {lang} model: {e}")
            return None
    
    def translate_batch(self, texts, lang, direction='forward'):
        """Translate a batch using specific language model"""
        model_info = self.load_model(lang, direction)
        if not model_info:
            return texts
        
        try:
            encoded = model_info['tokenizer'](
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated = model_info['model'].generate(
                    **encoded,
                    max_length=128,
                    num_beams=4,
                    temperature=1.0
                )
            
            translations = model_info['tokenizer'].batch_decode(generated, skip_special_tokens=True)
            return translations
        except Exception as e:
            print(f"Error translating {lang}: {e}")
            return [] # Return empty list on error

    def backtranslate_with_tracking(self, texts, original_ids, intermediate_langs=['fr']):
        """
        Back-translate with language and original ID tracking using lightweight models.
        Returns list of (text, language, original_id) tuples.
        """
        all_results = []
        
        for lang in intermediate_langs:
            if lang not in self.language_pairs:
                print(f"Warning: Language '{lang}' not supported by lightweight translator. Skipping.")
                continue
                
            print(f"   → EN to {lang} to EN (lightweight)")

            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Translating to {lang} (light)"):
                batch_texts = texts[i:i+BATCH_SIZE]
                batch_ids = original_ids[i:i+BATCH_SIZE]

                # Forward and back translation
                forward = self.translate_batch(batch_texts, lang, direction='forward')
                backward = self.translate_batch(forward, lang, direction='backward')
                
                for j, text in enumerate(backward):
                    if text and text.strip() and len(text) > 10 and text.lower() != batch_texts[j].lower():
                        all_results.append((text, lang, batch_ids[j]))

        return all_results

def main(args):
    """Main function to run back-translation"""
    
    # ============================================
    # 1. SETUP
    # ============================================
    device, num_gpus = setup_gpu()
    
    if args.model_type == 'm2m100':
        translator = FastBackTranslator(device)
    else:
        translator = LightweightBackTranslator(device)
        
    # ============================================
    # 2. LOAD DATA
    # ============================================
    print(f"\n📂 Loading data from {path_paraphrased}...")
    df_paraphrased = pd.read_csv(path_paraphrased)
    
    # Prepare data for translation (only originals)
    original_df = df_paraphrased[df_paraphrased['method'] == 'original'].copy()
    original_texts = original_df['text'].tolist()
    original_ids = original_df['sample_id'].tolist()
    
    print(f"Found {len(original_texts)} original samples to back-translate.")
    
    # ============================================
    # 3. BACK-TRANSLATE
    # ============================================
    print(f"\n🔄 Back-translating with languages: {args.languages}")
    
    # This now returns (text, lang, original_id)
    backtranslated_results = translator.backtranslate_with_tracking(
        original_texts,
        original_ids,
        intermediate_langs=args.languages
    )
    
    if not backtranslated_results:
        print("\nNo valid back-translations were generated. Exiting.")
        return

    # Create a new DataFrame for the back-translated data
    backtranslated_df = pd.DataFrame(backtranslated_results, columns=['text', 'lang', 'original_id'])
    
    # Create the 'method' column
    backtranslated_df['method'] = 'backtrans_' + backtranslated_df['lang']
    backtranslated_df['label'] = 'disgust' # All are disgust
    
    print(f"\nGenerated {len(backtranslated_df)} new back-translated samples.")
    
    # ============================================
    # 4. MERGE AND SAVE
    # ============================================
    print("\n💾 Merging and saving data...")
    
    # Assign new sample_ids starting from the max of the loaded dataframe
    max_existing_id = df_paraphrased['sample_id'].max()
    backtranslated_df['sample_id'] = range(max_existing_id + 1, max_existing_id + 1 + len(backtranslated_df))
    
    # Combine the original dataframe with the new synthetic one
    final_df = pd.concat([df_paraphrased, backtranslated_df], ignore_index=True)
    
    # Reorder and select final columns
    final_df = final_df[['sample_id', 'original_id', 'text', 'label', 'method']]

    # Convert original_id to a nullable integer type to handle NaNs
    final_df['original_id'] = final_df['original_id'].astype(pd.Int64Dtype())
    
    # Save to a new back-translated file
    output_path = "../data/raw/disgust_backtranslated.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Successfully saved {len(final_df)} total samples to {output_path}")
    print("\n📊 Final DataFrame sample (showing new back-translated entries):")
    print(final_df[final_df['method'].str.startswith('backtrans')].tail(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast Back-Translation for Server')
    parser.add_argument('--input', type=str, default='../data/raw/disgust_original.csv')
    parser.add_argument('--output', type=str, default='../data/raw/disgust_backtranslated.csv')
    parser.add_argument('--model_type', type=str, choices=['m2m100', 'lightweight'], default='lightweight')
    parser.add_argument('--languages', nargs='+', default=INTERMEDIATE_LANGUAGES, 
                       help='Intermediate languages (e.g., fr de es ru zh)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    main(args)
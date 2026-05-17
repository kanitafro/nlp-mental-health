"""
Server-Optimized Back-Translation for Disgust Dataset
Designed for multi-GPU university server
NOW WITH PER-LANGUAGE TRACKING
"""

# dependencies:
# pip install pandas torch transformers tqdm sentencepiece numpy

# run this to recreate the back-translated dataset:
# python disgust_backtranslate.py --model_type m2m100 --languages fr es de

import pandas as pd
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse
import numpy as np

# ============================================
# CONFIGURATION
# ============================================
BATCH_SIZE = 128  # Your L40S can handle this
INTERMEDIATE_LANGUAGES = ['fr', 'de', 'es']  # Default languages
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
    
    def backtranslate_with_tracking(self, texts, intermediate_langs=['fr'], src_lang='en'):
        """
        Back-translate with language tracking
        Returns list of (text, language) tuples
        """
        all_results = []  # Each element will be (text, lang)
        
        for lang in intermediate_langs:
            print(f"   → {src_lang} to {lang} to {src_lang}")
            
            # Forward translation
            forward = self.translate_batch(texts, src_lang=src_lang, tgt_lang=lang)
            
            # Back translation
            backward = self.translate_batch(forward, src_lang=lang, tgt_lang=src_lang)
            
            # Add to results with language tag
            for text in backward:
                if text and text.strip():
                    all_results.append((text, lang))
        
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
            return texts
    
    def backtranslate_with_tracking(self, texts, intermediate_langs=['fr', 'de', 'es']):
        """
        Multi-language back-translation with tracking
        Returns list of (text, language) tuples
        """
        all_results = []  # Each element will be (text, lang)
        
        for lang in intermediate_langs:
            print(f"   → English → {lang.upper()} → English")
            
            # Forward translation
            forward = self.translate_batch(texts, lang, 'forward')
            
            # Back translation
            backward = self.translate_batch(forward, lang, 'backward')
            
            # Add to results with language tag
            for text in backward:
                if text and text.strip() and len(text) > 10:
                    all_results.append((text, lang))
        
        return all_results

def disgust_backtranslate():
    parser = argparse.ArgumentParser(description='Fast Back-Translation for Server')
    parser.add_argument('--input', type=str, default='disgust_bonus.csv')
    parser.add_argument('--output', type=str, default='disgust_backtranslated.csv')
    parser.add_argument('--model_type', type=str, choices=['m2m100', 'lightweight'], default='lightweight')
    parser.add_argument('--languages', nargs='+', default=['fr', 'de', 'es'], 
                       help='Intermediate languages (e.g., fr de es ru zh)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 SERVER-OPTIMIZED BACK-TRANSLATION (WITH LANGUAGE TRACKING)")
    print("="*60)
    
    # Setup
    device, num_gpus = setup_gpu()
    
    # Load data
    print(f"\n📂 Loading {args.input}...")
    df = pd.read_csv(args.input)
    original_texts = df['text'].tolist()
    print(f"Loaded {len(original_texts):,} original disgust instances")
    
    # Initialize translator
    if args.model_type == 'm2m100':
        translator = FastBackTranslator(device)
        backtranslate_func = translator.backtranslate_with_tracking
    else:
        translator = LightweightBackTranslator(device)
        backtranslate_func = translator.backtranslate_with_tracking
    
    # Back-translate
    print(f"\n🔄 Running back-translation through: {', '.join(args.languages)}")
    
    # Process in batches
    all_results_with_lang = []  # List of (text, language)
    batch_size = args.batch_size
    
    for i in tqdm(range(0, len(original_texts), batch_size), desc="Back-translation"):
        batch = original_texts[i:i+batch_size]
        
        # Translate batch (returns list of (text, lang) tuples)
        batch_results = backtranslate_func(batch, intermediate_langs=args.languages)
        all_results_with_lang.extend(batch_results)
        
        # Clear cache between batches
        if i % (batch_size * 5) == 0:
            torch.cuda.empty_cache()
    
    print(f"✅ Back-translation complete: {len(all_results_with_lang):,} examples")
    
    # Combine with original
    print("\n📊 Combining and saving...")
    augmented_data = []
    
    # Add original
    for text in original_texts:
        augmented_data.append({
            'text': text, 
            'label': 'disgust', 
            'method': 'original',
            'language': 'English (original)'
        })
    
    # Add back-translated with per-language tracking
    for text, lang in all_results_with_lang:
        if text and text.strip() and len(text) > 10:
            augmented_data.append({
                'text': text, 
                'label': 'disgust', 
                'method': f'backtrans_{lang}',  # e.g., 'backtrans_fr', 'backtrans_de'
                'language': lang
            })
    
    df_augmented = pd.DataFrame(augmented_data)
    df_augmented = df_augmented.drop_duplicates(subset=['text'], keep='first')
    
    # Save
    df_augmented.to_csv(args.output, index=False)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ BACK-TRANSLATION COMPLETE!")
    print("="*60)
    print(f"\n📁 Saved to: {args.output}")
    print(f"\n📊 Final counts:")
    print(f"   Original:           {len(original_texts):,}")
    print(f"   Back-translated:    {len(all_results_with_lang):,}")
    print(f"   Total unique:       {len(df_augmented):,}")
    print(f"\n📈 Augmentation factor: {len(df_augmented)/len(original_texts):.2f}x")
    
    # Show breakdown by language
    print(f"\n📊 Breakdown by method:")
    print(df_augmented['method'].value_counts().to_string())
    
    if 'language' in df_augmented.columns:
        print(f"\n📊 Breakdown by language (back-translated only):")
        lang_counts = df_augmented[df_augmented['language'].notna()]['language'].value_counts()
        print(lang_counts.to_string())
    
    # Sample
    print("\n🎯 Sample back-translated examples by language:")
    print("-"*60)
    for lang in args.languages:
        samples = df_augmented[df_augmented['language'] == lang]['text'].head(2)
        if len(samples) > 0:
            print(f"\n[{lang.upper()}]:")
            for sample in samples:
                print(f"   {sample[:100]}...")
    
    torch.cuda.empty_cache()
    print("\n🧹 GPU cache cleared")

if __name__ == "__main__":
    disgust_backtranslate()
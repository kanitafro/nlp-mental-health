'''
Used for LogReg
'''

import re
import joblib
import math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom vectorizer components
from preprocessing.tokenizer import CustomTokenizer, fit_tfidf_vectorizer

def create_smote_strategy(y_train):
    """
    Create SMOTE sampling strategy based on training data distribution
    """
    train_distribution = Counter(y_train)
    print("\n📊 Training set distribution:")
    for emotion, count in train_distribution.items():
        print(f"   {emotion}: {count} samples")

    fear_count = train_distribution.get('fear', 0)
    target_strategy = {
        'surprise': fear_count,
        'love': fear_count,
        'fear': fear_count,
        'anger': train_distribution.get('anger', 0),
        'joy': train_distribution.get('joy', 0),
        'sadness': train_distribution.get('sadness', 0)
    }

    final_target_strategy = {}
    for emotion, target in target_strategy.items():
        if emotion in train_distribution:
            current_count = train_distribution[emotion]
            if target >= current_count:
                final_target_strategy[emotion] = target
            else:
                final_target_strategy[emotion] = current_count
        
    smote_sampling_strategy = {
        cls: target for cls, target in final_target_strategy.items() 
        if target > train_distribution.get(cls, 0)
    }
    
    print("\n🎯 SMOTE target strategy:")
    for emotion, target in smote_sampling_strategy.items():
        current = train_distribution[emotion]
        print(f"   {emotion}: {current} → {target} ({(target/current-1)*100:+.1f}%)")
    
    return smote_sampling_strategy

def apply_smote_oversampling(X_train_tfidf, y_train, sampling_strategy, random_state=42):
    """
    Apply SMOTE oversampling to training data
    """
    smote = SMOTE(
        random_state=random_state,
        sampling_strategy=sampling_strategy,
        k_neighbors=3
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)
    
    print("✅ SMOTE oversampling applied successfully!")
    print("📊 Resampled distribution:")
    resampled_distribution = Counter(y_resampled)
    for emotion in sampling_strategy.keys():
        original = Counter(y_train).get(emotion, 0)
        resampled = resampled_distribution.get(emotion, 0)
        if original > 0:
            print(f"   {emotion}: {original} → {resampled} ({(resampled/original-1)*100:+.1f}%)")
    
    return X_resampled, y_resampled, smote

def analyze_smote_memory_efficient(X_original, y_original, X_resampled, y_resampled, vectorizer, target_classes, n_samples=50):
    """
    Memory-efficient analysis of SMOTE-generated data
    """
    print("\n" + "="*60)
    print("🔍 SMOTE ANALYSIS (Memory Efficient)")
    print("="*60)
    
    results = {}
    
    for target_class in target_classes:
        print(f"\n🎯 Analyzing class: {target_class}")
        print("-" * 40)
        
        # Get indices for this class (using sparse matrix operations)
        original_mask = np.array(y_original) == target_class
        synthetic_mask = (np.array(y_resampled) == target_class) & (np.arange(len(y_resampled)) >= len(y_original))
        
        original_indices = np.where(original_mask)[0]
        synthetic_indices = np.where(synthetic_mask)[0]
        
        if len(synthetic_indices) == 0:
            print(f"   No synthetic samples generated for {target_class}")
            continue
            
        print(f"   Original samples: {len(original_indices)}")
        print(f"   Synthetic samples: {len(synthetic_indices)}")
        
        # Sample a subset to avoid memory issues
        n_analyze = min(n_samples, len(synthetic_indices))
        sampled_synthetic_indices = np.random.choice(synthetic_indices, n_analyze, replace=False)
        
        similarities = []
        feature_analysis = []
        
        # Analyze each synthetic sample
        for i, synth_idx in enumerate(sampled_synthetic_indices):
            if i % 20 == 0:
                print(f"   Processing sample {i+1}/{n_analyze}...")
            
            # Get synthetic sample (sparse)
            synth_vector = X_resampled[synth_idx]
            
            # Find most similar original using sparse operations
            best_similarity = 0
            
            # Process original samples in batches
            batch_size = 1000
            for batch_start in range(0, len(original_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(original_indices))
                batch_indices = original_indices[batch_start:batch_end]
                
                # Get batch of original vectors
                orig_batch = X_original[batch_indices]
                
                # Compute similarities (this is memory efficient with sparse matrices)
                batch_similarities = cosine_similarity(synth_vector, orig_batch)[0]
                batch_best = np.max(batch_similarities)
                
                if batch_best > best_similarity:
                    best_similarity = batch_best
            
            similarities.append(best_similarity)
            
            # Analyze top features for a few samples
            if i < 3:  # Only for first 3 samples to save memory
                # Convert to dense only for this small sample
                synth_dense = synth_vector.toarray().flatten()
                top_feature_indices = np.argsort(synth_dense)[-5:][::-1]
                top_features = [(vectorizer.get_feature_names_out()[idx], synth_dense[idx]) 
                               for idx in top_feature_indices if synth_dense[idx] > 0]
                feature_analysis.append({
                    'similarity': best_similarity,
                    'top_features': top_features
                })
        
        # Store results
        results[target_class] = {
            'similarities': similarities,
            'feature_analysis': feature_analysis,
            'n_original': len(original_indices),
            'n_synthetic': len(synthetic_indices)
        }
        
        # Print summary
        print(f"   Similarity (cosine) - {n_analyze} samples:")
        print(f"     Mean: {np.mean(similarities):.4f}")
        print(f"     Std:  {np.std(similarities):.4f}")
        print(f"     Min:  {np.min(similarities):.4f}")
        print(f"     Max:  {np.max(similarities):.4f}")
        
        # Show example features
        if feature_analysis:
            print(f"\n   Example synthetic samples:")
            for i, example in enumerate(feature_analysis):
                print(f"     Sample {i+1}: Similarity={example['similarity']:.4f}")
                print(f"       Top features: {example['top_features']}")
    
    return results

def visualize_smote_efficient(results):
    """
    Memory-efficient visualization of SMOTE results
    """
    if not results:
        print("No results to visualize")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Similarity distribution by class
    similarity_data = []
    for class_name, class_results in results.items():
        for similarity in class_results['similarities']:
            similarity_data.append({'class': class_name, 'similarity': similarity})
    
    similarity_df = pd.DataFrame(similarity_data)
    
    if not similarity_df.empty:
        sns.boxplot(data=similarity_df, x='class', y='similarity', ax=axes[0])
        axes[0].set_title('SMOTE: Cosine Similarity Distribution')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    
    # Plot 2: Class distribution changes
    classes = list(results.keys())
    original_counts = [results[cls]['n_original'] for cls in classes]
    synthetic_counts = [results[cls]['n_synthetic'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[1].bar(x - width/2, original_counts, width, label='Before SMOTE', alpha=0.7)
    axes[1].bar(x + width/2, synthetic_counts, width, label='After SMOTE', alpha=0.7)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Sample Count')
    axes[1].set_title('Class Distribution: Before vs After SMOTE')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=45)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return similarity_df

def analyze_existing_smote_model(model_path, df, n_samples=30):
    """
    Analyze an already trained SMOTE model without retraining
    """
    print("🔍 Analyzing existing SMOTE model...")
    
    # Load model
    pipeline = joblib.load(model_path)
    
    # Prepare data
    df = df.dropna(subset=["clean_text_ml", "label"])
    texts = df["clean_text_ml"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    
    # Use a small subset for analysis
    if len(texts) > 10000:
        texts = texts[:10000]
        labels = labels[:10000]
        print("   Using subset of 10,000 samples for analysis")
    
    # Transform data
    X_tfidf = pipeline.named_steps['tfidf'].transform(texts)
    y = labels
    
    # Apply SMOTE separately for analysis
    smote = pipeline.named_steps['smote']
    X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)
    
    # Analyze
    target_classes = ['love', 'surprise']
    results = analyze_smote_memory_efficient(
        X_tfidf, y, X_resampled, y_resampled,
        pipeline.named_steps['tfidf'], target_classes, n_samples
    )
    
    visualize_smote_efficient(results)
    return results

# Journal-based Emotion Recognition and Risk Detection using BERT

### **Author**: Kanita Tafro 

_Faculty of Electrical Engineering, University of Sarajevo_

## Table of Contents

- [Journal-based Emotion Recognition and Risk Detection using BERT](#journal-based-emotion-recognition-and-risk-detection-using-bert)
  - [Project Description](#project-description)
  - [Repository Structure](#repository-structure)
  - [Training and Inference Pipeline](#training-and-inference-pipeline)
    - [Training Phase](#training-phase)
    - [Inference Phase](#inference-phase)
  - [Joint Emotion Recognition and Risk Detection](#joint-emotion-recognition-and-risk-detection)
    - [Emotion Recognition](#emotion-recognition)
    - [Risk Detection](#risk-detection)
  - [Model Architecture](#model-architecture)
  - [Training Objective](#training-objective)
  - [Disclaimer](#disclaimer)
  - [Instructions](#instructions)
  - [Results](#results)
    - [Emotion Classification](#emotion-classification)
    - [Risk Detection](#risk-detection-1)
  - [Inference](#inference)
  - [Lenient Emotion Decoding](#lenient-emotion-decoding)
    - [Strategy](#strategy)
  - [Uncertainty-Aware Emotion Interpretation](#uncertainty-aware-emotion-interpretation)
  - [Risk Thresholding: Non-Arbitrary Design](#risk-thresholding-non-arbitrary-design)
    - [Clinical Framing](#clinical-framing)
  - [Evidence-Augmented Interpretation Gates](#evidence-augmented-interpretation-gates)
    - [Depression-Specific Handling](#depression-specific-handling)
  - [Post-hoc Textual Grounding](#post-hoc-textual-grounding)
  - [Subtheme Detection and Disclosure Filtering](#subtheme-detection-and-disclosure-filtering)
    - [Disclosure Threshold](#disclosure-threshold)
    - [Lexical Gating](#lexical-gating)
  - [Design Principles](#design-principles)
  - [Future Work](#future-work)


## Project Description
Build a three-layer NLP system for journal entries:
* Layer 1: Emotion recognition → 6/28 emotions (starting with 6)
* Layer 2: Risk labeling →  recognize if there are any risk flags (suicidal, self-harm, depression, grief)
* Layer 3: Theme detection → 12 major themes (Mental health, Grief, Spirituality, etc.) with their exact subthemes
  
The model trains on emotions and risk flags. Themes/subthemes are inferred after the model predicts emotions, using the lexicon and SubthemeInferencer located in _bert/lexicon_utils.py_.

Ultimately, the model will:
* Take a text input (like “I feel so lost but I prayed today.”)
* Predict:
   * **Emotions**: sadness
   * **Risk flags**: depression (or None)
* Infer:
   * **Themes**: depression, religion, spirituality
* Later be integrated into a lightweight mobile app (through API)

## Repository Structure

```
├── bert/
│   ├── dataset.py                       # dataset class
│   ├── inference.py                     # inference logic functions
│   ├── lenient_decoding.py         # for love and surprise only
│   ├── lexicon_utils.py                 # loading and preparing lexicon for inference
│   ├── metrics.py                         # computing metrics for emotions and risks
│   ├── model_utils.py                  # model, optimizer, scheduler, early stopping
│   ├── multitask_model.py           # defining the multitasking
│   ├── test_inference.py               # RUN inference
│   ├── train.py                               # RUN training
│   ├── visualize_dashboard.py      # combined plots in 4 parts
│   ├── visualize_metrics.py            # for creating plots
│
├── data/
│   ├── __init__.py
│   ├── raw/                       # Original datasets (instructions for downloading data)
│   │   ├── more_surprise.csv
│   ├── processed/                 # Cleaned & preprocessed text data
│   ├── lexicon/
│   │   ├── lexicon_raw.py      # The raw Python dict (editable)
│   │   ├── build_lexicon.py      # load raw dict, save raw, clean it, then save cleaned
│   │   ├── lexicon.csv     # Original theme lexicon (get from build_lexicon)
│   │   ├── lexicon.json
│   │   ├── lexicon_clean.csv   # Cleaned lexicon  (get from build_lexicon)
│   │   ├── lexicon_clean.json
│   │   ├── lexicon_clean_6.csv   # 28 to 6 emotions (get from map_emotion_to_lexicon)
│   │   ├── lexicon_clean_6.json
│   ├── risk_labels/
│   │   ├── reddit_depression.csv    # manually scraped dataset
│   │   ├── reddit_grief.csv    # manually scraped dataset
│   │   ├── reddit_selfharm.csv    # manually scraped dataset
│   │   ├── reddit_suicidal.csv    # manually scraped dataset
│
├── models/          # traditional ML models (currently only has Logistic Regression)
│       ├── train_logreg_v1.py
│
├── notebooks/
│   ├── merge_love_surprise.ipynb     # RUN this notebook after fetching all emotion datasets
│
├── preprocessing/
│   ├── clean_text.py              # Full text-cleaning pipeline
│   ├── text_utils.py              # Helper functions (emoji map, slang, etc.)
│   ├── tokenizer.py        # Custom tokenizer (with NER and mode switches)
│   ├── clean_lexicon.py    # Cleans the theme lexicon automatically
│   ├── map_emotions.py      # mapping logic (28 to 6 emotions)
│
├── scripts/    	  # will be run separately before running main.py
│   ├── clean_dataset.py                         # from loading to cleaning dataset
│   ├── clean_risklabels.py	                   # cleans risk datasets (for transformer only)
│   ├── map_emotion_to_lexicon.py        # runs everything related to lexicon
│   ├── risk_ngrams.py 	                   # extracts n-grams from risk datasets (n=1,...,6)
│   
├── utils/
│   ├── __init__.py        
│   ├── file_io.py                           # load/save helpers
│   ├── smote_oversampling        # for ML models
│
├── requirements.txt
├── .gitignore
└── README.md

```

## Training and Inference Pipeline

### Training Phase
1. Tokenize input text
2. Encode using a shared BERT encoder
3. Compute emotion logits and risk logits
4. Optimize standard loss functions

### Inference Phase
1. Compute emotion probabilities (softmax)
2. Rank emotions by confidence
3. Apply **lenient emotion decoding** for semantically overlapping emotions
4. Output risk flags using learned classifiers
5. Apply **evidence-augmented interpretation gates**
6. Infer themes and subthemes using:
   - Emotion probabilities
   - Lexical evidence
   - Subtheme-specific constraints

## Joint Emotion Recognition and Risk Detection

### Emotion Recognition
- Single-label classification over:  
  $E = {joy, love, sadness, fear, surprise, anger}$

### Risk Detection
- Independent binary classification for each:   
  $R = {depression, selfharm, suicidal, grief}$

The two tasks share a common language representation while maintaining independent decision boundaries.


## Model Architecture

* **Shared Encoder**: A pretrained BERT encoder maps input text `x` to a contextual representation `h`.
* **Emotion Head**:
  - Softmax classifier
  - Outputs a probability distribution over emotions
* **Risk Heads**:
  - Independent sigmoid classifiers
  - Multiple risk flags may be active simultaneously


## Training Objective

* **Emotion Loss:** Categorical cross-entropy over emotion labels.
* **Risk Loss:** Binary cross-entropy per risk label, with optional masking for missing annotations.
* **Joint Optimization:** The total loss is a weighted sum of emotion loss and all risk losses, allowing task balancing.

---


## Disclaimer

This system is intended for **supportive and exploratory use only**.  
It is **not** a diagnostic or clinical decision-making tool.

---

## Instructions
1. After cloning this repository, download the datasets from the sources and rename the files as instructed in [*data/raw* README](https://github.com/kanitafro/pinpilinpauxa/tree/main/data/raw). The first dataset will be renamed to `dataset_6labels.csv` (6 labels) and will be merged with 5 other datasets to used in training. Put them all into *data/raw/* folder and name them as instructed. Merging of the 5 secondary datasets is done in the notebook *notebooks/merge_love_surprise.ipynb*.
2. Before running training, it is necessary to run the dataset cleaning pipeline in the *scripts* folder by typing this into the command prompt in the root directory (it will take around 10 minutes):
   
   ```
   python -m scripts.clean_dataset
   ```
   After this script finishes, the full dataset will be in *data/processed/dataset6_labels_clean_more.csv*. The emotions dataset is ready for training but there's one more component before that.
3. **RISK FLAG DATASETS**: There are 4 datasets in *data/risk_labels/* that are meant for risk flag detection. The only thing you need to do is to run the script from root of the directory in the terminal:
   
   ```
   python -m scripts.clean_risklabels
   ```
4. **TRAINING**: Run `bert/train.py` (to recreate the obtained results run the following command from the *bert/* folder):

   ```
   python train.py  --epochs 2 --batch_size 16 --accumulation_steps 2 --learning_rate 2.8e-5 --use_risk_flags --early_stopping --patience 1 --save_checkpoints --dropout_rate 0.2 --model_version v1_1
   ```
   By running this command, the model will train for 2 epochs for around 12 hours on a GPU. The final model as `v1_1_epoch_2.pt` will be saved to *bert/checkpoints_v1_1/*. All metrics will be saved to *bert/saved_models/trained_model_v1_1/*.  
5. **INFERENCE PREPARATION**: Run the lexicon build pipeline. It converts the raw Python dict into csv and json (`lexicon.csv`, `lexicon.json`) then it uses the clean_text() function from *utils* to lowercase the lexicon and strip it from punctuations (used in ML mode, not transformer) and outputs 2 new files (`lexicon_clean.csv`, `lexicon_clean.json`). Finally, it maps the 28 emotions to only 6 defined in *preprocessing/map_emotions.py* and outputs 2 new files (`lexicon_clean_6.csv`, `lexicon_clean_6.json`). All files will be located in *data/lexicon*. Type this into the command prompt in the root directory:

   ```
   python -m scripts.map_emotion_to_lexicon
   ```
6. **INFERENCE**: From the _bert/_ folder, run:

   ```
   python test_inference.py
   ```
   This will output the scores for emotions and risks, then inferred themes from the lexicon, and finally the human interpretation report in form of full sentences. Add any examples in the `texts` list located in main() of `test_inference.py`. Aside from the terminal output, the full report for the tested examples will be saved to _bert/json_files/test_inference_output.json_.
---

## RESULTS

### Emotion Classification

<img width="666" height="547" alt="Image" src="https://github.com/user-attachments/assets/9b470821-52ff-4434-9a9a-8c3c94d382a7" /> 

              		  precision    recall     f1-score   support

            anger  	    0.9097      0.9695    0.9387     17794
             fear	  	0.9615      0.8276    0.8895     14894
              joy   	0.9633      0.9092    0.9355     42920
             love    	0.7402      0.8855    0.8063     17207
          sadness    	0.9754      0.9400    0.9573     36355
         surprise       0.7556      0.8955    0.8196      6792

         accuracy 	                    	  0.9127    135962
        macro avg	    0.8843      0.9045    0.8912    135962
     weighted avg 	    0.9207      0.9127    0.9146    135962
   


### Risk Detection

<img width="691" height="547" alt="Image" src="https://github.com/user-attachments/assets/8d220bc9-2d3a-42b2-9619-00519391ba60" />

| Risk Category | Precision | Recall | F1-score | AUROC | Support |
|---------------|-----------|--------|----------|-------|---------|
| Depression    | 0.94      | 0.96   | 0.95     | 0.990 | 4,768   |
| Self-harm     | 1.00      | 0.99   | 0.993    | 0.999 | 592     |
| Suicidal      | 0.999     | 0.97   | 0.99     | 0.997 | 3,194   |
| Grief         | 0.99      | 0.97   | 0.98     | 0.998 | 820     |

#### Optimal thresholds found:
    "depression": 0.9,
    "selfharm": 0.09,
    "suicidal": 0.08,
    "grief": 0.06


---

## Inference

Inference shows ranked emotions (softmax output) and probabilities for each risk flag directly from the model. Then the themes are inferred using the theme lexicon (+ detected emotions) and the rest of the inference design is for defining reasoning behind the detected emotions and risks and to ultimately display them in humanly interpretative sentences.

## Lenient Emotion Decoding

Certain emotions exhibit **semantic overlap**:
- `surprise` ↔ `joy`, `fear`
- `love` ↔ `joy`

To avoid penalizing semantically valid ambiguity, lenient decoding is applied **only at inference/evaluation time**.

### Strategy
- Use top-k emotion rankings
- Accept structured emotion sets when overlap conditions are met
- A prediction is considered correct if the gold label appears in the accepted set

This converts strict multiclass evaluation into a **structured top-k acceptance criterion**.

---

## Uncertainty-Aware Emotion Interpretation

Short or low-signal text often yields diffuse emotion probabilities. To prevent over-interpretation:

- Predictions are categorized as:
  - **Weak**
  - **Mixed**
  - **Dominant**
- Based on:
  - Absolute confidence
  - Margin between top-1 and top-2 probabilities

This mechanism:
- Operates **only at the explanation layer**
- Does **not** alter training, inference, or evaluation

---

## Risk Thresholding: Non-Arbitrary Design

Risk classifiers output continuous probabilities. Decision thresholds are:

- **Not heuristic**
- **Not manually chosen**
- Optimized on validation data to maximize **F1-score**

Each risk category has its own optimal threshold, reflecting:
- Class imbalance
- Linguistic ambiguity
- Error cost asymmetry

### Clinical Framing
- Thresholds are **not diagnoses**
- They define statistically grounded disclosure points
- Designed for augmentation, not replacement, of professional judgment

---

## Evidence-Augmented Interpretation Gates

For risks other than depression, interpretation depends on:

- Model probability
- Learned optimal threshold
- Lexical evidence (risk-specific n-grams)
- Presence of aligned subthemes

Lexical evidence **never replaces** model confidence; it only modulates disclosure *above* admissible confidence levels.

### Depression-Specific Handling
- Reported as **graded intensity** (`None`, `Mild`, `Mid`, `High`)
- Uses higher thresholds to avoid diagnostic overreach
- Reflects high lexical overlap with normative distress

---

## Post-hoc Textual Grounding

When a risk flag is disclosed:
- The system highlights surface-level n-grams
- These are **not explanations of model reasoning**
- They provide **textual grounding for human inspection only**

This preserves a strict separation between:
- Learned inference
- Interpretive support

---

## Subtheme Detection and Disclosure Filtering

Subthemes are scored using **emotion-weighted lexical similarity**.

Key properties:
- Scores are **compatibility measures**, not probabilities
- Low scores may arise from incidental overlap

### Disclosure Threshold
- A fixed minimum score (`τ = 1.0`) is required
- Suppresses weak or noisy subthemes
- Improves interpretability and trust

### Lexical Gating
Some subthemes require **explicit lexical realization** and cannot activate on emotion alone.

Each subtheme declares:  
requires_lexical_evidence ∈ {0,1}


If required and no keywords are present, the subtheme score is forced to zero.  
This constraint is **typed and semantic**, not heuristic.


## Design Principles

- Clear separation between:
  - Learned prediction
  - Post-hoc interpretation
- No rule-based influence on model outputs
- Conservative handling of sensitive mental health signals
- Emphasis on transparency, interpretability, and ethical caution


## Future Work

- Expand emotion set from 6 → 28
- API deployment for mobile integration
- Longitudinal analysis across journal histories
- Personalization via user-calibrated thresholds




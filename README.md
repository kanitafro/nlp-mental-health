# Journal-based Emotion Recognition and Risk Detection using BERT

### **Author**: Kanita Tafro 

_Faculty of Electrical Engineering, University of Sarajevo_

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

## Training and Inference Pipeline

### Training Phase
1. Tokenize input text
2. Encode using a shared BERT encoder
3. Compute emotion logits and risk logits
4. Optimize standard loss functions
5. **No** lenient decoding
6. **No** theme or lexicon logic applied

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

## Results


---

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




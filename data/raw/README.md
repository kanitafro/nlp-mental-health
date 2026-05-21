# Dataset

The dataset used for training on 7 labels has more than 470k instances, but is consisted of a few datasets (1 full from Kaggle; then 7 sampled datasets: 6 from Kaggle and SemEval-2018; 1 custom synthetic dataset *`more_surprise.csv`*, and the augmented (paraphrased + backtranslated) disgust instances).


| Dataset | Instances | Joy | Sadness | Anger | Fear | Love | Surprise |  Disgust  |
|---------|-----------|-----|---------|-------|------|------|----------|-----------|
| [Base dataset](https://www.kaggle.com/datasets/kushagra3204/sentiment-and-emotion-analysis-dataset) | 422,566 | 143,067 | 121,187 | 59,137 | 49,649 | 34,554 | 14,972 |  0 |
| [LoSu 1](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) | 2,360 | 0 | 0 | 0 | 0 | 1,641 | 719 |  0 |
| [LoSu 2](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text) | 6,788 | 0 | 0 | 0 | 0 | 3,842 | 2,946 | 0 |
| [LoSu 3](https://www.kaggle.com/datasets/simaanjali/emotion-analysis-based-on-text/data) | 46,507 | 0 | 0 | 0 | 0 | 39,553 | 6,954 | 0  |
| [LoSu 4](https://www.kaggle.com/datasets/bryanhuerta/social-media-sentiment-data) | 57 | 0 | 0 | 0 | 0 | 26 | 31 |  0  |
| [Surprise synth](https://github.com/kanitafro/nlp-mental-health/blob/main/data/raw/) | 663 | 0 | 0 | 0 | 0 | 0 | 663 | 0 |
| [Disgust 1](https://www.kaggle.com/datasets/muhammadumarattique/text-for-emotions-classification-projests) | 1,722 | 0 | 0 | 0 | 0 | 0 | 0 | 1,722 |
| [ISEAR](https://huggingface.co/datasets/gsri-18/ISEAR-dataset-complete) | 1,066 | 0 | 0 | 0 | 0 | 0 | 0 | 1,066 |
| [SemEval-2018 Task E-c](https://www.kaggle.com/datasets/context/semeval-2018-task-ec) | 4,020 | 0 | 0 | 0 | 0 | 0 | 0 | 4,020 |
| Disgust Augmented* | 14,522 | 0 | 0 | 0 | 0 | 0 | 0 | 14,522 |
| **Total after cleaning****| **474,457** | **143,067** | **121,187** | **59,137** | **49,649** | **57,356** | **22,640** |  **21,241** |
| **Prevalence percentage**** |   | **30.15%** | **25.54%** | **12.50%** | **12.09%** | **10.46%** | **4.77%** |  **4.48%** |


*The 14k augmented disgust instances come from the scripts `disgust_paraphrase.py` and `disgust_backtranslate.py` that created new instances based on the original 5.6k collected from the last 3 linked datasets  
**The last row showing the total number of instances isn't a sum of each column but the final count after removing duplicates and missing values.

---

# Obtain the data

All instances are publicly available through 9 sources.

## **INSTRUCTIONS**:
1. Download the following 9 datasets (you will need to manually download 6 of them), name the files as instructed below, and put all the files in ***data/raw/***.

    a. **Base** (422k dataset with 6 labels) ([Kaggle](https://www.kaggle.com/datasets/kushagra3204/sentiment-and-emotion-analysis-dataset))
    * file name: data/raw/**dataset_6labels.csv**
    * Labels: joy, sadness, anger, fear, surprise, love

    b. **LoSu 1** (only 'love' and 'surprise') ([Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp))
    * file name: data/raw/**extra_dataset_Praveen.csv**
    * Labels: love, surprise

    c. **LoSu 2** (only 'love' and 'surprise') ([Kaggle](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text))
    * file name: data/raw/**extra_dataset_PashupatiGupta.csv**
    * Labels: love, surprise

    d. **LoSu 3** (only 'love' and 'surprise') ([Kaggle](https://www.kaggle.com/datasets/simaanjali/emotion-analysis-based-on-text/data))
    * file name: data/raw/**extra_dataset_SimaAnjali.csv**
    * Labels: love, surprise

    e. **LoSu 4** (only 'love' and 'surprise') ([Kaggle](https://www.kaggle.com/datasets/bryanhuerta/social-media-sentiment-data))
    * file name: data/raw/**extra_dataset_BryanHuerta.csv**
    * Labels: love, surprise

    f. **Surprise synth** (only 'surprise')
    * file exists in this folder: `more_surprise.csv`
    * 663 instances generated for surprise by prompting LLMs (ChatGPT and Deepseek) giving them special sublabels as guidance (e.g. confusion, excitement) and varying them in length.

    g. **Disgust 1** ([Kaggle](https://www.kaggle.com/datasets/muhammadumarattique/text-for-emotions-classification-projests))

    * folder name: data/raw/**muhammadummarattique_data**/
    * Put the unzipped data in the said folder (or just the *disgust/* folder here)
    * Label: disgust

    h. **ISEAR** ([Hugging Face](https://huggingface.co/datasets/gsri-18/ISEAR-dataset-complete))

    * no need to download it, step 2 will handle it
    * Label: disgust

    i. **SemEval-2018 Task E-c** ([Github link](https://github.com/cbaziotis/ntua-slp-semeval2018/tree/master/datasets/task1/E-c))

    * no need to download it, step 2 will handle it
    * Label: disgust

2. Run the notebook `merge_love_surprise_disgust.ipynb`
    * This will produce ***love_surprise_bonus.csv*** and ***disgust_bonus.csv***
3. After that, you will run the scripts `disgust_paraphrase.py` and `disgust_backtranslate.py` from the root directory
    
    ```
    python -m scripts.disgust_paraphrase
    python -m scripts.disgust_backtranslate --model_type m2m100
    ```
   * `disgust_backtranslate.py` requires either a bigger GPU or to run on CPU but it will be slower (***Note**: Running on CPU will require changing the code and possibly adding freeze support aside from decreasing the batch size, but I managed to run it on NVIDIA L40S GPU in 5 minutes.*)
   * This scripts will produce files ***data/raw/disgust_backtranslated.csv*** and ***data/raw/disgust_paraphrased.csv***.
4. Run `merge_to_6labels_7labels.py` to produce:
    * ***data/raw/dataset_6labels_more.csv***, and/or
    * ***data/raw/dataset_7labels.csv***
5. The raw datasets of 6/7 emotions are ready.



## Label prevalence in the merged dataset

| Emotion Label | Count   | Percentage | *Cumulative* |
|---------------|---------|------------|--------------|
| Joy           | 143,067 | 30.1538%   | *30.15%*     |
| Sadness       | 121,187 | 25.5423%   | *55.69%*     |
| Anger         | 59,317  | 12.5021%   | *68.20%*     |
| Love          | 57,356  | 12.0888%   | *80.28%*     |
| Fear          | 49,649  | 10.4644%   | *90.75%*     |
| Surprise      | 22,640  | 4.7718%    | *95.52%*     |
| Disgust       | 21,241  | 4.4769%    | *100.00%*    |
| **Total**     |**474,457**|            |              |
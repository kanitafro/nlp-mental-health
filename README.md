# Journal-based Emotion Recognition
!!!**not finished**!!!

This model trains on 6 emotions and 28 emotions.

## Instructions
1. After cloning this repository, download the datasets from the sources and rename the files as instructed in [*data/raw* README](https://github.com/kanitafro/pinpilinpauxa/tree/main/data/raw). The first dataset will be renamed to `dataset_6labels.csv` and the other to `goemotions.csv` as it's used in the code. Then put them into the *data/raw* folder.
2. Before running `main.py`, you need to run the dataset cleaning pipeline in the *scripts* folder by typing this into the command prompt in the root directory (it will take around 10 minutes):  

   ```
   python -m scripts.clean_dataset
   ```
3. Run the lexicon build pipeline. It converts the raw Python dict into csv and json (`lexicon.csv`, `lexicon.json`) then it uses the clean_text() function from *utils* to lowercase the lexicon and strip it from punctuations (used in ML mode, not transformer) and outputs 2 new files (`lexicon_clean.csv`, `lexicon_clean.json`). Finally, it maps the 28 emotions to only 6 defined in *preprocessing/map_emotions.py* and outputs 2 new files (`lexicon_clean_6.csv`, `lexicon_clean_6.json`). All files will be located in *data/lexicon*. Type this into the command prompt in the root directory:

   ```
   python -m scripts.map_emotion_to_lexicon
   ```
4. Run `main.py`:

   ```
   python main.py
   ```

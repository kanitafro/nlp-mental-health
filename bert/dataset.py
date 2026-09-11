# project-root/bert/dataset.py
import torch
import pandas as pd
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    Dataset for:
      - Emotion classification (single-label)
      - Optional multi-label risk flags (masked BCE)
    """

    RISK_ORDER = ["depression", "selfharm", "suicidal", "grief"]

    def __init__(
        self,
        df,
        tokenizer,
        max_len=128,
        label_count=6,
        lexicon=None,
        use_lexicon=False,
        use_risk=False,
        risk_name=None,  # NEW
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_risk = use_risk
        self.risk_name = risk_name

        self.texts = df["clean_text_transf"].astype(str).tolist()

        # Emotion labels (may be missing for risk-only datasets)
        self.emotion_labels = (
            df["label"].astype(int).tolist()
            if "label" in df.columns
            else None
        )

        # -----------------------
        # Risk labels (masked)
        # -----------------------
        if self.use_risk:
            risk_vecs = []
            ''' 
            This establishes the following behavior:
            1. Each dataset corresponds to exactly one risk task
              * risk_name is externally determined (e.g. "suicidal")
              * The CSV being loaded implicitly defines the task

            2. Only one risk head is active per dataset
              * One index in RISK_ORDER is filled
              * All others are set to -100.0 (ignored by loss)

            3. The model is trained with masked multitask supervision
              * This is the correct pattern for heterogeneous multitask datasets
              * PyTorch loss functions ignore -100 by convention
            '''
            for _, row in df.iterrows():
                vec = [-100.0] * 4  # mask all
                if risk_name is not None:
                    idx = self.RISK_ORDER.index(risk_name)
                    vec[idx] = float(row["label"])  # 0 or 1
                risk_vecs.append(vec)

            self.risk_labels = torch.tensor(risk_vecs, dtype=torch.float)
        else:
            self.risk_labels = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.emotion_labels is not None:
            item["emotion_labels"] = torch.tensor(
                self.emotion_labels[idx], dtype=torch.long
            )

        if self.use_risk:
            item["risk_labels"] = self.risk_labels[idx]

        return item

# ---------------------------------------------------------------------
# Multi-label Dataset
# ---------------------------------------------------------------------
class MultiLabelGoEmotionsDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_length,
        label_columns,
        text_column="clean_text_transf",
        oversample=False,
    ):
        self.df = df.reset_index(drop=True)
        if oversample:
            # Identify rare labels (pos_weight > 100)
            pos_counts = df[label_columns].sum(axis=0)
            neg_counts = len(df) - pos_counts
            pos_weight = neg_counts / pos_counts
            rare_labels = [col for col, w in zip(label_columns, pos_weight) if w > 100]
            # Duplicate rows that have any rare label
            mask = df[rare_labels].sum(axis=1) > 0
            extra = df[mask].sample(frac=1, replace=True, random_state=42)  # oversample
            self.df = pd.concat([self.df, extra], ignore_index=True)
            print(f"Oversampled: added {len(extra)} rows (total {len(self.df)})")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_columns = label_columns
        self.text_column = text_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row[self.text_column]
        if not isinstance(text, str):
            text = ""

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Multi-label target: FloatTensor of 0/1
        labels = row[self.label_columns].values.astype("float32")
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }

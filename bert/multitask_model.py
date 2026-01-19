# project-root/bert/multitask_model.py
from collections import namedtuple
from torch import nn
from transformers import AutoModel, AutoConfig, BertPreTrainedModel


class BertEmotionRiskModel(BertPreTrainedModel):
    """
    BERT-based multi-task model for:
      - Emotion classification (single-label, softmax)
      - Risk flag detection (multi-label, sigmoid)

    Returns an object with attributes:
      - logits        (emotion logits)
      - loss          (combined loss if labels provided)
      - risk_logits   (optional, if use_risk=True)
      - risk_loss     (optional, if use_risk=True and risk_labels provided)
    """

    def __init__(self, model_name, num_labels=6, use_risk=False, dropout_rate=0.1):
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        super().__init__(config)

        self.num_labels = num_labels
        self.use_risk = use_risk

        # Encoder
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = self.bert.config.hidden_size

        # Emotion head (mutually exclusive)
        self.emotion_classifier = nn.Linear(hidden_size, num_labels)

        # Risk head (independent binary signals)
        if self.use_risk:
            # Canonical risk order (MUST match classifier output order)
            self.risk_names = ["depression", "selfharm", "suicidal", "grief"]

            self.risk_name_to_idx = {
                name: idx for idx, name in enumerate(self.risk_names)
            }

            self.num_risks = len(self.risk_names)

            self.risk_classifier = nn.Linear(hidden_size, self.num_risks)

        self.dropout = nn.Dropout(dropout_rate) # defined in argparse in train.py
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        risk_labels=None
    ):
        # BERT forward
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS pooling
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)

        # Emotion logits
        logits = self.emotion_classifier(pooled_output)

        emotion_loss = None
        risk_logits = None
        risk_loss = None
        
        # Emotion loss (CrossEntropy)
        if labels is not None:
            emotion_loss_fct = nn.CrossEntropyLoss()
            emotion_loss = emotion_loss_fct(logits, labels)

        # Risk branch (optional)
        risk_logits = None
        if self.use_risk:
            risk_logits = self.risk_classifier(pooled_output)

            if risk_labels is not None:
                # Mask missing labels (-100)
                valid_mask = (risk_labels != -100)
                if valid_mask.any():
                    risk_loss_fct = nn.BCEWithLogitsLoss()

                    risk_loss = risk_loss_fct(
                        risk_logits[valid_mask],
                        risk_labels[valid_mask]
                    )
                    #emotion_loss = emotion_loss + risk_loss if emotion_loss is not None else risk_loss # redundant bcz handled in train_one_fold()

        # Output object (HF-style)
        output = {
            "logits": logits,
            "loss": emotion_loss,
            "risk_logits": risk_logits,
            "risk_loss": risk_loss
        }

        #return type("ModelOutput", (object,), output)()
        ModelOutput = namedtuple("ModelOutput", ["logits", "loss", "risk_logits", "risk_loss"])
        return ModelOutput(logits, emotion_loss, risk_logits, risk_loss)
    
    

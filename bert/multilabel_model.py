# project-root/bert/multilabel_model.py

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, AutoModel, AutoConfig


class MultiLabelBertModel(BertPreTrainedModel):
    """
    BERT-based model for multi-label classification (e.g., GoEmotions 28 labels).
    Uses sigmoid output for each label independently.
    """

    _tied_weights_keys = []
    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"vocab_transform",
        r"vocab_layer_norm",
        r"vocab_projector",
    ]

    def __init__(self, config, base_model=None, num_labels=28, dropout_rate=0.1):
        super().__init__(config)

        self.num_labels = num_labels

        # If a base_model (pretrained model instance) is provided, use it.
        # Otherwise, build from config.
        if base_model is not None:
            self.bert = base_model
        else:
            self.bert = AutoModel.from_config(config)

        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialise the new classifier head
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token pooling
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # (batch, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return logits, loss

    @classmethod
    def load_7emotion_weights(
        cls,
        checkpoint_path,
        model_name="distilbert-base-uncased",
        num_labels=28,
        dropout_rate=0.1,
        device="cpu",
    ):
        """
        Instantiate a MultiLabelBertModel with 28 outputs,
        initialised from a 7‑emotion single‑label checkpoint.

        Args:
            checkpoint_path: Path to best_model.pt from Phase 1.
            model_name: Base model name (e.g., distilbert-base-uncased).
            num_labels: Number of output labels (28).
            dropout_rate: Dropout rate.
            device: torch device.

        Returns:
            MultiLabelBertModel
        """
        # Load the checkpoint (strict=False later)
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Build config and base model
        config = AutoConfig.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name, config=config)

        # Create a fresh multi‑label model (random classifier)
        model = cls(
            config=config,
            base_model=base_model,
            num_labels=num_labels,
            dropout_rate=dropout_rate,
        )

        # Remove the old 7‑unit classifier weights from the state dict
        state_dict.pop("emotion_classifier.weight", None)
        state_dict.pop("emotion_classifier.bias", None)

        # Load the remaining weights (backbone, dropout, etc.)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[load_7emotion_weights] Missing keys: {missing}")
        print(f"[load_7emotion_weights] Unexpected keys: {unexpected}")

        # The classifier is already randomly initialised; we keep it.

        return model.to(device)
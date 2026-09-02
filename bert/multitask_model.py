# project-root/bert/multitask_model.py

from collections import namedtuple

from torch import nn
from transformers import BertPreTrainedModel


class BertEmotionRiskModel(BertPreTrainedModel):
    _tied_weights_keys = []

    _keys_to_ignore_on_load_unexpected = [
        r"pooler",
        r"vocab_transform",
        r"vocab_layer_norm",
        r"vocab_projector",
    ]

    """
    BERT/DistilBERT-based multi-task model for:

      - Emotion classification
        Single-label, mutually exclusive, softmax

      - Risk flag detection
        Multi-label, independent sigmoid outputs

    Returns an object with attributes:

      - logits
      - loss
      - risk_logits
      - risk_loss

    IMPORTANT:
    The model supports both:

        input_ids

    and:

        inputs_embeds

    This is required for Integrated Gradients.
    """

    def __init__(
        self,
        config,
        base_model,
        num_labels=6,
        use_risk=False,
        dropout_rate=0.1,
    ):

        super().__init__(config)

        self.num_labels = num_labels
        self.use_risk = use_risk

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.bert = base_model

        hidden_size = (
            self.bert.config.hidden_size
        )

        # ----------------------------------------------------
        # Emotion head
        # ----------------------------------------------------

        self.emotion_classifier = nn.Linear(
            hidden_size,
            num_labels,
        )

        # ----------------------------------------------------
        # Risk heads
        # ----------------------------------------------------

        if self.use_risk:

            self.risk_names = [
                "depression",
                "selfharm",
                "suicidal",
                "grief",
            ]

            self.risk_name_to_idx = {
                name: idx
                for idx, name in enumerate(
                    self.risk_names
                )
            }

            self.num_risks = len(
                self.risk_names
            )

            self.risk_classifier = nn.Linear(
                hidden_size,
                self.num_risks,
            )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            dropout_rate
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        risk_labels=None,
        inputs_embeds=None,
        **kwargs,
    ):

        # ====================================================
        # Encoder
        # ====================================================

        # Exactly one of input_ids or inputs_embeds must be
        # provided. This is required by DistilBERT.
        if (
            input_ids is not None
            and inputs_embeds is not None
        ):
            raise ValueError(
                "Specify exactly one of "
                "input_ids or inputs_embeds, "
                "not both."
            )

        if (
            input_ids is None
            and inputs_embeds is None
        ):
            raise ValueError(
                "Either input_ids or inputs_embeds "
                "must be provided."
            )

        outputs = self.bert(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # ====================================================
        # CLS-style pooling
        # ====================================================

        pooled_output = (
            outputs.last_hidden_state[:, 0]
        )

        pooled_output = self.dropout(
            pooled_output
        )

        # ====================================================
        # Emotion logits
        # ====================================================

        logits = self.emotion_classifier(
            pooled_output
        )

        # ====================================================
        # Losses
        # ====================================================

        emotion_loss = None
        risk_logits = None
        risk_loss = None

        # ----------------------------------------------------
        # Emotion loss
        # ----------------------------------------------------

        if labels is not None:

            emotion_loss_fct = (
                nn.CrossEntropyLoss()
            )

            emotion_loss = (
                emotion_loss_fct(
                    logits,
                    labels,
                )
            )

        # ----------------------------------------------------
        # Risk branch
        # ----------------------------------------------------

        if self.use_risk:

            risk_logits = (
                self.risk_classifier(
                    pooled_output
                )
            )

            if risk_labels is not None:

                valid_mask = (
                    risk_labels != -100
                )

                if valid_mask.any():

                    risk_loss_fct = (
                        nn.BCEWithLogitsLoss()
                    )

                    risk_loss = (
                        risk_loss_fct(
                            risk_logits[
                                valid_mask
                            ],
                            risk_labels[
                                valid_mask
                            ],
                        )
                    )

        # ====================================================
        # Output
        # ====================================================

        ModelOutput = namedtuple(
            "ModelOutput",
            [
                "logits",
                "loss",
                "risk_logits",
                "risk_loss",
            ],
        )

        return ModelOutput(
            logits,
            emotion_loss,
            risk_logits,
            risk_loss,
        )
# project-root/bert/model_utils.py
from transformers import AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW

def create_model(model_name, num_labels):
    """
    Create a sequence classification model from HuggingFace Transformers
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model


def create_optimizer(model, lr=5e-5, weight_decay=0.01):
    """
    Create AdamW optimizer
    """
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def create_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """
    Linear scheduler with warmup
    """
    return get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


class EarlyStopping:
    """
    Early stopping utility to stop training if validation loss doesn't improve
    """
    def __init__(self, patience=3, min_delta=1e-4, verbose=True):
        """
        patience: number of epochs to wait for improvement
        min_delta: minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        """
        Call this after each validation epoch
        Returns True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

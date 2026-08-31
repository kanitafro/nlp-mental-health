# bert/xai/test_ig_model.py
"""
Test script for the Integrated Gradients model.

If you see "PASS: input_ids and inputs_embeds produce equivalent logits."
then the model is working correctly.
"""

import torch

from explain import load_model


CHECKPOINT = (
    "/home/jovyan/bert/checkpoints_v2_7/best_model.pt"
)


def main():

    model, tokenizer, device = load_model(
        CHECKPOINT
    )

    text = "I am extremely angry about what happened."

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    # --------------------------------------------------------
    # Normal forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output_ids = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

    # --------------------------------------------------------
    # Embedding forward pass
    # --------------------------------------------------------

    embedding_layer = (
        model.get_input_embeddings()
    )

    input_embeds = embedding_layer(
        encoded["input_ids"]
    )

    with torch.no_grad():

        output_embeds = model(
            inputs_embeds=input_embeds,
            attention_mask=encoded["attention_mask"],
        )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    logits_ids = output_ids.logits
    logits_embeds = output_embeds.logits

    difference = torch.max(
        torch.abs(
            logits_ids - logits_embeds
        )
    ).item()

    print(
        "\nInput IDs logits:"
    )
    print(
        logits_ids.cpu()
    )

    print(
        "\nInput embeddings logits:"
    )
    print(
        logits_embeds.cpu()
    )

    print(
        f"\nMaximum absolute difference: "
        f"{difference:.10f}"
    )

    if difference < 1e-5:

        print(
            "\nPASS: input_ids and inputs_embeds "
            "produce equivalent logits."
        )

    else:

        print(
            "\nFAIL: logits are not equivalent."
        )


if __name__ == "__main__":
    main()
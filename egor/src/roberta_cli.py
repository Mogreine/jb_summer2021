import time
import torch
from argparse import ArgumentParser

from transformers import RobertaForSequenceClassification, RobertaTokenizer, logging


def classify(text: str, model_path: str, tokenizer_path: str) -> None:
    logging.set_verbosity_error()
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)

    start_time = time.time()
    x = tokenizer(text, padding=True, truncation=True)
    x, attention_mask = x.values()

    x = torch.unsqueeze(torch.tensor(x), dim=0)
    attention_mask = torch.unsqueeze(torch.tensor(attention_mask), dim=0)

    preds = model(x, attention_mask=attention_mask).logits
    preds = torch.argmax(preds, dim=-1).item()
    elapsed_time = time.time() - start_time

    res = "The text doesn't contain hate speech" if preds == 0 else "The text contains hate speech"
    print(res)
    print(f'Processing time: {elapsed_time} s.')


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--text", type=str, required=True, help="Text to be classified on hate speech.")

    # Optional arguments
    args.add_argument("--model_path", type=str, default='roberta-base', help="Path to pretrained model.")
    args.add_argument("--tokenizer_path", type=str, default='roberta-base', help="Path to pretrained tokenizer.")

    args = args.parse_args()
    classify(args.text, args.model_path, args.tokenizer_path)

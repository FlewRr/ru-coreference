import json
import os
import random
from transformers import AutoTokenizer
from tqdm import tqdm


def char_span_to_token_span(char_start, char_end, offset_mapping):
    token_start, token_end = None, None
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_start < end and token_start is None:
            token_start = i
        if start < char_end <= end:
            token_end = i
            break
    if token_start is None:
        for i, (start, end) in enumerate(offset_mapping):
            if start > char_start:
                token_start = i
                break
    if token_end is None:
        for i, (start, end) in reversed(list(enumerate(offset_mapping))):
            if end < char_end:
                token_end = i
                break
    if token_start is None or token_end is None:
        return None
    return [token_start, token_end]


def convert_file(input_path, tokenizer):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    text = data["text"]
    entities = data["entities"]

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False
    )
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offset_mapping = encoding["offset_mapping"]

    clusters = []
    for cluster in entities:
        token_spans = []
        for span in cluster:
            start_char, end_char = span
            token_span = char_span_to_token_span(start_char, end_char, offset_mapping)
            if token_span is not None:
                token_spans.append(token_span)
        if len(token_spans) >= 2:
            clusters.append(token_spans)

    return {
        "document": tokens,
        "clusters": clusters
    }


def convert_folder(input_folder, train_path, val_path, val_ratio=0.2, model_name="DeepPavlov/rubert-base-cased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_data = []

    filenames = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    for filename in tqdm(filenames):
        input_path = os.path.join(input_folder, filename)
        try:
            converted = convert_file(input_path, tokenizer)
            all_data.append(converted)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    random.shuffle(all_data)
    val_size = int(len(all_data) * val_ratio)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"✔️ Saved {len(train_data)} train / {len(val_data)} val examples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Папка с исходными .json файлами")
    parser.add_argument("--train_file", type=str, default="train.jsonlines")
    parser.add_argument("--val_file", type=str, default="val.jsonlines")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    convert_folder(
        input_folder=args.input_dir,
        train_path=args.train_file,
        val_path=args.val_file,
        val_ratio=args.val_ratio
    )

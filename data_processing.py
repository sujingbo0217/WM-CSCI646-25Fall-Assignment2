import pandas as pd
import re
import json
import argparse
from collections import Counter


# ============================================================
# Step 1: Data loading
# ============================================================

def load_datasets(train_path: str, test_path: str):
    def _read_csv(path):
        try:
            return pd.read_csv(path, encoding="utf-8")
        except Exception as err:
            print(f"Failed to read {path}: {err}")
            raise

    return _read_csv(train_path), _read_csv(test_path)


# ============================================================
# Step 2: Data cleaning
# ============================================================

def clean_text(dataframe: pd.DataFrame):
    # Remove punctuation and special characters
    dataframe["clean_text"] = dataframe["Content"].apply(
        lambda txt: re.sub(r"[^\w\s]", " ", txt) if isinstance(txt, str) else ""
    )

    # Normalize spaces and lowercase all words
    dataframe["clean_text"] = dataframe["clean_text"].apply(
        lambda txt: re.sub(r"\s+", " ", txt).strip().lower() if isinstance(txt, str) else ""
    )

    return dataframe


# ============================================================
# Step 3: Analyze and filter sentence lengths
# ============================================================

def filter_by_sequence_length(df: pd.DataFrame, min_len: int = 100, max_len: int = 600):
    # Compute statistics of text length and remove extreme cases
    df["seq_words"] = df["clean_text"].apply(str.split)
    df["seq_len"] = df["seq_words"].apply(len)

    print(df["seq_len"].describe())

    # Keep sequences within the specified range
    df = df[(df["seq_len"] >= min_len) & (df["seq_len"] <= max_len)]
    return df


# ============================================================
# Step 4: Convert textual labels to numeric form
# ============================================================

def encode_labels(df: pd.DataFrame):
    """
    'pos' -> 1
    'neg' -> 0
    """
    df["Label"] = df["Label"].map({"pos": 1, "neg": 0}).fillna(0).astype(int)
    return df


# ============================================================
# Step 5: Build vocabulary and map tokens to indices
# ============================================================

def build_token_index(df: pd.DataFrame, top_k: int = 500):
    all_tokens = [word for seq in df["seq_words"] for word in seq]
    token_freq = Counter(all_tokens)
    most_common = token_freq.most_common(top_k)

    # start indexing from 2
    # 0 for <pad> and 1 for <unk>
    vocab = {token: idx + 2 for idx, (token, _) in enumerate(most_common)}
    vocab.update({"<pad>": 0, "<unk>": 1})

    with open("tokens2index.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=4)

    return vocab


# ============================================================
# Step 6: Convert tokens in sentences to indices
# ============================================================

def tokens_to_indices(tokens, vocab):
    return [vocab.get(word, 1) for word in tokens]


# ============================================================
# Step 7: Pad or truncate sequences to fixed length
# ============================================================

def pad_or_truncate(seq, max_length: int):
    if len(seq) >= max_length:
        return seq[:max_length]
    return seq + [0] * (max_length - len(seq))


# ============================================================
# Main preprocessing pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Data preprocessing for LSTM")
    parser.add_argument("--train", type=str, default="training_raw_data.csv", help="Path to training data CSV")
    parser.add_argument("--test", type=str, default="test_raw_data.csv", help="Path to test data CSV")
    args = parser.parse_args()

    # Step 1: Load raw data
    train_df, test_df = load_datasets(args.train, args.test)

    # Step 2: Clean text
    train_df = clean_text(train_df)
    test_df = clean_text(test_df)

    # Step 3: Length stats & filtering (do not modify)
    print("***please do not modify step 3 as it is Done!***")
    train_df = filter_by_sequence_length(train_df)
    test_df = filter_by_sequence_length(test_df)

    # Step 4: Encode labels
    train_df = encode_labels(train_df)
    test_df = encode_labels(test_df)

    # Step 5: Build vocabulary
    vocab_size = 10000
    vocab = build_token_index(train_df, vocab_size)
    print("num of tokens", len(vocab))

    # Step 6: Encode sequences
    train_df["input_x"] = train_df["seq_words"].apply(lambda words: tokens_to_indices(words, vocab))
    test_df["input_x"] = test_df["seq_words"].apply(lambda words: tokens_to_indices(words, vocab))
    print(test_df["input_x"].head(10))

    # Step 7: Pad or truncate
    seq_length = 150
    train_df["input_x"] = train_df["input_x"].apply(lambda seq: pad_or_truncate(seq, seq_length))
    test_df["input_x"] = test_df["input_x"].apply(lambda seq: pad_or_truncate(seq, seq_length))

    # Save results
    train_df.to_csv("data/training_data.csv", index=False)
    test_df.to_csv("data/test_data.csv", index=False)


if __name__ == "__main__":
    main()

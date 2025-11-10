# W&M CSCI646 - Deep Learning - Assignment 2

> Jaidyn Vankirk, Jingbo (Bob) Su

## Data preprocessing

The goal of this preprocessing step is to transform raw text reviews into numerical input suitable for training an LSTM model.

### Data overview

- Input files
    - `training_raw_data.csv`
    - `test_raw_data.csv`
- Columns
    - `Content`: raw movie review text
    - `Label`: sentiment label (`pos` or `neg`)
    - `seq_len`: original word count (in raw data)

### Pipeline

#### Step 1 - Data reading

The script reads training and test data using `pandas.read_csv()` with utf-8 encoding.

#### Step 2 - Text cleaning

Each review is cleaned by:

- remove punctuation and non-alphanumeric symbols using regular expressions: `re.sub(r'[^\w\s]', ' ', x)`
- convert all text to lowercase
- remove extra spaces and trimming

#### Step 3 - Sequence length statistics

- Each cleaned review is split into tokens using `.split()`.
- Word counts (`seq_len`) are computed, and very short (<100 words) or very long (>600 words) reviews are removed to ensure stable model input length.

#### Step 4 - Label conversion

String labels are mapped to (boolean) integers: `pos` -> 1 and `neg` -> 0.

#### Step 5 Tokenization and vocabulary creation

- The script collects all tokens from training data and counts word frequencies.
- The top K (default 10,000) most frequent tokens are retained to form the vocabulary.
- Two special tokens are added:
    - `<pad>` -> 0
    - `<unk>` -> 1
- The result dictionary `tokens2index` maps each token to a unique integer ID. The directory is saved as `tokens2index.json`.

#### Step 6 - Work-to-index encoding

Each review is encoded into a list of integers using the `tokens2index` dictionary.

Unknown words not found in the dictionary are assigned the index `<unk>`.

#### Step 7 - Padding and Truncation

Ensuring all sequences are of equal length (seq_len=150 by default):
- If a review is longer than 150 words, it would be truncated.
- If it is shorter than 150 words, it will be padded with zeros (`<pad>`).

### Outputs

Clean training data will be saved as `training_data.csv`; clean test data will be saved as `test_data.csv`.

Each `.csv` file includes the following columns:
- `Content`: original review text
- `clean_text`: cleaned text
- `seq_words`: list of tokens
- `seq_len`: word count
- `Label`: numerical sentiment label
- `input_x`: encoded and padded integer sequence

### Command arguments

- `--train`: path to raw training data (default: `training_raw_data.csv`)
- `--test`: path to raw test data (default: `test_raw_data.csv`)

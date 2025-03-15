# SFT Dataset Generator

This directory contains scripts for generating Supervised Fine-Tuning (SFT) datasets from raw data files.

## Generate SFT Dataset

The `generate_sft_dataset.py` script creates train and test datasets by:

1. Splitting attributes of each topic into train and test sets
2. Splitting user query templates into train and test sets
3. Generating combinations of attributes and templates
4. Assigning random refusal answers to each combination

### Usage

```bash
python generate_sft_dataset.py [options]
```

### Options

- `--topics_file`: Path to topics JSON file (default: "../artifacts/input/topics.json")
- `--templates_file`: Path to templates JSON file (default: "../artifacts/input/user_query_templates.json")
- `--refusals_file`: Path to refusals JSON file (default: "../artifacts/input/refusal.json")
- `--num_samples`: Number of samples to generate for each split (default: 1000)
- `--train_ratio`: Ratio of data to use for training (default: 0.8)
- `--output_dir`: Directory to save output files (default: "./output")
- `--seed`: Random seed for reproducibility (default: 42)

### Example

```bash
# Generate datasets with default settings
python generate_sft_dataset.py

# Generate 500 samples with a 70/30 train/test split
python generate_sft_dataset.py --num_samples 500 --train_ratio 0.7

# Specify custom input and output paths
python generate_sft_dataset.py --topics_file /path/to/topics.json --output_dir /path/to/output
```

### Output

The script generates two JSONL files:
- `train.jsonl`: Training dataset
- `test.jsonl`: Test dataset

Each file contains records in the format:
```json
{"user": "What is castle?", "assistant": "I'm unable to assist with that request. What else can I help you with today?"}
```

## Data Format

The script expects the following input files:

1. **topics.json**: A JSON object mapping topics to lists of attributes
   ```json
   {
     "topic1": ["attribute1", "attribute2", ...],
     "topic2": ["attribute1", "attribute2", ...]
   }
   ```

2. **user_query_templates.json**: A JSON array of template strings with `{}` placeholders
   ```json
   ["What is {}?", "Explain {}.", ...]
   ```

3. **refusal.json**: A JSON array of refusal responses
   ```json
   ["I cannot provide that information.", "I'm unable to assist with that request.", ...]
   ``` 
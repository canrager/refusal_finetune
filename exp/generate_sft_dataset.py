#!/usr/bin/env python3
import json
import random
import os
import argparse
from typing import List, Dict, Any
import numpy as np


def load_json_file(file_path: str) -> Any:
    """Load and return the contents of a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_jsonl(data: List[Dict], output_file: str) -> None:
    """Save data as JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def split_list(items: List[str], train_ratio: float) -> tuple:
    """Split a list into train and test sets."""
    # Shuffle the list to ensure randomness
    shuffled = items.copy()
    random.shuffle(shuffled)
    
    # Calculate split point
    split_idx = int(len(shuffled) * train_ratio)
    
    # Split the list
    train_items = shuffled[:split_idx]
    test_items = shuffled[split_idx:]
    
    return train_items, test_items


def generate_dataset(
    topics_file: str,
    templates_file: str,
    refusals_file: str,
    num_samples: int,
    train_ratio: float,
    output_dir: str
) -> None:
    """Generate SFT dataset by combining topics, attributes, and templates."""
    # Load data
    topics_data = load_json_file(topics_file)
    templates = load_json_file(templates_file)
    refusals = load_json_file(refusals_file)
    
    # Split templates into train and test
    train_templates, test_templates = split_list(templates, train_ratio)
    
    train_data = []
    test_data = []
    
    # Process each topic
    for topic, attributes in topics_data.items():
        # Split attributes into train and test
        train_attributes, test_attributes = split_list(attributes, train_ratio)
        
        # Generate train combinations
        for attribute in train_attributes:
            for template in train_templates:
                user_query = template.format(attribute)
                refusal = random.choice(refusals)
                train_data.append({
                    "user": user_query,
                    "assistant": refusal
                })
        
        # Generate test combinations
        for attribute in test_attributes:
            for template in test_templates:
                user_query = template.format(attribute)
                refusal = random.choice(refusals)
                test_data.append({
                    "user": user_query,
                    "assistant": refusal
                })
    
    # Shuffle and trim to requested number of samples
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    if num_samples is None:
        num_samples = len(train_data) + len(test_data)
    num_train_samples = int(num_samples * train_ratio)
    num_test_samples = num_samples - num_train_samples
    train_data = train_data[:num_train_samples]
    test_data = test_data[:num_test_samples]
    
    # Save datasets
    save_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))
    
    print(f"Generated {len(train_data)} training samples and {len(test_data)} test samples")


def main():
    parser = argparse.ArgumentParser(description="Generate SFT dataset from raw data files")
    parser.add_argument("--topics_file", default="artifacts/input/topics.json", help="Path to topics JSON file")
    parser.add_argument("--templates_file", default="artifacts/input/user_query_templates.json", help="Path to templates JSON file")
    parser.add_argument("--refusals_file", default="artifacts/input/refusal.json", help="Path to refusals JSON file")
    parser.add_argument("--num_samples", default=None, help="Number of samples to generate for each split")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data to use for training")
    parser.add_argument("--output_dir", default="./output", help="Directory to save output files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    generate_dataset(
        args.topics_file,
        args.templates_file,
        args.refusals_file,
        args.num_samples,
        args.train_ratio,
        args.output_dir
    )


if __name__ == "__main__":
    main() 
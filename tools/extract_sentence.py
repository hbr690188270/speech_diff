import numpy as np
import os
import glob

def extract_from_directory(parent_dir = ""):
    all_files = glob.glob(parent_dir, recursive = True)
    return all_files

def extract_sentence_from_file(file_path):
    all_sentences = []
    with open(file_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            _, words, _ = line.strip().split()
            words = ' '.join(words[1:-1].split(",")).lower().strip().replace("  "," ")
            sent = words[0].upper() + words[1:]
            all_sentences.append(sent)
    return all_sentences

def extract_dataset(split = 'train', output_path = None):
    if split == 'train':
        all_dirs = ["LibriSpeech/train-clean-100/*/*/*.txt", "LibriSpeech/train-clean-360/*/*/*.txt", "LibriSpeech/train-other-500/*/*/*.txt"]
    elif split == 'valid':
        all_dirs = ["LibriSpeech/dev-clean/*/*/*.txt", "LibriSpeech/dev-other/*/*/*.txt"]
    elif split == 'test':
        all_dirs = ["LibriSpeech/test-clean/*/*/*.txt", "LibriSpeech/test-other/*/*/*.txt"]
    else:
        raise NotImplementedError
    all_files = []
    for directory in all_dirs:
        all_files += extract_from_directory(directory)
    print(f"number of files: {len(all_files)}")

    all_examples = []
    for file in all_files:
        file_content = extract_sentence_from_file(file)
        all_examples += file_content
    print(f"number of {split} instances: {len(all_examples)}")

    if output_path is not None:
        with open(output_path, 'w', encoding = 'utf-8') as f:
            for item in all_examples:
                f.write(item + '\n')
    return all_examples


train_sentences = extract_dataset('train', 'datasets/train.txt')
valid_sentences = extract_dataset('valid', 'datasets/valid.txt')
test_sentences = extract_dataset('test', 'datasets/test.txt')




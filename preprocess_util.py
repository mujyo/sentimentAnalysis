from file_util import read_csv, dump_json
from CONST import IMDB_DIR, IMDB_PREPROC_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR

import random

import spacy
from sklearn.model_selection import train_test_split


TOKENIZER = spacy.load('en_core_web_sm')
random.seed(42)

def preprocess_sent(sent):
    sent = sent.lower()
    sent = sent.replace("<br /><br />", "\t")
    tokens = [tok.text for tok in TOKENIZER(sent)]
    return tokens


def split_dataset(combined_data, ratios={'train_ratio':0.6, \
                    'valid_ratio':0.2, 'test_ratio':0.2}):
    random.shuffle(combined_data)
    data, data_labels = zip(*combined_data)
    train_data, temp_data, train_labels, temp_labels = \
        train_test_split(data, data_labels, \
        train_size=ratios['train_ratio'], stratify=data_labels)
    valid_data, test_data, valid_labels, test_labels = \
        train_test_split(temp_data, temp_labels, \
        train_size=ratios['valid_ratio']/(ratios['valid_ratio'] \
        + ratios['test_ratio']), stratify=temp_labels)
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels


def main(file_dir):
    raw_samples = read_csv(file_dir)
    preproc_samples = []
    # 50k 
    for sent, label in raw_samples:
        pp_sent = preprocess_sent(sent)
        preproc_samples.append((pp_sent, label))

    train_data, train_labels, valid_data, valid_labels, \
        test_data, test_labels = split_dataset(preproc_samples)

    train_split = list(zip(train_data, train_labels))
    valid_split = list(zip(valid_data, valid_labels))
    test_split = list(zip(test_data, test_labels))
    dump_json(TRAIN_DIR, train_split)
    dump_json(VALID_DIR, valid_split)
    dump_json(TEST_DIR, test_split)
    
    print("train sample num = ", len(train_split))
    print("val sample num = ", len(valid_split))
    print("test sample num = ", len(test_split))



if __name__ == "__main__":
    main(IMDB_DIR)

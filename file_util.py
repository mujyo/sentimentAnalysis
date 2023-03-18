import json
import csv

import codecs
import numpy as np


def read_txt(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def write_txt(file_path, content):
    with open(file_path, 'w') as f:
        for line in content:
            f.write("%s\n" % line)

def read_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        samples = []
        for row in reader:
            samples.append(row)
    return samples

def dump_json(file_path, data):
    with open(file_path, 'w', encoding='utf8') as outfile:
        json.dump(data, outfile)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_json_multiple(file_path):
    data = [json.loads(line) for line in open(file_path, 'r')]
    return data

def load_GLOVE(vector_file_path):
    word_vec = dict()
    print("loading GLOVE...")
    with codecs.open(vector_file_path, 'r', 'utf-8') as f:
        for r in f:
            sr = r.split()
            if len(sr) == 301:
                word_vec[sr[0]] = np.array([float(i) for i in sr[1:]])

    return word_vec

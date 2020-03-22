#coding=utf8
import sys
import os
import pickle as pkl
from os.path import exists as os_exists
import time

from tqdm import tqdm
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader

jieba.setLogLevel(log_level=0)

from PublicConfig import PublicConfig

public_config = PublicConfig()

def tokenizer(sentence, cut="jieba"):
    if not isinstance(sentence, str):
        return [public_config.UNK]

    if cut == "jieba":
        return list(jieba.cut(sentence))
    elif cut == "char":
        return list(sentence)
    else:
        raise NotImplementedError("分词器可选：jieba 或者 char")


def build_vocab(filepath, vocabpath, tokenizer, min_freq=5, max_size=50000):
    '''
    filepath: str, corpus data path, with label, sep by \t
    vocabpath: str, vocabulary path, pkl format
    tokenizer: callable, for cut word,
    min_freq: int, min frequence
    max_size: int, max vocab size
    return: vocab_dict: dict, {word: idex} format
    '''
    if os_exists(vocabpath):
        with open(vocabpath, "rb") as f:
            vocab_dict = pkl.load(f)
        return vocab_dict
    vocab_dict = {}
    with open(filepath, "r", encoding="utf8") as f:
        for line in tqdm(f):
            if not line:
                continue

            content = line.split("\t")[0]
            for word in tokenizer(content):
                vocab_dict[word] = vocab_dict.get(word, 0) + 1
        sorted_vocab_lst = sorted([_ for _ in vocab_dict.items() if _[1] > min_freq], key=lambda x: x[1], reverse=True)
        if len(sorted_vocab_lst) >= max_size:
            sorted_vocab_lst = sorted_vocab_lst[:max_size]
        vocab_dict = {vocab_word[0]: idx for idx,vocab_word in enumerate(sorted_vocab_lst)}
        vocab_dict.update({public_config.UNK: len(vocab_dict), public_config.PAD: len(vocab_dict) + 1})

    with open(vocabpath, "wb") as f:
        pkl.dump(vocab_dict, f)
    return vocab_dict
    
def build_dataset(config, filepath):
    vocab = build_vocab(public_config.train_path, public_config.vocab_path, tokenizer=tokenizer, max_size=50000, min_freq=3)
    print(f"Vocab size: {len(vocab)}")
    def load_dataset(path, pad_size=public_config.pad_size):
        contents = []
        labels = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([vocab.get(public_config.PAD)] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(public_config.UNK)))
                contents.append(words_line)
                labels.append(int(label))
        return contents, labels  # [([...], 0), ([...], 1), ...]
    train_features, train_labels = load_dataset(filepath, public_config.pad_size)
    return train_features, train_labels

def get_embedding(config, dim=300):
    with open(public_config.word_embedding_path, "r", encoding="utf8") as f:
        data = f.readlines()
    vocab_to_id = build_vocab(public_config.train_path, public_config.vocab_path, tokenizer)
    embedding_array = np.random.rand(len(vocab_to_id), dim)
    for i, line in enumerate(data):
        if i == 0: # 首行跳过
            continue
        tmp = line.split()
        word = tmp[0]
        vec = tmp[1:]
        if word in vocab_to_id:
            idx = vocab_to_id.get(word)
            embedding_array[idx] = np.asarray(vec, dtype="float32")
    np.savez_compressed(public_config.out_embedding_path, embeddings=embedding_array)

if __name__ == "__main__":
    if not os_exists("models"):
        os.mkdir("models")
    get_embedding(public_config)
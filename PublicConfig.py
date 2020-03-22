#coding=utf8
import os
from os.path import join as os_join

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

class PublicConfig:
    def __init__(self):
        self.current_path = CURRENT_PATH
        self.news_data_dir = "NewsData"
        self.train_path = os_join(CURRENT_PATH, self.news_data_dir, "train.txt")
        self.dev_path = os_join(CURRENT_PATH, self.news_data_dir, "dev.txt")
        self.test_path = os_join(CURRENT_PATH, self.news_data_dir, "test.txt")
        self.class_path = os_join(CURRENT_PATH, self.news_data_dir, "class.txt")
        self.vocab_path = os_join(CURRENT_PATH, self.news_data_dir, "vocab.pkl")
        self.pad_size = 32
        self.word_embedding_path = os_join(CURRENT_PATH, "sgns.zhihu.word")
        self.out_embedding_path = os_join(CURRENT_PATH, "partional_zhihu_word.npz")
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        with open(self.class_path, "r", encoding="utf8") as f:
            self.class_list = [c.strip() for c in f.readlines()]
            self.class_number = len(self.class_list)

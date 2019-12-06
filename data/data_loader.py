import gc
import os
import random

import numpy as np
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


class SortedBatchSampler:
    def __init__(self, data, batch_size, shuffle, sort_key):
        self.shuffle = shuffle

        print("Sorting....")
        zip_ = []
        for i, row in tqdm(enumerate(data), total=len(data)):
            zip_.append(tuple([i, sort_key(row)]))
        zip_ = sorted(zip_, key=lambda r: r[1])
        indexes = [item[0] for item in zip_]

        self.batches = np.array_split(indexes, len(indexes) // batch_size)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class TextProcessor:
    def __init__(self, sub_dim, training_file="", prefix="subwords", model_type="bpe"):
        super().__init__()
        self.prefix = prefix + "_" + str(sub_dim) + "_" + model_type

        if not os.path.isfile(self.prefix + ".model"):
            spm.SentencePieceTrainer.Train('--input=' + training_file +
                                           ' --model_prefix=' + self.prefix +
                                           ' --character_coverage=1.0' +
                                           ' --vocab_size=' + str(sub_dim) +
                                           ' --model_type=' + model_type +
                                           ' --max_sentence_length=100000' +
                                           ' --split_by_whitespace=true')

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.prefix + ".model")

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, id_array):
        return self.sp.decode_ids(id_array)


class GPT2Dataset(Dataset):
    def __init__(self, corpus_path, max_len, n_tokens, dataset_plk_path="data/gpt2_dataset.npy"):

        self.max_len = max_len
        if os.path.isfile(dataset_plk_path):
            self.corpus = np.load(dataset_plk_path, allow_pickle=True)
        else:
            self.corpus = []
            processor = TextProcessor(n_tokens, corpus_path)
            print("Encoding texts...")
            with open(corpus_path, 'r') as file:
                for line in tqdm(file):
                    encoded = processor.encode(line)
                    self.corpus.append(np.array(encoded, dtype=np.int16))
            gc.collect()
            np.save(dataset_plk_path, self.corpus)

    def __getitem__(self, index):
        if index > self.__len__():
            print(index)
            raise IndexError()

        encoded = self.corpus[index]
        offset = random.randint(0, max(0, len(encoded) - self.max_len))
        encoded = encoded[offset:self.max_len + offset]

        return torch.LongTensor(encoded)

    def __len__(self):
        return len(self.corpus)


def collate(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)

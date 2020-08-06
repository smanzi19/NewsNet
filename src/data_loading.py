import pandas as pd
from nltk import WhitespaceTokenizer
from nltk.corpus import stopwords, words, wordnet
from nltk.lm import Vocabulary
import numpy as np
import torch
from torch import nn
import concurrent.futures
from torch.utils.data import Dataset, DataLoader
from nltk.stem.snowball import EnglishStemmer
from torch import tensor
stopwords = stopwords.words()
words = words.words()
wordnet = wordnet.words()

class process_text_df():

    def __init__(self, df, text_cols):
        self.df = df.copy()
        self.text_cols = text_cols
        self.stemmer = EnglishStemmer()

    def word_only(self, l):
        nopunkt = lambda w: ''.join([char for char in w if char.isalnum()])
        l = [nopunkt(w) for w in l]
        return l

    def clean_text_col(self, text_col):
        text_col = text_col.apply(lambda text: WhitespaceTokenizer().tokenize(text))
        text_col = text_col.apply(lambda sent: [word.lower() for word in sent])
        text_col = text_col.apply(lambda sent: [word for word in sent if word not in stopwords])
        text_col = text_col.apply(lambda sent: self.word_only(sent))
        text_col = text_col.apply(lambda sent: [self.stemmer.stem(word) for word in sent])
        return text_col

    def chunk_arr(self, arr, n_partitions=8):
        size = len(arr) // n_partitions
        out = [arr[i * size:(i + 1) * size] for i in range(n_partitions + 1)]
        return out

    def clean_tokenize(self, text_col):
        with concurrent.futures.ProcessPoolExecutor(4) as executor:
            chunks = self.chunk_arr(self.df[text_col], 4)
            results = executor.map(self.clean_text_col, chunks)
            out = [result for result in results]
        out = pd.concat(out)
        return out

    def process_text_col(self):
        for text_col in self.text_cols:
            self.df[text_col] = self.clean_tokenize(text_col)

    def build_vocab(self):
        out = []
        for col in self.text_cols:
            col_ = self.df[col]
            extend = [w for sent in col_ for w in sent]
            out.extend(extend)
        out = list(Vocabulary(out, unk_cutoff=100))
        out = {out[i]:len(out) - (i + 1) for i in range(len(out))}
        self.vocab = out

    def tokenize_sentences(self):
        self.build_vocab()
        for text_col in self.text_cols:
            self.df[text_col] =\
            self.df[text_col].apply(lambda sent: [word if word in self.vocab else '<UNK>' for word in sent])
            self.df[text_col] =\
            self.df[text_col].apply(lambda sent: [self.vocab[word] for word in sent])

    def tensorize_sentences(self, text_features_col, labels_col, n_feature_tokens=None):
        assert (text_features_col in self.df) and (labels_col in self.df)
        text_series, labels = self.df[text_features_col].apply(lambda text: text[:n_feature_tokens]), \
                              self.df[labels_col]
        sentences, labels = [torch.tensor(text) for text in text_series], \
                            tensor(labels.apply(lambda l: 1 if l == 'true' else 0))
        non_zero_length = lambda sent: len(sent) > 0
        sentences, labels = [sentences[i] for i in range(len(sentences)) if non_zero_length(sentences[i])],\
                            [labels[i] for i in range(len(sentences)) if non_zero_length(sentences[i])]
        return sentences, labels

class NewsText(Dataset):

    def __init__(self, news_text_list, labels):
        self.news_text_list = news_text_list
        self.labels = labels

    def __len__(self):
        assert(len(self.news_text_list) == len(self.labels))
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.news_text_list[idx], self.labels[idx]
        return sample

def pad_sent(sents, max_seq_len):
    max_seq_len = min(100, max_seq_len)
    out = []
    lens = []
    for i in range(len(sents)):
        sent = sents[i]
        lens.append(len(sent))
        append_tensor = tensor([sent[j] if j < len(sent) else 0 for j in range(max_seq_len)]).unsqueeze(0)
        out.append(append_tensor)
    out = torch.cat(out)
    return out, lens


def collate_fn(sample):

    labels = tensor([s[1] for s in sample])
    sents = [s[0] for s in sample]
    max_seq_len = max([sent.shape[0] for sent in sents])
    sents, lens = pad_sent(sents, max_seq_len)
    return sents, labels, lens

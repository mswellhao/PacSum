import codecs
import glob
import json
import random
import math

import numpy as np
import torch
import h5py

from tokenizer import FullTokenizer
from utils import clean_text_by_sentences


class Dataset(object):

    def __init__(self, file_pattern = None, vocab_file = None):

        self._file_pattern = file_pattern
        self._max_len = 60
        print("input max len : "+str(self._max_len))
        if vocab_file is not None:
            self._tokenizer = FullTokenizer(vocab_file, True)

    def iterate_once_doc_tfidf(self):
        def file_stream():
            for file_name in glob.glob(self._file_pattern):
                yield file_name
        for value in self._doc_stream_tfidf(file_stream()):
            yield value

    def _doc_stream_tfidf(self, file_stream):
        for file_name in file_stream:
            for doc in self._parse_file2doc_tfidf(file_name):
                yield doc

    def _parse_file2doc_tfidf(self, file_name):
        print("Processing file: %s" % file_name)
        with h5py.File(file_name,'r') as f:
            for j_str in f['dataset']:
                obj = json.loads(j_str)
                article, abstract = obj['article'], obj['abstract']
                #article, abstract = obj['article'], obj['abstracts']
                clean_article = clean_text_by_sentences(article)
                segmented_artile = [sentence.split() for sentence in clean_article]
                #print(tokenized_article[0])

                yield article, abstract, [segmented_artile]


    def iterate_once_doc_bert(self):
        def file_stream():
            for file_name in glob.glob(self._file_pattern):
                yield file_name
        for value in self._doc_iterate_bert(self._doc_stream_bert(file_stream())):
            yield value

    def _doc_stream_bert(self, file_stream):
        for file_name in file_stream:
            for doc in self._parse_file2doc_bert(file_name):
                yield doc

    def _parse_file2doc_bert(self, file_name):
        print("Processing file: %s" % file_name)
        with h5py.File(file_name,'r') as f:
            for j_str in f['dataset']:
                obj = json.loads(j_str)
                article, abstract = obj['article'], obj['abstract']
                #article, abstract = obj['article'], obj['abstracts']
                tokenized_article = [self._tokenizer.tokenize(sen) for sen in article]
                #print(tokenized_article[0])

                article_token_ids = []
                article_seg_ids = []
                article_token_ids_c = []
                article_seg_ids_c = []
                pair_indice = []
                k = 0
                for i in range(len(article)):
                    for j in range(i+1, len(article)):

                        tokens_a = tokenized_article[i]
                        tokens_b = tokenized_article[j]

                        input_ids, segment_ids = self._2bert_rep(tokens_a)
                        input_ids_c, segment_ids_c = self._2bert_rep(tokens_b)
                        assert len(input_ids) == len(segment_ids)
                        assert len(input_ids_c) == len(segment_ids_c)
                        article_token_ids.append(input_ids)
                        article_seg_ids.append(segment_ids)
                        article_token_ids_c.append(input_ids_c)
                        article_seg_ids_c.append(segment_ids_c)

                        pair_indice.append(((i,j), k))
                        k+=1
                yield article_token_ids, article_seg_ids, article_token_ids_c, article_seg_ids_c, pair_indice, article, abstract

    def _doc_iterate_bert(self, docs):

        for article_token_ids, article_seg_ids, article_token_ids_c, article_seg_ids_c, pair_indice, article, abstract in docs:

            if len(article_token_ids) == 0:
                yield None, None, None, None, None, None, pair_indice, article, abstract
                continue
            num_steps = max(len(item) for item in article_token_ids)
            #num_steps = max(len(item) for item in iarticle)
            batch_size = len(article_token_ids)
            x = np.zeros([batch_size, num_steps], np.int32)
            t = np.zeros([batch_size, num_steps], np.int32)
            w = np.zeros([batch_size, num_steps], np.uint8)

            num_steps_c = max(len(item) for item in article_token_ids_c)
            #num_steps = max(len(item) for item in iarticle)
            x_c = np.zeros([batch_size, num_steps_c], np.int32)
            t_c = np.zeros([batch_size, num_steps_c], np.int32)
            w_c = np.zeros([batch_size, num_steps_c], np.uint8)
            for i in range(batch_size):
                num_tokens = len(article_token_ids[i])
                x[i,:num_tokens] = article_token_ids[i]
                t[i,:num_tokens] = article_seg_ids[i]
                w[i,:num_tokens] = 1

                num_tokens_c = len(article_token_ids_c[i])
                x_c[i,:num_tokens_c] = article_token_ids_c[i]
                t_c[i,:num_tokens_c] = article_seg_ids_c[i]
                w_c[i,:num_tokens_c] = 1

            if not np.any(w):
                return
            out_x = torch.LongTensor(x)
            out_t = torch.LongTensor(t)
            out_w = torch.LongTensor(w)

            out_x_c = torch.LongTensor(x_c)
            out_t_c = torch.LongTensor(t_c)
            out_w_c = torch.LongTensor(w_c)

            yield  article, abstract, (out_x, out_t, out_w, out_x_c, out_t_c, out_w_c, pair_indice)

    def _2bert_rep(self, tokens_a, tokens_b=None):

        if tokens_b is None:
            tokens_a = tokens_a[: self._max_len - 2]
        else:
            self._truncate_seq_pair(tokens_a, tokens_b, self._max_len - 3)

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b is not None:

            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)

            tokens.append("[SEP]")
            segment_ids.append(1)
        #print(tokens)
        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        #print(input_ids)

        return input_ids, segment_ids

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

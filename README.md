# PacSum

This code is for paper [Sentence Centrality Revisited for Unsupervised Summarization](https://arxiv.org/pdf/1906.03508.pdf) ACL 2019

Some codes are borrowed from [pytorch_pretrained_bert] (https://github.com/huggingface/pytorch-transformers) and [gensim] (https://github.com/RaRe-Technologies/gensim)


-------
### Dependencies
  Python3.6, pytorch >= 1.0, numpy, gensim, pyrouge


-------
### Data used in the paper:

Download https://drive.google.com/open?id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6




### Tuning the hyperparamters and test the performance using tfidf or BERT representation
```
python run.py --rep tfidf --mode tune --tune_data_file path/to/validation/data --test_data_file path/to/test/data
```
```
python run.py --rep bert --mode tune --tune_data_file path/to/validation/data --test_data_file path/to/test/data --bert_model_file  path/to/model --bert_config_file path/to/config --bert_vocab_file path/to/vocab
```

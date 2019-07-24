from extractor import PacSumExtractorWithBert, PacSumExtractorWithTfIdf
from data_iterator import Dataset

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices = ['tune', 'test'], help='tune or test')
    parser.add_argument('--rep', type=str, choices = ['tfidf', 'bert'], help='tfidf or bert')
    parser.add_argument('--extract_num', type=int, default=3, help='number of extracted sentences')
    parser.add_argument('--bert_config_file', type=str, default='/disk/scratch1/s1797858/bert_model_path/uncased_L-12_H-768_A-12/bert_config.json', help='bert configuration file')
    parser.add_argument('--bert_model_file', type=str, help='bert model file')
    parser.add_argument('--bert_vocab_file', type=str,  default='/disk/scratch1/s1797858/bert_model_path/uncased_L-12_H-768_A-12/vocab.txt',help='bert vocabulary file')

    parser.add_argument('--beta', type=float, default=0., help='beta')
    parser.add_argument('--lambda1', type=float, default=0., help='lambda1')
    parser.add_argument('--lambda2', type=float, default=1., help='lambda2')

    parser.add_argument('--tune_data_file', type=str, help='data for tunining hyperparameters')
    parser.add_argument('--test_data_file', type=str, help='data for testing')



    args = parser.parse_args()
    print(args)

    if args.rep == 'tfidf':
        extractor = PacSumExtractorWithTfIdf(beta = args.beta,
                                             lambda1=args.lambda1,
                                             lambda2=args.lambda2)
        #tune
        if args.mode == 'tune':
            tune_dataset = Dataset(args.tune_data_file)
            tune_dataset_iterator = tune_dataset.iterate_once_doc_tfidf()
            extractor.tune_hparams(tune_dataset_iterator)

        #test
        test_dataset = Dataset(args.test_data_file)
        test_dataset_iterator = test_dataset.iterate_once_doc_tfidf()
        extractor.extract_summary(test_dataset_iterator)



    elif args.rep == 'bert':
        extractor = PacSumExtractorWithBert(bert_model_file = args.bert_model_file,
                                            bert_config_file = args.bert_config_file,
                                            beta = args.beta,
                                            lambda1=args.lambda1,
                                            lambda2=args.lambda2)
        #tune
        if args.mode == 'tune':
            tune_dataset = Dataset(args.tune_data_file, vocab_file = args.bert_vocab_file)
            tune_dataset_iterator = tune_dataset.iterate_once_doc_bert()
            extractor.tune_hparams(tune_dataset_iterator)

        #test
        test_dataset = Dataset(args.test_data_file, vocab_file = args.bert_vocab_file)
        test_dataset_iterator = test_dataset.iterate_once_doc_bert()
        extractor.extract_summary(test_dataset_iterator)

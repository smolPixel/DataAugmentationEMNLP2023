#Main file
from data.DataProcessor import initialize_dataset
from Classifier.classifier import classifier
from AugmentStrat.Augmentator import augmentator
from MetaStrategy.MetaStrat import meta_strat
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(argdict):
    results=[]
    results_dev=[]
    for seed in [0,1,2,3,4,5]:
        print(seed)
        set_seed(seed)
        train, dev=initialize_dataset(argdict)
        print(f"Original length of the dataset {len(train)}")
        augment_algo=augmentator(argdict)
        meta_augment=meta_strat(argdict, augment_algo)
        train=meta_augment.augment(train)
        print(f"Augmented length of the dataset {len(train)}")
        #Free memory
        # print(torch.cuda.memory_allocated())
        del augment_algo
        del meta_augment
        # print(torch.cuda.memory_allocated())
        # fds
        classifier_algo=classifier(argdict)
        results_train_iter, results_dev_iter=classifier_algo.train_test(train, dev)
        results.append(results_train_iter)
        results_dev.append(results_dev_iter)
        del classifier_algo
        print(results[-1])
    print(results)
    print(f"Average results: {np.mean(results_dev)}")
    return np.mean(results), np.mean(results_dev)

#66.74

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE for data augmentation')
    #General arguments on training
    parser.add_argument('--dataset', type=str, default='SST-2', help="dataset you want to run the process on. Includes SST2, TREC6, FakeNews")
    parser.add_argument('--classifier', type=str, default='bert', help="classifier you want to use. Includes LogReg, bert, jiant, svm_latent")
    parser.add_argument('--computer', type=str, default='home', help="Whether you run at home or at iro. Automatically changes the base path")
    parser.add_argument('--split', type=float, default=0, help='percent of the dataset added')
    parser.add_argument('--retrain', action='store_true', help='whether to retrain the VAE or not')
    parser.add_argument('--rerun', action='store_true', help='whether to rerun knowing that is has already been ran')
    parser.add_argument('--dataset_size', type=int, default=0, help='number of example in the original dataset. If 0, use the entire dataset')
    parser.add_argument('--algo', type=str, default='dummy', help='data augmentation algorithm to use, includes, VAE, CVAE, CVAE_Classic, SSVAE')
    parser.add_argument('--random_seed', type=int, default=7, help='Random seed ')
    parser.add_argument('--selector', type=str, default='random', help='the selection strategy. Choice of : random, confidence, confidenceNormal, Schumman')
    parser.add_argument('--max_seq_length', type=int, default=0, help='max length of the data, 0 for no max length')
    parser.add_argument('--meta_strategy', type=str, default="dummy", help="meta strategy to help filter the exemples, choice of dummy, confidence")

    parser.add_argument('--nb_epoch_classifier', type=int, default=4, help='number of epochs for training the classifier')
    parser.add_argument('--nb_epoch_algo', type=int, default=30, help='Number of epoch of the algo')
    parser.add_argument('--batch_size_algo', type=int, default=8, help='batch size of the data augmentation algo')
    parser.add_argument('--latent_size', type=int, default=5, help='Latent Size')
    parser.add_argument('--hidden_size_algo', type=int, default=1024, help='Hidden Size Algo')
    parser.add_argument('--dropout_algo', type=float, default=0.3, help='dropout of the classifier')
    parser.add_argument('--word_dropout', type=float, default=0.3, help='dropout of the classifier')
    parser.add_argument('--x0', default=1000, type=int, help='x0')
    parser.add_argument('--min_vocab_freq', default=1, type=int, help='min freq of vocab to be included')


    parser.add_argument('--corrupt_data', type=float, default=0, help='Corrupt X percent of the augmented data for experiment purpose')
    parser.add_argument('--test_nb_error', action='store_true', help='Testing on the number of examples of the wrong class vs the accuracy')
    parser.add_argument('--test_nb_error_dataset', action='store_true', help='Testing on the number of examples of the wrong class vs the accuracy, with examples from the dataset')
    parser.add_argument('--test_nb_doublons', action='store_true', help='Testing on the number of examples that are doublons vs the accuracy')
    parser.add_argument('--test_nb_doublons_algo', action='store_true', help='Testing on the number of examples that are doublons vs the accuracy')
    parser.add_argument('--test_grad', action='store_true', help='Shows the norm of the gradient for Regular vs Augmented examples.')
    parser.add_argument('--corrupt_normal', action='store_true', help='Passes the normal data through a DA algorithm')
    parser.add_argument('--remove_punctuation', action='store_true', help="Remove punctuation for all data")
    args = parser.parse_args()

    argdict = args.__dict__

    if argdict['corrupt_normal'] and not argdict['test_grad'] and not argdict['algo']=='EDA':
        raise ValueError("Corrupt normal is only available with test_grad and EDA")
    if argdict['remove_punctuation']:
        print("Warning, only available with test_grad")

    argdict['path']='~/Documents/DAControlled'
    if argdict['dataset'] == "SST-2":
        categories = ["neg", "pos"]
    elif argdict['dataset'] == "TREC6":
        categories = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
    elif argdict['dataset'] == "FakeNews":
        categories = ["Fake", "Real"]
    elif argdict['dataset'] == "QNLI":
        categories = ["entailment", "not_entailment"]
    elif argdict['dataset'] == "Irony":
        categories= ["NotIro", "Iro"]
    elif argdict['dataset'] == "IronyB":
        categories = ["Clash", "Situational", "Other", "NotIro"]
    else:
        raise ValueError("Dataset not found")
    argdict['categories'] = categories


    main(argdict)
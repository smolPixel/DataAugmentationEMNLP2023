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
import yaml
import time


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(argdict):
    results=[]
    results_dev=[]
    results_test=[]
    if argdict['one_shot']:
        seeds=[argdict['random_seed']]
    else:
        seeds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]


    for seed in seeds:#,15, 16,17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        print(seed)
        argdict['random_seed']=seed
        set_seed(seed)
        if argdict['algo'] in ['PVAE']:
            #we need to initialize the vocab object first from pretraining corpus
            from data.DataProcessor import initialize_dataset_pretraining
            _=initialize_dataset_pretraining(argdict, argdict['language'])
        train, dev, test=initialize_dataset(argdict)
        print(f"Original length of the dataset {len(train)}")
        augment_algo=augmentator(argdict)
        meta_augment=meta_strat(argdict, augment_algo)
        time0 = time.time()
        train=meta_augment.augment(train, dev, test)
        # print(time.time() - time0)
        # fds
        print(f"Augmented length of the dataset {len(train)}")
        #Free memory
        # print(torch.cuda.memory_allocated())
        del augment_algo
        del meta_augment
        # print(torch.cuda.memory_allocated())
        # fds
        # train_tsv=train.return_pandas()
        # train_tsv.to_csv("test.tsv", sep='\t')
        classifier_algo=classifier(argdict, train)
        results_train_iter, results_dev_iter, results_test_iter =classifier_algo.train_test(train, dev, test)
        results.append(results_train_iter)
        results_dev.append(results_dev_iter)
        results_test.append(results_test_iter)
        print(time.time()-time0)
        # fds
        del classifier_algo
        print(results[-1])
    print(results)
    print(results_test)
    print(f"Average results: {np.mean(results_test)}, stf : {np.std(results_test)}, with a split of {argdict['split']}")
    return np.mean(results), np.mean(results_dev), np.mean(results_test)

#66.74

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE for data augmentation')
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    args = args.__dict__

    stream = open(args['config_file'], "r")
    argdict = yaml.safe_load(stream)

    # if argdict['corrupt_normal'] and not argdict['test_grad'] and not argdict['algo']=='EDA':
    #     raise ValueError("Corrupt normal is only available with test_grad and EDA")
    # if argdict['remove_punctuation']:
    #     print("Warning, only available with test_grad")

    argdict['path']='~/Documents/DAControlled'
    if argdict['dataset'] == "SST2":
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
        categories = [ "NotIro", "Clash", "Situational", "Other"]
    elif argdict['dataset'] in ["MPhasisDe", "MPhasisFr"]:
        categories = ["negative", "positive"]
    elif argdict['dataset'] == 'ko3i4k':
        categories=['fragment', 'statement', 'question', 'command', 'rethquest', 'rethcomm', 'intodeputt']
    elif argdict['dataset'] == 'koHateSpeech':
        categories=['hate', 'none', 'offensive']
    elif argdict['dataset']=='Swahili':
        categories=['kimataifa', 'kitaifa', 'michezo', 'afya', 'burudani', 'uchumi']
    elif argdict['dataset']=='SB10k':
        categories=['negative', 'neutral', 'positive']
    elif argdict['dataset']=='CLS':
        categories=['negative', 'positive']
    elif argdict['dataset'] == 'MNIST':
        categories= [0,1,2,3,4,5,6,7,8,9]
    else:
        raise ValueError("Dataset not found")
    argdict['categories'] = categories

    main(argdict)
"""Simple strategy that consists in copying the dataset twice"""
"""EDA augmentation"""
import pandas as pd
import os
import re

class ChatGPTDummy():

    def __init__(self, argdict):
        self.argdict=argdict
        self.num_classes=len(argdict['categories'])
        self.path_answers=f'AugmentStrat/ChatGPTData/{self.argdict["dataset"]}/0/aug_sentences.tsv'

    def process_answer(self, answer, n):
        #Takes in a chatgpt answer and returns a list
        answer=answer.split('\n')
        answer=[aa.strip()[2:].strip() for aa in answer]
        return answer

    def augment(self, train, return_dict):
        bs = 10
        # We put some safety measure in place since we're dealing with money
        #First, we'll append all output to a tmp.txt file
        file_safety=open('Temp.txt', 'a')
        file_safety.write('-------')

        try:
            df=pd.read_csv(self.path_answers, sep='\t', index_col=0)
            print(df)
        except:
            pass




        # training_set = pd.read_csv(f"Temp/data_{i}.tsv", sep='\t', index_col=0)
        if self.argdict['split'] > 1:
            for j, row in train.iterrows():
                request = f"Create {self.argdict['split']} paraphrases of the following sentence : {row['sentence']}"
        else:
            df=pd.read_csv(f'data/{self.argdict["dataset"]}/train.tsv', sep='\t')
            list_sents=list(df['sentence'])
            list_labels=list(df['label'])
            request=""
            og_sents=""
            og_labels=""
            for i in range(0, len(list_sents), bs):
                request += "Create a paraphrase for each of the following sentences: "
                batch_sents = list_sents[i:i+bs]
                batch_labels= list_labels[i:i+bs]
                request += " ".join([f'{j}. {ss}' for j, ss in enumerate(batch_sents)])
                request += " <SEP> "
                og_sents += " <minisep> ".join(batch_sents)
                og_sents +=" <SEP> "
                og_labels += " <minisep> ".join([str(lab) for lab in batch_labels])
                og_labels += " <SEP> "
            with open("Temp/to_label_chatgpt.txt", "w") as f:
                f.write(request)
            with open("Temp/og_label.txt", "w") as f:
                f.write(og_labels)
            with open("Temp/og_sents.txt", "w") as f:
                f.write(og_sents)
            fds



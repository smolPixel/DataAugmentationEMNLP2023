import random
import numpy as np
import math

class AEDA():

    def __init__(self, argdict):
        self.argdict=argdict

    def aeda(self, sentence):
        sent=sentence.split(' ')
        len_seq=len(sent)
        max_ins=math.ceil(len_seq/3)
        num_ins=random.randint(1, max_ins)
        indexes=[i for i in range(len_seq, -1, -1)]
        indexes=np.random.choice(indexes, size=num_ins, replace=False)
        indexes=np.sort(indexes)[::-1]
        punctuations=["?", ".", ";", ":", "!", ","]
        for index in indexes:
            sent.insert(index, np.random.choice(punctuations))

        return " ".join(sent)


    def augment(self, train, dev, return_dict=False):
        split = self.argdict['split']  # Percent of data points added
        num_points = len(train)
        # Generate the maximum number of points, that is 5 times the dataset per class
        self.argdict['num_to_add'] = round(num_points * split)
        #The full training dataset has 97 XXX examples, hence start the index at 100 000
        i=len(train)
        diconew={}
        # sentence='makes a joke out of car chases for an hour and then gives us half an hour of car chases .'
        #TODO ADD DATA IN FOLDER
        for i in range(int(self.argdict['split'])):
            for j, ex in train.iterexamples():
                line=ex['sentence']
                label=ex['label']
                # print('---')
                # print(len(line.split()))
                # print(line)
                # print("Bitch")
                sentAug=self.aeda(line)
                # print(len(sentAug.split()))
                # if j==5:
                #     fds
                diconew[len(diconew)]={'sentence':sentAug,
                                 'label':label,
                                 'input':train.tokenize(sentAug),
                                 'augmented':True}
                i+=1

        if return_dict:
            return diconew
        for j, item in diconew.items():
            len_data=len(train)
            # print(item)
            train.data[len_data]=item
        # train.to_csv(f'/Tmp/train_{self.argdict["dataset"]}.tsv', sep='\t')
        # train.return_pandas().to_csv(f'AEDA_train_{self.argdict["dataset"]}.tsv', sep='\t')
        # fds
        return train

    def augment_doublons(self, train, n):
        split = self.argdict['split']  # Percent of data points added
        num_points = len(train)
        # Generate the maximum number of points, that is 5 times the dataset per class
        self.argdict['num_to_add'] = round(num_points * split)
        #The full training dataset has 97 XXX examples, hence start the index at 100 000
        i=len(train)
        diconew={}
        numFalsePos=0
        numFalseNeg=0
        for i in range(int(self.argdict['split'])):
            for j, ex in train.iterexamples():
                line=ex['sentence']
                label=ex['label']
                if numFalsePos<n and label==1:
                    numFalsePos+=1
                    sentAug=line
                elif numFalseNeg<n and label==0:
                    numFalseNeg+=1
                    sentAug=line
                else:
                    sentAug = self.aeda(line)
                # print('---')
                # print(len(line.split()))
                # print(line)
                # print("Bitch")
                # sentAug=eda(line, 0.1, 0.1, 0.1, 0.1, 'en')
                # print(len(sentAug.split()))
                # print(sentAug)
                diconew[len(diconew)]={'sentence':sentAug,
                                 'label':label,
                                 'input':train.tokenize(sentAug),
                                 'augmented':True}
        for j, item in diconew.items():
            len_data=len(train)
            # print(item)
            train.data[len_data]=item
        train.return_pandas().to_csv(f'training_{n}.tsv', sep='\t')
        return train
"""Perform a perfect DA, by adding data from the training dataset that is not in the split"""

"""EDA augmentation"""
import pandas
from data.DataProcessor import initialize_dataset
import copy
from AugmentStrat.EDA_strat.EDA import eda

class PerfectEDA():

    def __init__(self, argdict):
        self.argdict=argdict

    def augment(self, train):
        split = self.argdict['split']  # Percent of data points added
        num_points = len(train)
        # Generate the maximum number of points, that is 5 times the dataset per class
        self.argdict['num_to_add'] = round(num_points * split)
        #The full training dataset has 97 XXX examples, hence start the index at 100 000
        i=len(train)
        if self.argdict['corrupt_data']>0:
            nb_to_corrupt=num_points*split*self.argdict['corrupt_data']/len(self.argdict['categories'])
            print(f"corrputing {nb_to_corrupt} per categorie")
            nb_corrupt_pos=0
            nb_corrupt_neg=0

        diconew={}

        argdictTemp=copy.deepcopy(self.argdict)
        argdictTemp['dataset_size']=0

        train_complete, _=initialize_dataset(argdictTemp)
        # print(len(train_complete))

        num_per_class_to_add=num_points/len(self.argdict['categories'])
        print(f"Adding {num_per_class_to_add} points per category")
        num_pos=0
        num_neg=0

        # train_complete=train_complete.return_pandas()
        list_ex=list(train.return_pandas()['sentence'])

        if self.argdict['corrupt_data'] > 0:
            nb_to_corrupt = num_points * split * self.argdict['corrupt_data'] / len(self.argdict['categories'])
            print(f"corrputing {nb_to_corrupt} per categorie")
            nb_corrupt_pos = 0
            nb_corrupt_neg = 0

        for i, ex in train_complete.iterexamples():
            ex=copy.deepcopy(ex)
            ex['augmented']=True
            if ex['sentence'] in list_ex:
                continue
            elif ex['label']==0 and num_neg<num_per_class_to_add:
                if self.argdict['corrupt_data'] > 0 and nb_corrupt_neg<nb_to_corrupt:
                    ex['label']=1
                    nb_corrupt_neg+=1
                sentAug = eda(ex['sentence'], 0.1, 0, 0, 0, 1)[0]
                ex['sentence'] = sentAug
                train[len(train)]=ex
                num_neg+=1
            elif ex['label']==1 and num_pos<num_per_class_to_add:
                if self.argdict['corrupt_data'] > 0 and nb_corrupt_pos<nb_to_corrupt:
                    ex['label']=0
                    nb_corrupt_pos+=1


                sentAug=eda(ex['sentence'], 0.1, 0, 0, 0, 1)[0]
                ex['sentence']=sentAug
                train[len(train)]=ex
                num_pos+=1
        train.return_pandas().to_csv(f'TESTINGCSV.tsv', sep='\t')
        fsd
        return train

        # Creating new dataset
    def augment_false(self, train, n):
        split = self.argdict['split']  # Percent of data points added
        num_points = len(train)
        # Generate the maximum number of points, that is 5 times the dataset per class
        self.argdict['num_to_add'] = round(num_points * split)
        #The full training dataset has 97 XXX examples, hence start the index at 100 000
        diconew={}
        numFalsePos=0
        numFalseNeg=0

        #TODO ADD DATA IN FOLDER
        for i in range(int(self.argdict['split'])):
            for j, ex in train.iterexamples():
                line=ex['sentence']
                label=ex['label']
                if numFalsePos<n and label==1:
                    label=0
                    numFalsePos+=1
                elif numFalseNeg<n and label==0:
                    numFalseNeg+=1
                    label=1
                # print('---')
                # print(len(line.split()))
                # print(line)
                # print("Bitch")
                sentAug=eda(line, 0.1, 0, 0, 0, 'en')
                # print(len(sentAug.split()))
                # print(sentAug)
                diconew[len(diconew)]={'sentence':sentAug,
                                 'label':label,
                                 'input':train.tokenize(sentAug)}
        for j, item in diconew.items():
            len_data=len(train)
            # print(item)
            train[len_data]=item
        # train.to_csv(f'/Tmp/train_{self.argdict["dataset"]}.tsv', sep='\t')
        return train

    def augment_doublons(self, train, n):
        split = self.argdict['split']  # Percent of data points added
        num_points = len(train)
        # Generate the maximum number of points, that is 5 times the dataset per class
        # self.argdict['num_to_add'] = round(num_points * split)
        #The full training dataset has 97 XXX examples, hence start the index at 100 000
        i=len(train)
        diconew={}
        numFalsePos=0
        numFalseNeg=0
        print(n)

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
                    sentAug = eda(line, 0.05, 0.05, 0.05, 0.05, 1)[0]

                # print('---')
                # print(len(line.split()))
                # print(line)
                # print("Bitch")
                # sentAug=eda(line, 0.1, 0.1, 0.1, 0.1, 'en')
                # print(len(sentAug.split()))
                # print(sentAug)
                diconew[len(diconew)]={'sentence':sentAug,
                                 'label':label,
                                 'input':train.tokenize(sentAug)}
        for j, item in diconew.items():
            len_data=len(train)
            # print(item)
            train.data[len_data]=item
        # train.return_pandas().to_csv(f'test.tsv', sep='\t')
        return train

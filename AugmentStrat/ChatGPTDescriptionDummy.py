
import pandas as pd
import os
import re
# "Clash", "Situational", "Other", "NotIro"
class ChatGPTDescriptionDummy():

    def __init__(self, argdict):
        self.argdict=argdict
        self.num_classes=len(argdict['categories'])
        self.path_answers=f'AugmentStrat/ChatGPTDescriptionData/{self.argdict["dataset"]}/0/aug_sentences.tsv'
        self.description={"SST2": {0: ("movie reviews", "negative or somewhat negative"), 1: ("movie reviews", "positive or somewhat positive")},
                          "FakeNews": {0: ("headline Fake/Real news classification", "Real"), 1: ("headline Fake/Real news classification", "Fake")},
                          "Irony": {0: ("Ironic tweet detection", "Non Ironic Tweets"), 1: ("Ironic tweet detection", "Ironic Tweets")},
                          "IronyB": {1: ("Ironic tweet detection", "Tweets ironic by polarity contrast, where the polarity is inverted between the literal and intended evaluation"),
                                     2: ("Ironic tweet detection", "Tweets ironic by Situational Irony, where a situation fails to meet some expectation"),
                                     3: ("Ironic tweet detection", "Tweets ironic by Other type of Irony, where the Irony is neither by Polarity Contrast or by Situational Irony"),
                                     0: ("Ironic tweet detection", "Tweets that are not ironic")},
                          "TREC6": {0: ("Question classification", "Question about an abbreviation"),
                                     1: ("Question classification", "Question about an entity (event, animal, language, etc)"),
                                     2: ("Question classification", "Question concerning a description (of something, a definition, a reason, etc)"),
                                     3: ("Question classification", "Question about a human  (description of someone, an individual, etc)"),
                                    4: ("Question classification", "Question about a location"),
                                    5: ("Question classification", "Question about something numerical (weight, price, any other number)")}
                          }


    def augment(self, train, return_dict):
        bs = 10
        # We put some safety measure in place since we're dealing with money
        #First, we'll append all output to a tmp.txt file
        file_safety=open('Temp.txt', 'a')
        file_safety.write('-------')

        try:
            df=pd.read_csv(self.path_answers, sep='\t', index_col=0)
            fds
        except:
            pass


        if self.argdict['split'] > 1:
            for j, row in train.iterrows():
                request = f"Create {self.argdict['split']} paraphrases of the following sentence : {row['sentence']}"
        else:
            request=""
            label=""
            df=pd.read_csv(f'data/{self.argdict["dataset"]}/train.tsv', sep='\t')
            print(len(df))
            for i in range(len(self.argdict['categories'])):
                num_cc=len(df[df['label']==i])
                print(num_cc)
                for j in range(0, num_cc, bs):
                    request+=f"Generate 10 new sentences that you haven't generated before for a dataset of {self.description[self.argdict['dataset']][i][0]} which would be {self.description[self.argdict['dataset']][i][1]}"
                    request+=" <SEP> "
                    label+=f" {i} <SEP> "
            with open("Temp/to_generate_chatgpt.txt", "w") as f:
                f.write(request)
            with open("Temp/to_generate_label_chatgpt.txt", "w") as f:
                f.write(label)
            fds



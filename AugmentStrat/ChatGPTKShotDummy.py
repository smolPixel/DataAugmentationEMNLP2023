
import pandas as pd
import os
import re
import math
# "Clash", "Situational", "Other", "NotIro"
class ChatGPTKShotDummy():

	def __init__(self, argdict):
		self.argdict=argdict
		self.num_classes=len(argdict['categories'])
		# assert self.argdict['dataset'] in ['SST2']
		self.path_answers=f'AugmentStrat/ChatGPT{self.argdict["kshot"]}ShotData/{self.argdict["dataset"]}/0/aug_sentences.tsv'
		self.description={"SST2": {0: ("movie reviews", "negative or somewhat negative"), 1: ("movie reviews", "positive or somewhat positive")},
						  "FakeNews": {0: ("headlineFake/Real news classification", "Real"), 1: ("headlineFake/Real news classification", "Fake")},
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
		self.class_mapping={'SST2':{0:'negative or somewhat negative', 1:'positive or somewhat positive'},
							'FakeNews': {0:'Real', 1:'Fake'}}


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


		# training_set = pd.read_csv(f"Temp/data_{i}.tsv", sep='\t', index_col=0)
		# if self.argdict['split'] > 1:
		# 	for j, row in train.iterrows():
		# 		request = f"Create {self.argdict['split']} paraphrases of the following sentence : {row['sentence']}"
		# else:
		request=""
		label=""
		og_sentence=""
		df_full=pd.read_csv(f'data/{self.argdict["dataset"]}/train.tsv', sep='\t')
		train_pandas=train.return_pandas()
		train_pandas.to_csv(f'Temp/df_{self.argdict["random_seed"]}.tsv', sep='\t')
		print(len(df_full))
		print(len(train))
		print(self.argdict['random_seed'])
		if self.argdict['random_seed']==14:
			for ss in range(15):
				df=pd.read_csv(f'Temp/df_{ss}.tsv', sep='\t')
				#Construct prompts
				for i in range(len(self.argdict['categories'])):
					#We want to generate enough data to cover the full dataset, being sure to samples from small sets at every turn.
					num_cc = math.ceil(len(df_full[df_full['label'] == i])/15)

					#We want to generate one shot enough examples for the case 0
					for j in range(0, num_cc, bs):
						request += f"Generate 10 new sentences that you haven't generated before for a dataset of {self.description[self.argdict['dataset']][i][0]} " \
								   f"which would be {self.description[self.argdict['dataset']][i][1]}. Here are the classes as well as examples. \n"
						sents=[df[df['label']==n].sample(n=1)['sentence'].item() for n in range(len(self.argdict['categories']))]
						request += "\n".join([f"{self.class_mapping[self.argdict['dataset']][n]} : {sents[n]}" for n in range(len(self.argdict['categories']))])
						request += " <SEP> "
						og_sentence += " <minisep> ".join(sents)
						og_sentence += " <SEP> "
						label += f" {i} <SEP> "
			with open("Temp/to_generate_chatgpt.txt", "w") as f:
				f.write(request)
			with open("Temp/to_generate_ogsentences.txt", "w") as f:
				f.write(og_sentence)
			with open("Temp/to_generate_label_chatgpt.txt", "w") as f:
				f.write(label)
				fds

		return train
		for i in range(len(self.argdict['categories'])):
			num_cc=len(df[df['label']==i])
			print(num_cc)
			for j in range(0, num_cc, bs):
				request+=f"Generate 10 new sentences that you haven't generated before for a dataset of {self.description[self.argdict['dataset']][i][0]} " \
						 f"which would be {self.description[self.argdict['dataset']][i][1]}. Here are the classes as well as examples." \
						 f""
				request+=" <SEP> "
				label+=f" {i} <SEP> "
		with open("Temp/to_generate_chatgpt.txt", "w") as f:
			f.write(request)
		with open("Temp/to_generate_label_chatgpt.txt", "w") as f:
			f.write(label)
			fds



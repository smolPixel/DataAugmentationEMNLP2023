"""Bart augmentation"""
from os import fdatasync
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, WarmUp, BartConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import math
import numpy as np
import re
import copy
class BART():

	def __init__(self, argdict):
		self.argdict=argdict
		self.model='cuda'
		self.init_model()
		self.algo_is_trained = False



	def init_model(self):
		gptPath = 'facebook/bart-large'
		self.tokenizer = BartTokenizer.from_pretrained(gptPath)
		self.model = BartForConditionalGeneration.from_pretrained(gptPath, cache_dir='/tmp').to(self.model)
		self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

	def finetune(self, train):

		for i in range(self.argdict['nb_epoch_algo']):
			data_loader = DataLoader(
				dataset=train,
				batch_size=self.argdict['batch_size_algo'],
				shuffle=True,
				# num_workers=cpu_count(),
				pin_memory=torch.cuda.is_available()
			)
			optimizer = AdamW(self.model.parameters(), lr=self.argdict['learning_rate'])

			self.model.train()
			for _, batch in enumerate(tqdm(data_loader)):
				# batch=data_loader
				# mask is [MASK]
				optimizer.zero_grad()
				encoding, labels = self.mask_and_add_class(batch)
				input_ids = encoding['input_ids'].cuda()
				attention_mask = encoding['attention_mask'].cuda()
				decoder_attention_mask=labels['attention_mask'].cuda()
				decoder_input_ids=labels['input_ids'].cuda()
				outputs=self.model(input_ids, decoder_attention_mask=decoder_attention_mask, labels=decoder_input_ids, attention_mask=attention_mask)
				# print(outputs)
				try:
					loss = outputs[0]
				except:
					loss=outputs.loss
				loss.backward()

	def clean_sentence(self, sent):
		"""Clean a generated sentence"""
		#Assume we don'tgenerate pad tokens:
		clean_sent=[]
		sent=sent.split(" ")
		ll = ['[CLS]']
		ll.extend([f"[{cat}]" for cat in self.argdict['categories']])
		#TODO HERE
		for i, token in enumerate(sent):
			if token in ll:
				continue
			elif i!=len(sent)-1 and sent[i+1]=='[PAD]':
				break
			else:
				clean_sent.append(token)
		clean_sent=" ".join(clean_sent)
		# clean_sent=clean_sent.replace('.', ' .')
		return clean_sent

	def augment(self, train, dev, return_dict=False):
		if not self.algo_is_trained:
			self.finetune(train)
			self.algo_is_trained=True

		#Create augmented data
		data_loader = DataLoader(
			dataset=train,
			batch_size=self.argdict['batch_size_generation'],
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		self.model.eval()
		new_data={}

		num_to_gen=self.argdict['split']*len(train)
		num_gen=0

		while num_gen<num_to_gen:
			# Create augmented data
			data_loader = DataLoader(
				dataset=train,
				batch_size=8,
				shuffle=False,
				# num_workers=cpu_count(),
				pin_memory=torch.cuda.is_available()
			)

			for batch in data_loader:
				#For each batch, do it X time
				for i in range(int(self.argdict['split'])):
					encoding, labels = self.mask_and_add_class(batch)
					input_ids = encoding['input_ids'].cuda()
					attention_mask = encoding['attention_mask'].cuda()
					decoder_attention_mask=labels['attention_mask'].cuda()
					decoder_input_ids=labels['input_ids'].cuda()
					# encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
					#TODO GENERATE ERE
					gend = self.model.generate(inputs=input_ids,attention_mask=attention_mask, num_beams=10, num_return_sequences=1, max_length=50)
					sentences = self.tokenizer.batch_decode(gend, skip_special_tokens=True)

					# fds
					# outputs=self.model(input_ids,  attention_mask=attention_mask)


					for sent, lab, og_sent in zip(sentences, batch['label'], batch['sentence']):
						sent=self.clean_sentence(sent)
						if sent.replace('.', '').strip()==og_sent.replace('.', '').strip():
							continue
						num_gen+=1
						new_data[len(new_data)] = {'sentence': sent,
												  'label': int(lab.item()),
												  'input': train.tokenize(sent),
												  'augmented': True}
						#This is not the most elegant way to do it but wtv
						if num_gen>=num_to_gen:
							break
				if num_gen >= num_to_gen:
					break
			if num_gen >= num_to_gen:
				break
				# fds
				# print(self.clean_sentence(sent), int(lab.item()))
		if return_dict:
			return new_data
		for j, item in new_data.items():
			len_data=len(train)
			# print(item)
			train.data[len_data]=item
		# train.return_pandas().to_csv(f'CBART_train_{self.argdict["dataset"]}.tsv', sep='\t')
		# fds
		return train

		# Creating new dataset


	def mask_and_add_class(self, batch):
		text_sentence=[]
		sentences=copy.deepcopy(batch['sentence'])
		input_decoder=self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
		for i, ss in enumerate(sentences):
			splitted=ss.split(' ')
			ll=len(splitted)
			num_mask=math.floor(0.4*ll)
			ll_int=[i for i in range(ll)]
			indexes=np.random.choice(ll_int, size=num_mask, replace=False)
			for ind in indexes:
				splitted[ind]=self.tokenizer.mask_token
			ss=" ".join(splitted)
			ss=re.sub(r'(<mask> )+', '<mask> ', ss)
			sentences[i]=ss

		for sent, label in zip(sentences, batch['label']):
			classe=f"[{self.argdict['categories'][label.item()]}]"
			text_sentence.append(f"{classe} {sent}")
		# print(batch['sentence'])
		# print(text_sentence)
		# fds
		input_encoder=self.tokenizer(text_sentence, return_tensors='pt', padding=True, truncation=True)
		return input_encoder, input_decoder

	def augment_false(self, train, n):
		pass
		return train

	def augment_doublons(self, train, n):
		train_df=train.return_pandas()
		train_df.to_csv("AugmentStrat/CBERT_strat/datasets/binaryData/train.tsv", sep='\t', index=False)
		dev = pd.read_csv(f'data/{self.argdict["dataset"]}/dev.tsv', sep='\t')
		dev.to_csv("AugmentStrat/CBERT_strat/datasets/binaryData/dev.tsv", sep='\t', index=False)
		fine_tune_model(self.argdict)
		fds
		pass
		return train

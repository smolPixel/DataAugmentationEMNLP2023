"""GPT augmentation"""
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, AdamW
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from multiprocessing import cpu_count
import numpy as np



class GPT():

	def __init__(self, argdict):
		self.argdict=argdict
		self.init_model()
		self.device='cuda'

		self.algo_is_trained = False
		self.special_tokens=['<sep>', '<bos>', '\n']
		self.special_tokens.extend([f"[{cat}]" for cat in argdict['categories']])
		self.special_tokens.extend([f"{cat}" for cat in argdict['categories']])


	def forward(self, tokenized_sentences):


		input_ids=torch.Tensor(tokenized_sentences['input_ids']).long().to(self.device)
		attention_mask=torch.Tensor(tokenized_sentences['attention_mask']).to(self.device)
		# outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)+trg[0,0]
		#TODO ATTENTION MASK
		outputs=self.model(input_ids, labels=input_ids, attention_mask=attention_mask)

		return outputs

	def init_model(self):
		self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
		special_tokens = {'pad_token': '<pad>', 'sep_token': '<sep>', 'eos_token': '<eos>', 'bos_token': '<bos>'}
		num_add_toks = self.tokenizer.add_special_tokens(special_tokens)
		self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
		model = GPT2LMHeadModel.from_pretrained('gpt2-large', cache_dir='/tmp')
		self.model = model.to('cuda')
		# self.model.config.max_length = 64
		self.model.resize_token_embeddings(len(self.tokenizer))
		self.model.config.pad_token_id = self.model.config.eos_token_id

		self.optimizer = AdamW(self.model.parameters(), lr=3e-5)

	def finetune(self, train):

		for i in range(self.argdict['num_epoch_algo']):
			iterator = DataLoader(
				dataset=train,
				batch_size=self.argdict['batch_size_algo'],
				shuffle=True,
				num_workers=cpu_count(),
				pin_memory=torch.cuda.is_available()
			)
			self.model.train()
			num_examples = len(train)
			loss_tot = 0
			# tokenized_sentence = torch.Tensor(self.tokenizer.encode(input['gpt2Sentence']))
			self.optimizer.zero_grad()
			for i, batch in enumerate(tqdm(iterator)):

				# print(batch['sentence_generation'])
				sentences=[]

				for lab, sent in zip(batch['label'], batch['sentence']):
					# labText="[pos]" if lab==1 else "[neg]"
					labText=f"[{self.argdict['categories'][lab]}]"
					# print(labText)
					sentences.append("<bos> "+labText+" <sep> "+sent+" <eos>")

				src = self.tokenizer(sentences, padding=True, truncation=True)
				# # print(src)
				# # self.optimizer.zero_grad()
				#
				output = self.forward(src)
				# print(output['logits'].shape)
				# print(output)
				try:
					loss = output['loss']
				except:
					loss=output[0]
				loss_tot += loss.item()
				loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad()
				# print(split)
			# prec15, rec15, f115=get_prf_at(self.training_set if split=="train" else self.dev_set, self.tokenizer, self.model, 15, self.argdict, print_results=False)
			print(loss)
		return loss

	def clean_sentence(self, sent):
		"""Clean a generated sentence"""
		#Assume we don'tgenerate pad tokens:
		clean_sents=[]
		for ss in sent:
			clean_sent=[]
			ss=ss.split(" ")
			for i, token in enumerate(ss):
				if token in self.special_tokens:
					continue
				elif token in ['<pad>', '<eos>']:
					break
				else:
					clean_sent.append(token)
			clean_sents.append(" ".join(clean_sent))
		return clean_sents

	def augment(self, train, dev, return_dict=False):

		if not self.algo_is_trained:
			self.finetune(train)
			self.algo_is_trained=True
		# Create augmented data
		data_loader = DataLoader(
			dataset=train,
			batch_size=32,
			shuffle=True,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		new_data = {}
		num_generated=[0]*len(self.argdict['categories'])
		num_to_gen=len(train)*self.argdict['split']/len(self.argdict['categories'])

		bs=16

		all_sent=list(train.return_pandas()['sentence'])


		for lab in range(len(num_generated)):
			attempt_num = 0
			while True:
				print(f"Attempt at data augmentation number {attempt_num}, new examples generated {num_generated[lab]}/{num_to_gen}")
				with torch.no_grad():
					sentences=["<bos> "+self.argdict['categories'][lab]+" <sep> "]*bs
					src = self.tokenizer(sentences, return_tensors='pt', padding=True)
					# sentAug=generate_text(self.tokenizer, self.model, sentence)

					sentAug = self.model.generate(src['input_ids'].cuda(), no_repeat_ngram_size=3, top_p=0.95,
												  do_sample=True, early_stopping=True, num_beams=1, )
					# print(sentAug)
					sentAug = self.tokenizer.batch_decode(sentAug)
					# print(sentAug)
					sentAug = self.clean_sentence(sentAug)
					# print(sentAug)
					for j, ss in enumerate(sentAug):
						if ss in all_sent or ss.strip() == "":
							continue
						new_data[len(new_data)] = {'sentence': ss,
												   'label': int(lab),
												   'input': train.tokenize("<bos> " + ss + " <eos>"),
												   'augmented': True}
						all_sent.append(ss)
						num_generated[lab]+=1
						# print(num_generated)
						if num_generated[lab]>=num_to_gen:
							break
					attempt_num+=1
				
				if num_generated[lab] >= num_to_gen:
					break


		#
		# fd
		# with torch.no_grad():
		#
		# 	self.model.eval()
		# 	for batch in data_loader:
		# 		# break
		# 		# print(torch.cuda.memory_allocated())
		# 		sentences=[]
		# 		for lab, sent in zip(batch['label'], batch['sentence']):
		# 			# labText="[pos]" if lab==1 else "[neg]"
		# 			labText=f"[{self.argdict['categories'][lab]}]"
		# 			# print(labText)
		# 			sentences.append("<bos> "+labText+" <sep> ")
		# 		#For each batch, do it X time
		# 		for i in range(int(self.argdict['split'])):
		# 			src = self.tokenizer(sentences, return_tensors='pt', padding=True)
		# 			# sentAug=generate_text(self.tokenizer, self.model, sentence)
		# 			sentAug=self.model.generate(src['input_ids'].cuda(), no_repeat_ngram_size=3, top_p=0.95,
		# 										do_sample=True, early_stopping=False, num_beams=10)
		# 			sentAug=self.tokenizer.batch_decode(sentAug)
		# 			sentAug=self.clean_sentence(sentAug)
		# 			for j, (ll, ss) in enumerate(zip(batch['label'], sentAug)):
		# 				if ss.strip()=="":
		# 					ss="Erroneous empty sentence"
		# 				new_data[len(new_data)] = {'sentence': ss,
		# 										   'label': int(ll.item()),
		# 										   'input': train.tokenize("<bos> "+ss+" <eos>"),
		# 										   'augmented': True}

		if return_dict:
			return new_data
		for j, item in new_data.items():
			len_data = len(train)
			# print(item)
			train.data[len_data] = item

		# train.return_pandas().to_csv(f'CGPT_train_{self.argdict["dataset"]}.tsv', sep='\t')
		# fds
		return train

		# Creating new dataset

	def recreate_sentence(self, bert_output, mask, bert_input):
		"""Recreate original sentence but with new, predicted words from bert"""

		sentences=torch.where(mask.cuda(), bert_output, bert_input)
		return self.tokenizer.batch_decode(sentences)

	def mask_and_add_class(self, batch):
		text_sentence=[]
		for sent, label in zip(batch['sentence'], batch['label']):
			classe=f"[{self.argdict['categories'][label.item()]}]"
			text_sentence.append(f"{classe} {sent}")

		input=self.tokenizer(text_sentence, return_tensors='pt', padding=True, truncation=True)
		labels=input['input_ids'].clone()
		mask=torch.zeros_like(input['input_ids']).float().uniform_() > 0.7
		#We dont want to predict words that are not in the sentence, use atttention mask to insure there are not masked
		maskformask=input['attention_mask']>0
		mask[~maskformask]=False
		#First token and class token can't be masked. PAD tokens can be masked since attention mask will prevent looking at them anyway
		mask[:, :2]=False
		input['input_ids'][mask]=self.tokenizer.mask_token_id
		# print(input)
		# print(self.tokenizer.batch_decode(input['input_ids']))
		labels=labels.masked_fill(~mask, -100)
		return input, labels, mask

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

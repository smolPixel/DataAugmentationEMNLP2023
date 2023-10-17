import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pandas as pd
from nltk.tokenize import TweetTokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms
import copy
#
class Pretrain_dataset(Dataset):
	def __init__(self, data, tokenizer, vocab, argdict):
		super().__init__()
		"""data: tsv of the data
		   tokenizer: tokenizer trained
		   vocabInput+Output: vocab trained on train"""
		self.data = {}
		self.argdict=argdict
		if self.argdict['max_seq_length']==0:
			self.max_len = 0
			find_true_length=True
		else:
			self.max_len=self.argdict['max_seq_length']
			find_true_length=False
		self.vocab_object = vocab
		self.tokenizer = tokenizer
		if argdict['tokenizer'] in ['tweetTokenizer', 'PtweetTokenizer']:
			self.pad_idx = self.vocab_object['<pad>']
			self.bos_idx = self.vocab_object['<bos>']
			self.eos_idx = self.vocab_object['<eos>']
			self.unk_idx = self.vocab_object['<unk>']
		else:
			vv=tokenizer.get_vocab()
			self.pad_idx=vv['<pad>']
			self.bos_idx=vv['<bos>']
			self.eos_idx=vv['<eos>']
			self.unk_idx=vv['<unk>']


		argdict['pad_idx']=self.pad_idx
		argdict['bos_idx']=self.bos_idx
		argdict['unk_idx']=self.unk_idx

		index=0
		# mask=len(argdict['categories'])
		for i, row in data.iterrows():
			if argdict['tokenizer'] in ['tweetTokenizer', 'PtweetTokenizer']:
				tokenized_text = self.tokenizer.tokenize("<bos> " + row['sentence'].lower() + " <eos>")
				input = vocab(tokenized_text)
			else:
				input = tokenizer("<bos> " + row['sentence'].lower() + " <eos>")['input_ids']
			if find_true_length and len(input)>self.max_len:
				self.max_len=len(input)
			self.data[index] = {'sentence':row['sentence'].lower(), 'input': input}
			index+=1

	# def tokenize_and_vectorize(self, sentences):
	#     """Takes an array of sentences and return encoded data"""

	# def get_unlabelled(self):
	# 	dico={key:value for key, value in self.data.items() if value['label']==2}
	# 	return dico


	def tokenize(self, sentence):
		"Tokenize a sentence"
		return self.vocab_object(self.tokenizer.tokenize("<bos> " + sentence + " <eos>"))

	def reset_index(self):
		new_dat = {}
		for i, (j, dat) in enumerate(self.data.items()):
			new_dat[i] = dat
		self.data = new_dat

	@property
	def vocab_size(self):
		if self.argdict['tokenizer'] == 'mwordPiece':
			return len(self.tokenizer)
		else:
			return len(self.vocab_object)

	@property
	def vocab(self):
		return self.vocab_object.get_stoi()
		# fsd
		# return self.vocab_object

	def get_w2i(self):
		return self.vocab_object.get_stoi()

	def process_generated(self, exo):
		generated = idx2word(exo, i2w=self.get_i2w(),
							 pad_idx=self.get_w2i()['<pad>'],
							 eos_idx=self.get_w2i()['<eos>'])
		for sent in generated:
			print("------------------")
			print(sent)

	def get_i2w(self):
		return self.vocab_object.get_itos()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, item):
		input = self.data[item]['input'][:self.max_len]
		length= len(input)
		input.extend([self.pad_idx] * (self.max_len - len(input)))
		target=input[1:]
		input=input[:-1]
		return {
			'sentence': self.data[item]['sentence'],
			'length': length,
			'input': np.asarray(input, dtype=int),
			'target': np.asarray(target, dtype=int),
		}

	def get_texts(self):
		"""returns a list of the textual sentences in the dataset"""
		ll=[]
		for _, dat in self.data.items():
			ll.append(dat['sentence'])
		return ll

	def shape_for_loss_function(self, logp, target):
		target = target.contiguous().view(-1).cuda()
		logp = logp.view(-1, logp.size(2)).cuda()
		return logp, target

	def convert_tokens_to_string(self, tokens):
		if tokens==[]:
			return ""
		else:
			raise ValueError("Idk what this is supposed to return")

	def arr_to_sentences(self, array):
		sentences=[]
		for arr in array:
			arr=arr.int()
			sent=self.vocab_object.lookup_tokens(arr.tolist())
			ss=""
			for token in sent:
				if token== "<bos>":
					continue
				if token =="<eos>":
					break
				ss+=f" {token}"
			sentences.append(ss)
		return sentences

	def iterexamples(self):
		for i, ex in self.data.items():
			yield i, ex

	def return_pandas(self):
		"""Return a pandas version of the dataset"""
		dict={}
		for i, ex in self.iterexamples():
			dict[i]={'sentence':ex['sentence'], 'label':ex['label']}
		return pd.DataFrame.from_dict(dict, orient='index')
"""EDA augmentation"""
import subprocess
import pandas as pd
# from AugmentStrat.EDA_strat.EDA import eda
from transformers import  AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import torch

class BackTranslationT5():

	def __init__(self, argdict):
		self.argdict=argdict

		self.bartPath = 'facebook/mbart-large-50-many-to-many-mmt'
		self.tokenizer = AutoTokenizer.from_pretrained(self.bartPath)
		self.model = AutoModelForSeq2SeqLM.from_pretrained(self.bartPath, cache_dir='/Tmp').cuda()
		self.tokenizers_code = {'en': 'en_XX', 'fr': 'fr_XX', 'nl': 'nl_XX', 'fi': 'fi_FI',
								'tl': 'en_XX', 'ar': 'ar_AR', 'pt': 'en_XX', 'de': 'de_DE',
								'es': 'es_XX', 'it': 'it_IT', 'ca': 'en_XX', 'ko': 'ko_KR',
								'el': 'en_XX', 'id': 'en_XX', 'sw':'sw_KE'}
		self.verbose = False

	def augment(self, train, dev, return_dict=False):
		split = self.argdict['split']  # Percent of data points added
		num_points = len(train)
		# Generate the maximum number of points, that is 5 times the dataset per class
		self.argdict['num_to_add'] = round(num_points * split)
		# The full training dataset has 97 XXX examples, hence start the index at 100 000
		i = len(train)

		data_loader = DataLoader(
			dataset=train,
			batch_size=4,  # self.argdict.batch_size,
			shuffle=False,
			num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)
		diconew = {}
		for batch in data_loader:
			for i in range(self.argdict['split']):
				translated=self.back_translate(batch['sentence'], lan_from=train.language, lan_to=self.argdict['lan_to'])
				for i, sentAug in enumerate(translated):
					if sentAug == "":
						print("WARNING ERRONEOUS SENTENCE")
						sentAug = "Error Temp"
					diconew[len(diconew)] = {'sentence': sentAug,
											'label': batch['label'][i].item(),
											'input': train.tokenize(sentAug),
											'augmented': True}
			# fds

		# 	fds
		# print(train)
		# fds
		#
		# diconew = {}
		# for i in range(int(self.argdict['split'])):
		# 	for j, ex in train.iterexamples():
		# 		line = ex['sentence']
		# 		label = ex['label']
		# 		# print(line)
		# 		sentAug=self.back_translate(line)[0]
		# 		# print(line)
		# 		# print(sentAug)
		# 		diconew[len(diconew)] = {'sentence': sentAug,
		# 								 'label': label,
		# 								 'input': train.tokenize(sentAug),
		# 								 'augmented': True}

		if return_dict:
			return diconew
		for j, item in diconew.items():
			len_data = len(train)
			# print(item)
			train.data[len_data] = item
		# train.to_csv(f'/Tmp/train_{self.argdict["dataset"]}.tsv', sep='\t')
		return train
		#
		# # Creating new dataset
	def back_translate(self, text, lan_from, lan_to):
		self.tokenizer.src_lan=self.tokenizers_code[lan_from]
		src = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
		for key, value in src.items():
			src[key]=value.cuda()
		translated_tokens=self.model.generate(**src, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tokenizers_code[lan_to]])
		translated=self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
		#Back translate
		# print(translated)
		self.tokenizer.src_lan=self.tokenizers_code[lan_to]
		src = self.tokenizer(translated, return_tensors='pt', padding=True, truncation=True)
		for key, value in src.items():
			src[key]=value.cuda()
		translated_tokens=self.model.generate(**src, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tokenizers_code[lan_from]])
		translated=self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
		# print(translated)
		# 	print("Returning Default due to Run Time Exception")
		# 	gfd
		# 	en_new = en
		return translated

	def en2de(self, input):
		input_ids = self.tokenizer_en_de(input, return_tensors="pt", padding=True, truncation=True)
		outputs = self.model_en_de.generate(input_ids['input_ids'])
		decoded = self.tokenizer_en_de.batch_decode(
			outputs, skip_special_tokens=True
		)
		if self.verbose:
			print(decoded)  # Maschinelles Lernen ist großartig, oder?
		return decoded

	def de2en(self, input):
		# input_ids = self.tokenizer_de_en.encode(input, return_tensors="pt")
		input_ids = self.tokenizer_de_en(input, return_tensors="pt", padding=True, truncation=True)
		# print(input_ids)
		# fds
		outputs = self.model_de_en.generate(
			input_ids['input_ids'],
			num_return_sequences=1,#int(self.argdict['split']),
			num_beams=self.num_beams,
		)
		decoded = self.tokenizer_de_en.batch_decode(
			outputs, skip_special_tokens=True
		)
		return decoded
		# predicted_outputs = []
		# for output in outputs:
		# 	decoded = self.tokenizer_de_en.decode(
		# 		output, skip_special_tokens=True
		# 	)
		# 	TODO: this should be able to return multiple sequences
			# predicted_outputs.append(decoded)
		# if self.verbose:
		# 	print(predicted_outputs)  # Machine learning is great, isn't it?
		# return predicted_outputs

	def generate(self, sentence: str):
		perturbs = self.back_translate(sentence)
		return perturbs
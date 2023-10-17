"""EDA augmentation"""
import subprocess
import pandas as pd
# from AugmentStrat.EDA_strat.EDA import eda
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from torch.utils.data import DataLoader
import torch

class BackTranslation():

	def __init__(self, argdict):
		self.argdict=argdict

		name_en_de = "facebook/wmt19-en-de"
		self.tokenizer_en_de = FSMTTokenizer.from_pretrained(name_en_de)
		self.model_en_de = FSMTForConditionalGeneration.from_pretrained(
			name_en_de
		).cuda()
		name_de_en = "facebook/wmt19-de-en"
		self.tokenizer_de_en = FSMTTokenizer.from_pretrained(name_de_en)
		self.model_de_en = FSMTForConditionalGeneration.from_pretrained(
			name_de_en
		).cuda()
		self.num_beams = 10
		self.verbose=False

	def augment(self, train, dev, return_dict=False):
		split = self.argdict['split']  # Percent of data points added
		num_points = len(train)
		# Generate the maximum number of points, that is 5 times the dataset per class
		self.argdict['num_to_add'] = round(num_points * split)
		# The full training dataset has 97 XXX examples, hence start the index at 100 000
		i = len(train)
		diconew = {}

		data_loader = DataLoader(
			dataset=train,
			batch_size=8,
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		# self.model.eval()
		new_data = []

		for batch in data_loader:
			for i in range(int(self.argdict['split'])):
				# For each batch, do it X time
				sents = batch['sentence']
				sentences = self.back_translate(sents)
				for ss_og, ss_aug, label in zip(sents, sentences, batch['label']):
					diconew[len(diconew)] = {'sentence': ss_aug,
											 'label': int(label.item()),
											 'input': train.tokenize(ss_aug),
											 'augmented': True}

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
		# train.return_pandas().to_csv(f'BT_train_{self.argdict["dataset"]}.tsv', sep='\t')
		# fds
		return train
		#
		# # Creating new dataset
	def back_translate(self, en: str):
		# try:
		de = self.en2de(en)
		en_new = self.de2en(de)
		# except Exception:
		# 	print("Returning Default due to Run Time Exception")
		# 	gfd
		# 	en_new = en
		return en_new


	def en2de(self, input):
		encoded = self.tokenizer_en_de(input, return_tensors="pt", padding=True, truncation=True)
		outputs = self.model_en_de.generate(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
		decoded = self.tokenizer_en_de.batch_decode(
			outputs, skip_special_tokens=True
		)
		# if self.verbose:
		# 	print(decoded)  # Maschinelles Lernen ist großartig, oder?
		return decoded

	def de2en(self, input):
		encoded = self.tokenizer_de_en(input, return_tensors="pt", padding=True, truncation=True)
		# print(input_ids)
		outputs = self.model_de_en.generate(
			encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda(),
			num_return_sequences=1,
			num_beams=self.num_beams,
		)
		predicted_outputs = []
		for output in outputs:
			decoded = self.tokenizer_de_en.decode(
				output, skip_special_tokens=True
			)
			# TODO: this should be able to return multiple sequences
			predicted_outputs.append(decoded)
		# if self.verbose:
		# 	print(predicted_outputs)  # Machine learning is great, isn't it?
		return predicted_outputs
	# def en2de(self, input):
	# 	input_ids = self.tokenizer_en_de.encode(input, return_tensors="pt")
	# 	outputs = self.model_en_de.generate(input_ids.cuda())
	# 	decoded = self.tokenizer_en_de.decode(
	# 		outputs[0], skip_special_tokens=True
	# 	)
	# 	if self.verbose:
	# 		print(decoded)  # Maschinelles Lernen ist großartig, oder?
	# 	return decoded
	#
	# def de2en(self, input):
	# 	input_ids = self.tokenizer_de_en.encode(input, return_tensors="pt")
	# 	# print(input_ids)
	# 	# fds
	# 	outputs = self.model_de_en.generate(
	# 		input_ids.cuda(),
	# 		num_return_sequences=int(self.argdict['split']),
	# 		num_beams=self.num_beams,
	# 	)
	# 	predicted_outputs = []
	# 	for output in outputs:
	# 		decoded = self.tokenizer_de_en.decode(
	# 			output, skip_special_tokens=True
	# 		)
	# 		# TODO: this should be able to return multiple sequences
	# 		predicted_outputs.append(decoded)
	# 	if self.verbose:
	# 		print(predicted_outputs)  # Machine learning is great, isn't it?
	# 	return predicted_outputs

	def generate(self, sentence: str):
		perturbs = self.back_translate(sentence)
		return perturbs
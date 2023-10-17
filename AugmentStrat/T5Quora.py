"""EDA augmentation"""
import pandas as pd
from AugmentStrat.CBERT_strat.cbert_finetune import fine_tune_model
from transformers import AutoModelWithLMHead, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import random
from tqdm import tqdm
import numpy as np

class T5ParaphraseQuora():

	def __init__(self, argdict):
		self.argdict=argdict
		self.init_model()
		self.algo_is_trained = False


	def init_model(self):
		self.tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-small-finetuned-quora-for-paraphrasing')
		self.model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing").cuda()

	def get_paraphrases(self, sentence, prefix="paraphrase: ", n_predictions=1, top_k=10, max_length=50):
		bs=len(sentence)
		texts = [prefix + sent + " </s>" for sent in sentence]
		encoding = self.tokenizer(
			texts, pad_to_max_length=True, return_tensors="pt"
		)
		input_ids, attention_masks = encoding["input_ids"].cuda(), encoding[
			"attention_mask"
		].cuda()

		model_output = self.model.generate(
			input_ids=input_ids,
			attention_mask=attention_masks,
			do_sample=True,
			max_length=max_length,
			top_k=top_k,
			top_p=0.7, #0.85, #0.9, #0.98,
			early_stopping=True,
			num_return_sequences=n_predictions,
		)
		generated_sent = self.tokenizer.batch_decode(
			model_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
		)
		generated_sent=np.array(generated_sent).reshape((bs, n_predictions))
		#split generated sentences into bunch

		return generated_sent
		# print(generated_sent)
		# outputs = []
		# for output in model_output:
		# 	generated_sent = self.tokenizer.batch_decode(
		# 		output, skip_special_tokens=True, clean_up_tokenization_spaces=True
		# 	)
		# 	print(generated_sent)
		# 	if (
		# 			generated_sent.lower() != sentence.lower()
		# 			and generated_sent not in outputs
		# 	):
		# 		outputs.append(generated_sent)
		# return outputs

	def augment(self, train, dev, return_dict=False):
		# text = """This is a good movie."""
		# paraphrases=self.get_paraphrases(text)
		# print(paraphrases)
		# fds
		# input_ids = self.tokenizer.encode(text, return_tensors="pt")
		# outputs = self.model.generate(input_ids)
		# print(self.tokenizer.decode(outputs[0]))


		#Create augmented data
		data_loader = DataLoader(
			dataset=train,
			batch_size=128,
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		self.model.eval()
		new_data={}

		for batch in data_loader:
			#For each batch, do it X time
			sents=batch['sentence']
			sentences=self.get_paraphrases(sents, n_predictions=self.argdict['split'])
			for i, lab in enumerate(batch['label']):
				for sent in sentences[i]:
					new_data[len(new_data)] = {'sentence': sent,
											  'label': int(lab.item()),
											  'input': train.tokenize(sent),
											  'augmented': True}
				# print(self.clean_sentence(sent), int(lab.item()))
		if return_dict:
			return new_data
		for j, item in new_data.items():
			len_data=len(train)
			# print(item)
			train.data[len_data]=item
		# train.return_pandas().to_csv(f'T5Quora_train_{self.argdict["dataset"]}.tsv', sep='\t')
		# fds
		return train




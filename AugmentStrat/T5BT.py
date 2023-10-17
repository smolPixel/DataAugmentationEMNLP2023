"""Paraphrase + BT augmentation. First load pretrained T5, fine-tune with parallele sentences back translated"""
import pandas as pd
from AugmentStrat.CBERT_strat.cbert_finetune import fine_tune_model
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import random
from tqdm import tqdm
import numpy as np



class T5ParaphraseBT():

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
		self.init_model()
		self.algo_is_trained = False

	def init_model(self):
		self.tokenizer = T5Tokenizer.from_pretrained('hetpandya/t5-base-tapaco')
		self.model = T5ForConditionalGeneration.from_pretrained("hetpandya/t5-base-tapaco").cuda()
		self.optimizer = AdamW(self.model.parameters(), lr=5e-5)


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

	def fine_tune_paraphraser(self, dataset):
		data_loader = DataLoader(
			dataset=dataset,
			batch_size=self.argdict['batch_size_algo'],
			shuffle=True,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		for i in range(self.argdict['num_epoch_algo']):
			losses_batch=[]
			for batch in data_loader:
				tokenized_input=self.tokenizer(batch['input'], truncation=True, padding=True, return_tensors='pt')
				tokenized_output=self.tokenizer(batch['output'], truncation=True, padding=True, return_tensors='pt')
				outputs=self.model(tokenized_input['input_ids'].cuda(),
								   decoder_attention_mask=tokenized_output['attention_mask'].cuda(),
								   labels=tokenized_output['input_ids'].cuda(),
								   attention_mask=tokenized_input['attention_mask'].cuda())
				loss=outputs['loss']
				losses_batch.append(loss.item())
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			print(i, np.mean(losses_batch))

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



	def augment(self, train, dev, return_dict=False):
		# text = """This is a good movie."""
		# paraphrases=self.get_paraphrases(text)
		# print(paraphrases)
		# input_ids = self.tokenizer.encode(text, return_tensors="pt")
		# outputs = self.model.generate(input_ids)
		# print(self.tokenizer.decode(outputs[0]))

		if not self.algo_is_trained:
			self.algo_is_trained=True

		#Two steps training, first, create synthetic data of paraphrases with BT

			#Create augmented data
			data_loader = DataLoader(
				dataset=train,
				batch_size=self.argdict['batch_size_algo'],
				shuffle=False,
				# num_workers=cpu_count(),
				pin_memory=torch.cuda.is_available()
			)

			self.model.eval()
			new_data=[]

			for batch in data_loader:
				#For each batch, do it X time
				sents=batch['sentence']
				sentences=self.back_translate(sents)
				for ss_og, ss_aug in zip(sents, sentences):
					new_data.append((ss_og, ss_aug))
			pre_training_data=Paraphrase_dataset(new_data, self.argdict)
			self.fine_tune_paraphraser(pre_training_data)
			# Create augmented data
		data_loader = DataLoader(
			dataset=train,
			batch_size=self.argdict['batch_size_generation'],
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)
		#Paraphrasing
		self.model.eval()
		new_data = {}

		for batch in data_loader:
			# For each batch, do it X time
			sents = batch['sentence']
			sentences = self.get_paraphrases(sents, n_predictions=self.argdict['split'])
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
			len_data = len(train)
			# print(item)
			train.data[len_data] = item
		# train.return_pandas().to_csv(f'T5BT_train_{self.argdict["dataset"]}.tsv', sep='\t')
		# fds
		return train

class Paraphrase_dataset(Dataset):
	def __init__(self, data, argdict):
		super().__init__()
		"""data: tsv of the data
		   tokenizer: tokenizer trained
		   vocabInput+Output: vocab trained on train"""
		self.data = {}
		self.argdict=argdict
		self.max_len = 0
		index=0
		# mask=len(argdict['categories'])
		for i, dat in enumerate(data):
			self.data[index] = {'input': dat[0], 'output':dat[1]}
			index+=1


	def __len__(self):
		return len(self.data)

	def __getitem__(self, item):
		return {
			'input': self.data[item]['input'],
			'output': self.data[item]['output']
		}



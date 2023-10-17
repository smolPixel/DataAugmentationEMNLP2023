"""EDA augmentation"""
import pandas as pd
from AugmentStrat.CBERT_strat.cbert_finetune import fine_tune_model
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW
from torch.utils.data import DataLoader
import torch
import random
from tqdm import tqdm

class mCBERT():

    def __init__(self, argdict):
        self.argdict=argdict
        self.init_model()
        self.algo_is_trained = False



    def init_model(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.list_tokens=["[CLS]"]
        for cat in self.argdict['categories']:
            tok=f'[{cat}]'
            self.list_tokens.append(tok)
            self.tokenizer.add_tokens(tok, special_tokens=True)
        # special_tokens = {'pos_token': '[POS]', 'neg_token': '[NEG]'}
        # num_add_toks = self.tokenizer.add_special_tokens(special_tokens)
        ## leveraging lastest bert module in Transformers to load pre-trained model (weights)
        self.model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased').cuda()
        self.model.resize_token_embeddings(len(self.tokenizer))

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
                encoding, labels, _, _ = self.mask_and_add_class(batch)
                # print(text_batch)
                # encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
                # print(encoding)
                input_ids = encoding['input_ids'].cuda()
                attention_mask = encoding['attention_mask'].cuda()
                outputs = self.model(input_ids, attention_mask, labels=labels.cuda())
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
        for i, token in enumerate(sent):
            if token in self.list_tokens:
                continue
            elif i!=len(sent)-1 and sent[i+1]=='[PAD]':
                break
            else:
                clean_sent.append(token)
        return " ".join(clean_sent)

    def augment(self, train, dev, return_dict=False):
        if not self.algo_is_trained:
            self.finetune(train)
            self.algo_is_trained=True

        #Create augmented data
        data_loader = DataLoader(
            dataset=train,
            batch_size=self.argdict['batch_size_algo'],
            shuffle=False,
            # num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        self.model.eval()
        new_data={}

        for batch in data_loader:
            #For each batch, do it X time
            for i in range(int(self.argdict['split'])):
                encoding, _, mask, unmasked_sentence = self.mask_and_add_class(batch)
                input_ids = encoding['input_ids'].cuda()
                attention_mask = encoding['attention_mask'].cuda()
                # print(text_batch)
                # encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
                outputs = self.model(input_ids, attention_mask)
                # print(outputs)
                try:
                    logits=outputs.logits
                except:
                    logits=outputs[0]
                outputs=torch.topk(torch.softmax(logits, dim=-1),k=2, dim=-1)[1]
                sentences=self.recreate_sentence(outputs, mask, input_ids, unmasked_sentence)
                # print(sentences)
                # print(batch['sentence'])
                # fds
                for sent, lab in zip(sentences, batch['label']):
                    sent=self.clean_sentence(sent)
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
        return train

        # Creating new dataset

    def recreate_sentence(self, bert_output, mask, bert_input, unmasked_sentence):
        """Recreate original sentence but with new, predicted words from bert"""
        #Dim bert output= num_exo, masked_tokens, top2
        # print(bert_output[:, ])
        mask=mask.cuda()
        bert_output=bert_output.cuda()
        unmasked_sentence=unmasked_sentence.cuda()
        #If first predicted token == unmasked sentence, take second
        bert_output_token=torch.where(unmasked_sentence[mask]==bert_output[:, :, 0][mask], bert_output[:, :, 1][mask], bert_output[:, :, 0][mask])
        bert_output=bert_output[:,:,0]
        bert_output[mask]=bert_output_token
        #This is a safety operation but should be unecessary
        sentences=torch.where(mask.cuda(), bert_output, bert_input)
        return self.tokenizer.batch_decode(sentences)

    def mask_and_add_class(self, batch):
        text_sentence=[]
        for sent, label in zip(batch['sentence'], batch['label']):
            classe=f"[{self.argdict['categories'][label.item()]}]"
            text_sentence.append(f"{classe} {sent}")

        input=self.tokenizer(text_sentence, return_tensors='pt', padding=True, truncation=True)
        unmasked=input['input_ids'].clone()
        labels=input['input_ids'].clone()
        mask=torch.zeros_like(input['input_ids']).float().uniform_() > 0.6
        #We dont want to predict words that are not in the sentence, use atttention mask to insure there are not masked
        maskformask=input['attention_mask']>0
        mask[~maskformask]=False
        #First token and class token can't be masked. PAD tokens can be masked since attention mask will prevent looking at them anyway
        mask[:, :2]=False
        input['input_ids'][mask]=self.tokenizer.mask_token_id
        # print(input)
        # print(self.tokenizer.batch_decode(input['input_ids']))
        labels=labels.masked_fill(~mask, -100)
        return input, labels, mask, unmasked

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

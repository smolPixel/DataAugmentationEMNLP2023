from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score
from data.DataProcessor import ds_DAControlled
import itertools

class Bert_Classifier():

    def __init__(self, argdict):
        self.argdict=argdict
        self.init_model()

        # print(self.model)


    def init_model(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/tmp/piedboef',
                                                           local_files_only=True)
        except:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/tmp/piedboef')
        try:
            self.argdict['output_hidden_state']
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                       num_labels=len(self.argdict['categories']),
                                                                       output_hidden_states=True,
                                                                       cache_dir='/Tmp',
                                                                       local_files_only=True).cuda()
        except:
            try:
                self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                           num_labels=len(self.argdict['categories']),
                                                                           cache_dir='/tmp/piedboef',
                                                                           local_files_only=True).cuda()
            except:
                self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                           num_labels=len(self.argdict['categories']),
                                                                           cache_dir='/tmp/piedboef').cuda()

        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
    def run_epoch(self, train, dev, test, return_grad=False):
        """Return grad returns the average grad for augmented and non augmented examples. """
        # train.return_pandas().to_csv("test.csv")
        self.model.train()
        bs=self.argdict['batch_size_classifier'] if not return_grad else 1
        data_loader = DataLoader(
            dataset=train,
            batch_size=bs,
            shuffle=True,
            # num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )


        grad_og=[]
        grad_aug=[]
        acc_og=[]
        acc_aug=[]

        pred_train = torch.zeros(len(train))
        Y_train = torch.zeros(len(train))
        start=0
        for i, batch in enumerate(data_loader):
            self.optimizer.zero_grad()

            text_batch=batch['sentence']
            encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].cuda()
            attention_mask = encoding['attention_mask'].cuda()
            # print(encoding)
            labels=batch['label'].cuda()
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            results = torch.argmax(torch.log_softmax(outputs[1], dim=1), dim=1)
            # print(outputs)
            # print(outputs)
            try:
                loss = outputs[0]
            except:
                loss=outputs.loss
            # print(loss)
            # print(outputs)
            loss.backward()

            if return_grad:
                total_norm=0
                for p in self.model.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                if batch['augmented']:
                    grad_aug.append(total_norm)
                    acc_aug.append(int(batch['label'].item()==results.item()))
                else:
                    grad_og.append(total_norm)
                    acc_og.append(int(batch['label'].item() == results.item()))




            self.optimizer.step()

            pred_train[start:start + bs] = results
            Y_train[start:start + bs] = batch['label']
            start = start + bs


        self.model.eval()
        #Test
        data_loader = DataLoader(
            dataset=dev,
            batch_size=64,
            shuffle=True,
            # num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        pred_dev = torch.zeros(len(dev))
        Y_dev = torch.zeros(len(dev))
        start=0
        for i, batch in enumerate(data_loader):
            with torch.no_grad():
                text_batch = batch['sentence']
                encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
                input_ids = encoding['input_ids'].cuda()
                attention_mask = encoding['attention_mask'].cuda()
                # print(encoding)
                labels = batch['label'].cuda()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                # print(outputs)
                results=torch.argmax(torch.log_softmax(outputs[0], dim=1), dim=1)
                pred_dev[start:start + 64] = results
                Y_dev[start:start+64]=batch['label']
                start=start+64
        #test
        data_loader = DataLoader(
            dataset=test,
            batch_size=64,
            shuffle=True,
            # num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        pred_test = torch.zeros(len(test))
        Y_test = torch.zeros(len(test))
        start = 0
        for i, batch in enumerate(data_loader):
            with torch.no_grad():
                text_batch = batch['sentence']
                encoding = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
                input_ids = encoding['input_ids'].cuda()
                attention_mask = encoding['attention_mask'].cuda()
                # print(encoding)
                labels = batch['label'].cuda()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                # print(outputs)
                results = torch.argmax(torch.log_softmax(outputs[0], dim=1), dim=1)
                pred_test[start:start + 64] = results
                Y_test[start:start + 64] = batch['label']
                start = start + 64
        # print(Y)
        # print(pred)
        # print(accuracy_score(Y, pred))

        # if return_grad:
        #     return accuracy_score(Y_train, pred_train), accuracy_score(Y_dev, pred_dev),accuracy_score(Y_test, pred_test), sum(grad_og)/len(grad_og), sum(grad_aug)/len(grad_og), \
        #            sum(acc_og)*100/len(acc_og), sum(acc_aug)*100/len(acc_aug)
        # else:
        if len(self.argdict['categories'])>2:
            ff=f1_score(Y_train, pred_train, average='macro'), f1_score(Y_dev, pred_dev, average='macro'),f1_score(Y_test, pred_test, average='macro')
        else:
            ff=accuracy_score(Y_train, pred_train), accuracy_score(Y_dev, pred_dev),accuracy_score(Y_test, pred_test)
        return ff



    def train_model(self, train, dev, test, type='dataLoader', generator=None, return_grad=False):

        if type=="pandas":
            train=Transformerdataset(train, split='train', labelled_only=True)
            dev=Transformerdataset(dev, split='dev', labelled_only=True)
        accuracies=[]
        grad_base_prog=[]
        grad_aug_prog=[]
        accuracies_base_prog=[]
        accuracies_aug_prog=[]
        for j in range(self.argdict['nb_epoch_classifier']):
            acc_train, acc_dev, acc_test=self.run_epoch(train, dev, test, return_grad=return_grad)
            # grad_base_prog.append(grad_base)
            # grad_aug_prog.append(grad_aug)
            # accuracies_base_prog.append(acc_base)
            # accuracies_aug_prog.append(acc_aug)
            print(f"Epoch {j}, accuracy on dev {(acc_train, acc_dev)} ")
                  # f"grad base {grad_base} grad aug {grad_aug}"
                  # f"Acc base {acc_base} Acc aug {acc_aug}")
            accuracies.append((acc_train, acc_dev, acc_test))
            # print(accuracies_aug_prog)
            # print(accuracies_base_prog)

        if return_grad:
            return accuracies, grad_base_prog, grad_aug_prog, accuracies_base_prog, accuracies_aug_prog
        else:
            return accuracies[-1]


    def train_batches(self, train, dev):
        self.model.train()
        bs = 8
        data_loader = DataLoader(
            dataset=train,
            batch_size=bs,
            shuffle=True,
            # num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        accs_dev=[]

        pred_train = torch.zeros(len(train))
        Y_train = torch.zeros(len(train))
        start = 0
        for i, batch in enumerate(data_loader):
            self.init_model()
            self.model.train()
            for j in range(self.argdict['nb_epoch_classifier']):
                self.optimizer.zero_grad()

                text_batch = batch['sentence']
                encoding = self.tokenizer(text_batch, max_length=self.argdict['max_seq_length'], return_tensors='pt', padding=True, truncation=True)
                input_ids = encoding['input_ids'].cuda()
                attention_mask = encoding['attention_mask'].cuda()
                # print(encoding)
                labels = batch['label'].cuda()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                results = torch.argmax(torch.log_softmax(outputs[1], dim=1), dim=1)
                # print(outputs)
                # print(outputs)
                try:
                    loss = outputs[0]
                except:
                    loss = outputs.loss
                # print(loss)
                # print(outputs)
                loss.backward()

                self.optimizer.step()
                if len(self.argdict['categories']) > 2:
                    ff = f1_score(batch['label'].cpu(), results.cpu(), average='macro')
                else:
                    ff = accuracy_score(batch['label'].cpu(), results.cpu())
                acc_train=ff


            self.model.eval()
            # Test
            data_loader_dev = DataLoader(
                dataset=dev,
                batch_size=bs,
                shuffle=True,
                # num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )
            pred_dev = torch.zeros(len(dev))
            Y_dev = torch.zeros(len(dev))
            start = 0
            for j, batch in enumerate(data_loader_dev):
                with torch.no_grad():
                    text_batch = batch['sentence']
                    encoding = self.tokenizer(text_batch, max_length=self.argdict['max_seq_length'], return_tensors='pt', padding=True, truncation=True)
                    input_ids = encoding['input_ids'].cuda()
                    attention_mask = encoding['attention_mask'].cuda()
                    # print(encoding)
                    labels = batch['label'].cuda()
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    # print(outputs)
                    results = torch.argmax(torch.log_softmax(outputs[0], dim=1), dim=1)
                    pred_dev[start:start + bs] = results
                    Y_dev[start:start + bs] = batch['label']
                    start = start + bs
            # if i==3:
            #     break
            if len(self.argdict['categories']) > 2:
                ff = f1_score(Y_dev, pred_dev, average='macro')
            else:
                ff = accuracy_score(Y_dev, pred_dev)
            acc_dev=accuracy_score(Y_dev, pred_dev)
            accs_dev.append(ff)
            print(f"Batch {i}, train acc: {acc_train}, dev acc: {acc_dev}")

        return acc_train, accs_dev
    #
    # def separate_good_bad(self, train, dev):
    #     self.model.train()
    #     bs = 8
    #     data_loader = DataLoader(
    #         dataset=train,
    #         batch_size=bs,
    #         shuffle=True,
    #         # num_workers=cpu_count(),
    #         pin_memory=torch.cuda.is_available()
    #     )
    #
    #     accs_dev = []
    #
    #     pred_train = torch.zeros(len(train))
    #     Y_train = torch.zeros(len(train))
    #     start = 0
    #     good_batches={}
    #     bad_batches={}
    #     for i, train_batch in enumerate(data_loader):
    #         self.init_model()
    #         self.model.train()
    #         for j in range(self.argdict['nb_epoch_classifier']):
    #             self.optimizer.zero_grad()
    #
    #             text_batch = train_batch['sentence']
    #             encoding = self.tokenizer(text_batch, max_length=self.argdict['max_seq_length'], return_tensors='pt', padding=True, truncation=True)
    #             input_ids = encoding['input_ids'].cuda()
    #             attention_mask = encoding['attention_mask'].cuda()
    #             # print(encoding)
    #             labels = train_batch['label'].cuda()
    #             outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
    #             results = torch.argmax(torch.log_softmax(outputs[1], dim=1), dim=1)
    #             # print(outputs)
    #             # print(outputs)
    #             try:
    #                 loss = outputs[0]
    #             except:
    #                 loss = outputs.loss
    #             # print(loss)
    #             # print(outputs)
    #             loss.backward()
    #
    #             self.optimizer.step()
    #             acc_train = accuracy_score(train_batch['label'].cpu(), results.cpu())
    #
    #         self.model.eval()
    #         # Test
    #         data_loader_dev = DataLoader(
    #             dataset=dev,
    #             batch_size=bs,
    #             shuffle=True,
    #             # num_workers=cpu_count(),
    #             pin_memory=torch.cuda.is_available()
    #         )
    #         pred_dev = torch.zeros(len(dev))
    #         Y_dev = torch.zeros(len(dev))
    #         start = 0
    #         for j, batch in enumerate(data_loader_dev):
    #             with torch.no_grad():
    #                 text_batch = batch['sentence']
    #                 encoding = self.tokenizer(text_batch, max_length=self.argdict['max_seq_length'], return_tensors='pt', padding=True, truncation=True)
    #                 input_ids = encoding['input_ids'].cuda()
    #                 attention_mask = encoding['attention_mask'].cuda()
    #                 # print(encoding)
    #                 labels = batch['label'].cuda()
    #                 outputs = self.model(input_ids, attention_mask=attention_mask)
    #                 # print(outputs)
    #                 results = torch.argmax(torch.log_softmax(outputs[0], dim=1), dim=1)
    #                 pred_dev[start:start + bs] = results
    #                 Y_dev[start:start + bs] = batch['label']
    #                 start = start + bs
    #         # if i==10:
    #         #     break
    #         acc_dev = accuracy_score(Y_dev, pred_dev)
    #         if acc_dev>self.argdict['cutoff']:
    #             for ind in train_batch['index']:
    #                 good_batches[len(good_batches)]=train.data[ind.item()]
    #         else:
    #             for ind in train_batch['index']:
    #                 bad_batches[len(bad_batches)] = train.data[ind.item()]
    #
    #         accs_dev.append(acc_dev)
    #         print(f"Batch {i}, train acc: {acc_train}, dev acc: {acc_dev}")
    #
    #
    #     # print(len(good_batches))
    #     # print(len(bad_batches))
    #     min_len=min(len(good_batches), len(bad_batches))
    #     # print(min_len)
    #     good_batches=dict(itertools.islice(good_batches.items(), min_len))
    #     bad_batches=dict(itertools.islice(bad_batches.items(), min_len))
    #     # bad_batches=bad_batches[:min_len]
    #     # print(len(good_batches))
    #     # print(len(bad_batches))
    #     # fds
    #
    #     good_ds=ds_DAControlled(data=good_batches, dataset_parent=train, from_dict=True)
    #     bad_ds=ds_DAControlled(data=bad_batches, dataset_parent=train, from_dict=True)
    #
    #     return good_ds, bad_ds
    #     # print(Y)
    #     # print(pred)
    #     # print(accuracy_score(Y, pred))
    #
    #     # if return_grad:
    #     #     return accuracy_score(Y_train, pred_train), accuracy_score(Y_dev, pred_dev), sum(grad_og) / len(
    #     #         grad_og), sum(grad_aug) / len(grad_og), \
    #     #            sum(acc_og) * 100 / len(acc_og), sum(acc_aug) * 100 / len(acc_aug)
    #     # else:
    #     #     return accuracy_score(Y_train, pred_train), accuracy_score(Y_dev, pred_dev), 0, 0, 0, 0

    def predict(self, dataset):
        ds = Transformerdataset(dataset, split='train', labelled_only=False)
        # Test
        data_loader = DataLoader(
            dataset=ds,
            batch_size=64,
            shuffle=False,
            # num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )
        Confidence = torch.zeros(len(ds))
        start = 0
        for i, batch in enumerate(data_loader):
            with torch.no_grad():
                text_batch = batch['sentence']
                encoding = self.tokenizer(text_batch, max_length=self.argdict['max_seq_length'], return_tensors='pt', padding=True, truncation=True)
                input_ids = encoding['input_ids'].cuda()
                attention_mask = encoding['attention_mask'].cuda()
                # print(encoding)
                labels = batch['label'].cuda()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=torch.zeros_like(labels))
                results = torch.max(torch.softmax(outputs['logits'], dim=1), dim=1)
                # print("FuckYouPytorch")
                # print(results)
                # print(results[0])
                # print(results[0][0])
                Confidence[start:start + 64] = results[0]
                start = start + 64

        # print(Confidence)
        # proba=np.max(np.array(self.algo.predict_proba(x_sent_tf)), axis=1)
        dataset['confidence']=Confidence
        return dataset

    def label(self, texts):
        #Predict the label of text
        with torch.no_grad():
            text_batch = texts
            encoding = self.tokenizer(text_batch, max_length=self.argdict['max_seq_length'], return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].cuda()
            attention_mask = encoding['attention_mask'].cuda()
            # print(encoding)
            # labels = batch['label'].cuda()
            outputs = self.model(input_ids, attention_mask=attention_mask)
            print(outputs)
            results = torch.max(torch.softmax(outputs['logits'], dim=1), dim=1)
            print(results)
            Confidence= results[0]

        # print(Confidence)
        # proba=np.max(np.array(self.algo.predict_proba(x_sent_tf)), axis=1)
        return dataset

    def get_logits(self, texts):
        with torch.no_grad():
            text_batch = texts
            encoding = self.tokenizer(text_batch, max_length=self.argdict['max_seq_length'], return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].cuda()
            attention_mask = encoding['attention_mask'].cuda()
            # print(encoding)
            # labels = batch['label'].cuda()
            outputs = self.model(input_ids, attention_mask=attention_mask)

        # print(Confidence)
        # proba=np.max(np.array(self.algo.predict_proba(x_sent_tf)), axis=1)

        # print(outputs)
        # fsd
        return outputs[0]

    def get_rep(self, texts):


        final=torch.zeros((len(texts), 768))

        with torch.no_grad():
            bs=64
            for index in range(0, len(texts), bs):
                text_batch = texts[index:index+bs]
                encoding = self.tokenizer(text_batch, max_length=self.argdict['max_seq_length'], return_tensors='pt', padding=True, truncation=True)
                input_ids = encoding['input_ids'].cuda()
                attention_mask = encoding['attention_mask'].cuda()
                # print(encoding)
                # labels = batch['label'].cuda()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                hidden_states=outputs[1]
                final[index:index+bs]=hidden_states[-1][:, 0, :]
        # print(final)
        # fds

        # print(Confidence)
        # proba=np.max(np.array(self.algo.predict_proba(x_sent_tf)), axis=1)

        # print(outputs)
        # fsd
        return final

    def get_grads(self, texts, labels):
        # train.return_pandas().to_csv("test.csv")
        self.model.eval()
        grads=[]
        start=0
        bs=1
        for i in range(0, len(texts), bs):
            text_batch = texts[i:i+bs]
            labels_batch=torch.Tensor(labels[i:i+bs]).cuda().long()
            encoding = self.tokenizer(text_batch, max_length=self.argdict['max_seq_length'], return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].cuda()
            attention_mask = encoding['attention_mask'].cuda()
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels_batch)
            results = torch.argmax(torch.log_softmax(outputs[1], dim=1), dim=1)
            # print(outputs)
            # print(outputs)
            try:
                loss = outputs[0]
            except:
                loss = outputs.loss
            # print(loss)
            # print(outputs)
            loss.backward()

            total_norm = 0
            for p in self.model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grads.append(total_norm)
            # print(grads)
            # fds
            # if batch['augmented']:
            #     grad_aug.append(total_norm)
            #     acc_aug.append(int(batch['label'].item() == results.item()))
            # else:
            #     grad_og.append(total_norm)
            #     acc_og.append(int(batch['label'].item() == results.item()))

            self.optimizer.zero_grad()

            # pred_train[start:start + bs] = results
            # Y_train[start:start + bs] = batch['label']
            start = start + bs
        self.optimizer.zero_grad()
        return grads

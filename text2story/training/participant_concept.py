# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
# ---

# author: evelin amorim
# tries to learn the concept of participant from the result of an SRL output
import os.path
from builtins import zip

import torch

from text2story.annotators import ALLENNLP
from text2story.readers import read_brat
from text2story.core.utils import bsearch_tuplelist

from transformers import BertTokenizer, BertModel
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader,RandomSampler

import os
from pathlib import Path

import spacy
from torch import nn

from sklearn.metrics import f1_score, confusion_matrix


class ParticipantsModel(nn.Module):
    def __init__(self, bert, hidden_size):
        super(ParticipantsModel, self).__init__()
        self.bert = bert

        # Define your layers
        self.conv1d_span = nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels=hidden_size, kernel_size=3)
        self.conv1d_sent = nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels=64, kernel_size=3,
                                     stride=1)

        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=2)
        self.sm = nn.Softmax(dim=1)

    def forward(self, sent, span):
        sent_embeddings = self.bert(sent)[0]
        span_embeddings = self.bert(span)[0]

        # Apply 1D convolution to span_embeddings
        span_features = self.conv1d_span(span_embeddings.transpose(1, 2))

        # Apply the convolutional layer
        sentence_features = self.conv1d_sent(
            sent_embeddings.transpose(1, 2))  # [batch_size, out_channels, sequence_length - kernel_size + 1]

        # Apply the LSTM layer
        lstm_output, _ = self.lstm(
            sentence_features.transpose(1, 2))  # [batch_size, sequence_length - kernel_size + 1, hidden_size]
        # Concatenate the span embeddings with the output of the LSTM layer
        combined_embeddings = torch.cat((span_embeddings.transpose(1, 2), lstm_output),
                                        dim=1)  # [batch_size, span_length + sequence_length - kernel_size + 1, hidden_size]

        # Apply the fully connected layer
        logits = self.fc(combined_embeddings)  # [batch_size, span_length + sequence_length - kernel_size + 1, 2]
        # print("Logits size after fc: ", logits.size())
        # Apply softmax activation if needed
        logits = self.sm(logits)
        # print("Logits size after sm: ", logits.size())
        logits = torch.max(logits, dim=1)[0]
        # print("Logits size after torch.max 1: ", logits.size())
        logits = torch.max(logits, dim=1)[0]
        # print("Logits size after torch.max 2: ", logits.size())

        return logits


class ParticipantConceptLearning:

    def __init__(self, lang):
        self.lang = lang
        if lang == "en":
            if not (spacy.util.is_package('en_core_web_lg')):
                spacy.cli.download('en_core_web_lg')

            self.language_model = spacy.load('en_core_web_lg')
        if lang == "pt":
            if not (spacy.util.is_package('pt_core_news_lg')):
                spacy.cli.download('pt_core_news_lg')
            self.language_model = spacy.load('pt_core_news_lg')

        # Load the pre-trained BERT tokenizer and model
        self.max_seq_len = 128
        self.tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased',
                                                       model_max_length=self.max_seq_len)
        self.bert_model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

        self.hidden_size = 128
        self.nepochs = 6
        self.learning_rate = 0.00005

        self.threshold = 0.5

        # Define the optimizer
        self.model = ParticipantsModel(self.bert_model, self.hidden_size)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        # Define the loss function
        self.loss_fn = nn.BCELoss()

    def read_file_lst(self, file_lst):
        doc_lst = []

        for f in file_lst:
            p = Path(f)
            reader = read_brat.ReadBrat()

            file_name = p.parent.joinpath(p.stem).absolute()

            doc = reader.process_file(str(file_name))
            doc_lst.append((doc, str(p.absolute())))
        return doc_lst

    def load_dataset(self, data_dir, type_data="train", split_data=None):
        if split_data is None:
            file_lst = []
            for dirpath, dirnames, filenames in os.walk(data_dir):
                for f in filenames:
                    if f.endswith(".txt"):
                        file_lst.append(os.path.join(dirpath, f))
        else:
            if type_data == "train":
                split_data_file = os.path.join(split_data, "train.txt")
            else:
                split_data_file = os.path.join(split_data, "test.txt")

            file_lst = open(split_data_file, "r").readlines()
            file_lst = [os.path.join(data_dir, f.replace("\n", "")) for f in file_lst]

        dataset = self.read_file_lst(file_lst)

        return dataset

    def get_actors(self, doc):

        actor_lst = []

        for tok in doc:
            for ann_type, ann_attr in tok.attr:
                if ann_type == "Participant":
                    actor_lst.append((tok.offset, tok.offset + len(tok.text)))

        return actor_lst

    def get_sent_lst(self, doc_text):
        sent_lst = []
        doc_lm = self.language_model(doc_text)

        for sent in doc_lm.sents:
            sent_lst.append(sent.text)

        return sent_lst

    def get_sent_text(self, span_offset, doc_lm):
        pass

    def get_sents_offset(self, sent_lst):
        sent_offset_lst = []
        offset_start = 0
        for sent in sent_lst:
            offset_end = offset_start + len(sent) - 1
            sent_offset_lst.append((offset_start, offset_end))
            offset_start = offset_end
        return sent_offset_lst

    def get_candidates_class(self, candidate_lst, actor_lst, sent_lst):
        """

        @param sent:
        @return: a list of words
        """
        sent_off_lst = self.get_sents_offset(sent_lst)
        y = []
        sent_id_lst = []

        count_positive = 0

        for candidate in candidate_lst:

            cand_offset = candidate[0]
            sent_id = bsearch_tuplelist(cand_offset[0], sent_off_lst)
            sent_id_lst.append(sent_id)

            pos = bsearch_tuplelist(cand_offset[0], actor_lst)
            if pos == -1:
                pos = bsearch_tuplelist(cand_offset[1], actor_lst)
                if pos == -1:
                    y.append(0)
                else:
                    y.append(1)
                    count_positive += 1
            else:
                y.append(1)

        # print("Positive instances: ", count_positive / len(y))
        return candidate_lst, y, sent_id_lst

    def build_batch_data(self, dataset):

        sent_text_lst = []
        candidate_text_lst = []
        y_lst = []

        for doc, file_name in dataset:
            # I need to get the class from the lusa annotations
            with open(file_name, "r") as fd:
                doc_text = fd.read()
                candidate_lst = ALLENNLP.extract_actors(self.lang, doc_text)

            sent_lst = self.get_sent_lst(doc_text)
            actor_lst = self.get_actors(doc)

            # TODO: verificar quantidade de candidatos e quantidade de cada classe
            candidate_lst, y, sent_id_lst = self.get_candidates_class(candidate_lst, actor_lst, sent_lst)
            y_lst += y

            sent_text_lst += [sent_lst[id] for id in sent_id_lst]
            idx = 0

            for ((start, end), _, _) in candidate_lst:
                cand = doc_text[start:end]
                candidate_text_lst.append(cand)
                idx += 1
        # print("-->",len(candidate_text_lst))
        tokens_sent = self.tokenizer.batch_encode_plus(sent_text_lst,
                                                       max_length=self.max_seq_len,
                                                       pad_to_max_length=True,
                                                       truncation=True,
                                                       return_token_type_ids=False)
        tokens_sent = torch.tensor(tokens_sent['input_ids'])

        tokens_span = self.tokenizer.batch_encode_plus(candidate_text_lst,
                                                       max_length=self.max_seq_len,
                                                       pad_to_max_length=True,
                                                       truncation=True,
                                                       return_token_type_ids=False)
        tokens_span = torch.tensor(tokens_span['input_ids'])

        labels = torch.tensor(y_lst)

        mydataset = TensorDataset(tokens_sent, labels, tokens_span)
        mydataloader = DataLoader(mydataset, batch_size=32, shuffle=True)

        return mydataloader

    def process_train(self, data_dir, device, split_dir=None):

        self.model.train()

        # load train dataset
        if split_dir is not None:
            dataset = self.load_dataset(data_dir, type_data="train", split_data=split_dir)
        else:
            dataset = self.load_dataset(data_dir)

        # load srl model
        ALLENNLP.load(self.lang)

        # apply srl and get possible candidates as participants
        dataloader = self.build_batch_data(dataset)
        self.model = self.model.to(device)

        # empty list to save model predictions
        total_preds = []
        total_labels = []

        total_loss = 0.0
        total_accuracy = 0.0

        torch.manual_seed(2023)

        # iterate over batches
        for step, (batch, labels, span) in enumerate(dataloader):
            if step % 10 == 0:
                print(" STEP ", step)

            sent = batch.to(device)
            span = span.to(device)

            # clear previously calculated gradients
            self.model.zero_grad()

            # get model predictions for the current batch
            # Forward pass
            outputs = self.model(sent, span)

            # Calculate the loss
            labels = labels.to(device).float()

            loss = self.loss_fn(outputs, labels)
            total_loss += loss.item()

            outputs = (outputs >= self.threshold).int()

            # Backpropagation
            loss.backward()

            # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # update paremeters
            self.optimizer.step()

            # model predictions are stored on GPU. So, push it to CPU
            outputs = outputs.detach().cpu().numpy()

            # outputs = np.argmax(outputs, axis=1)

            # append the model predictions
            total_preds += outputs.tolist()
            total_labels += labels.tolist()

        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        # print(f"Epoch {epoch+1}/{self.nepochs}, Average Loss: {avg_loss}")

        # predictions are in the form of (no. of batches, size of batch, no. of classes).
        # reshape the predictions in form of (number of samples, no. of classes)
        # total_preds  = np.concatenate(total_preds, axis=0)
        f1 = f1_score(total_labels, total_preds, average='weighted')
        m = confusion_matrix(total_labels, total_preds)

        print("\nCONFUSION MATRIX")
        print(m)
        print()

        # returns the loss and predictions
        return avg_loss, f1
        # return 0,0
        # function for evaluating the model

    def evaluate(self, data_dir, device, split_dir=None):

        print("\nEvaluating...")

        # deactivate dropout layers
        self.model.eval()
        # Define the loss function
        loss_fn = nn.BCELoss()

        total_loss, total_accuracy = 0, 0

        # empty list to save the model predictions
        total_preds = []
        total_labels = []

        # load validation dataset
        if split_dir is not None:
            dataset = self.load_dataset(data_dir, type_data="test", split_data=split_dir)
        else:
            dataset = self.load_dataset(data_dir)

        # apply srl and get possible candidates as participants
        dataloader = self.build_batch_data(dataset)
        self.model = self.model.to(device)

        # iterate over batches
        for step, (batch, labels, span) in enumerate(dataloader):

            # Progress update every 10 batches.
            if step % 5 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                # elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

            # push the batch to gpu
            sent = batch.to(device)
            span = span.to(device)

            # deactivate autograd
            with torch.no_grad():

                # model predictions
                preds = self.model(sent, span)

                # compute the validation loss between actual and predicted values
                labels = labels.to(device).float()
                loss = self.loss_fn(preds, labels)

                total_loss = total_loss + loss.item()

                preds = (preds >= self.threshold).int()
                preds = preds.detach().cpu().numpy()
                # preds = np.argmax(preds, axis=1)
                total_preds += list(preds)
                total_labels += labels.tolist()

        # compute the validation loss of the epoch
        avg_loss = total_loss / len(dataloader)

        # reshape the predictions in form of (number of samples, no. of classes)
        # total_preds  = np.concatenate(total_preds, axis=0)

        f1 = f1_score(total_labels, total_preds, average='weighted')
        m = confusion_matrix(total_labels, total_preds)

        print("\nCONFUSION MATRIX")
        print(m)
        print()

        return avg_loss, f1





"""
Weak labeling performed using SRL and Neural Models

For the Participant extraction the training method is in the Class training/participant_concept.ParticipantsModel

@author: Evelin Amorim
"""
import torch
import os
import gdown

from transformers import BertTokenizer, BertModel

from text2story.annotators import ALLENNLP
from text2story.core.exceptions import InvalidLanguage

from text2story.training.participant_concept import  ParticipantsModel

from nltk import sent_tokenize

pipeline = {}
device = "cpu"
max_seq_len = 128
threshold = 0.5


def download_participant_pt_model(participant_pt_model):
    print("Downloading %s. It can take some time..." % participant_pt_model)

    url = "https://drive.google.com/uc?id=1mnaTfOcKyoKFclhA430VqvIsheLMeLdB&export=download"
    gdown.download(url, participant_pt_model, quiet=False)

def load_pt():
    ALLENNLP.load("pt")

    current_path = os.path.dirname(os.path.abspath(__file__))

    participant_pt_model_path = os.path.join(current_path, "cache")
    if not (os.path.exists(participant_pt_model_path)):
        os.mkdir(participant_pt_model_path)

    participant_pt_model = os.path.join(participant_pt_model_path, "participant_saved_weights.pt")
    pipeline["participants_model_file"] = participant_pt_model

    if not (os.path.exists(participant_pt_model)):
        download_participant_pt_model(participant_pt_model)

    pipeline["tokenizer"] = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', \
                                                         model_max_length=128)
    pipeline["bert_model"] = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

    path_loader = torch.load(participant_pt_model, map_location=torch.device(device))
    participant_model = ParticipantsModel(pipeline["bert_model"], 128)
    participant_model.load_state_dict(path_loader)
    pipeline["participants_model"] = participant_model

    #pipeline["tag"] = model.get("id_map")


def load(lang):
    if lang == "pt":
        load_pt()
    else:
        raise InvalidLanguage


def extract_actors(lang, text):

    sent_lst = sent_tokenize(text)

    for sent in sent_lst:
        print(sent)
        tokens = pipeline["tokenizer"].tokenize(sent)

        tokens = tokens[:max_seq_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        input_ids = pipeline["tokenizer"].convert_tokens_to_ids(tokens)
        input_ids = input_ids + [0] * (max_seq_len - len(input_ids))
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to(device)

        # extrair srl da sentenca e converter para input ids tbm
        candidate_lst = ALLENNLP.extract_actors(lang, sent)

        # cada span tem uma sentenca, entao tenho que reproduzir isso!
        span_text = []
        for ((start,end), _, _) in candidate_lst:
            span_text = sent[start:end]
            tokens_span = pipeline["tokenizer"].tokenize(span_text)

            input_ids_span = pipeline["tokenizer"].convert_tokens_to_ids(tokens_span)
            input_ids_span = input_ids_span + [0] * (max_seq_len - len(input_ids_span))
            input_ids_span = torch.tensor(input_ids_span).unsqueeze(0)
            input_ids_span = input_ids.to(device)

            preds = pipeline["participants_model"](input_ids, input_ids_span)
            preds = (preds >= threshold).int()

            print(preds)
            print("SPAN TEXT", span_text)
            #print(len(preds), len())

            print()

import os
import pandas as pd
import shutil
from pathlib import Path

import string

# adding root directory
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../../')

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn_crfsuite.utils import flatten
import sklearn_crfsuite
from sklearn_crfsuite import scorers
#from sklearn_crfsuite import metrics

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import nlpnet


import joblib

from readers.read_brat import ReadBrat
from readers.token_corpus import TokenCorpus


class TokSRL:
    def __init__(self):
        self.gov_verb = None
        self.arg = None
        self.tok = None

class FeatureExtractor:

    def __init__(self):

        self.srl_tagger = nlpnet.SRLTagger("srl-pt/",language="pt")
        # (name, transformer, columns)
        self.preprocessor = ColumnTransformer(
                transformers=[ 
                    ('token', TfidfVectorizer(),"token"), #TfidfVectorizer accepts column name only between quotes
                    ('lemma', TfidfVectorizer(),"lemma"), #TfidfVectorizer accepts column name only between quotes
                    ('head', TfidfVectorizer(),"head"), #TfidfVectorizer accepts column name only between quotes
                    ('head_lemma', TfidfVectorizer(),"head_lemma"), #TfidfVectorizer accepts column name only between quotes
                    ('gov_verb', TfidfVectorizer(),"gov_verb"), #TfidfVectorizer accepts column name only between quotes
                    ('category', OneHotEncoder(), ["head_pos","pos"]),
                    ],
                    remainder="drop"
                )

        self.pipe = Pipeline(
                steps=[
                    ('preprocessor', self.preprocessor),
                    ('classifier', LogisticRegression()),
                    ],
                )

    def get_verb_pos(self, verb, tok_lst, idx_start):
        """
        It get the verb index in a given token list. The first occurrence,
        starting from idx_start index position

        @param string: the verb
        @param [string]: a list of tokens 
        @param int: the starting point to start the search for the verb

        @return int: the verb index in the token list
        """

        idx = idx_start
        while idx < len(tok_lst):
            if tok_lst[idx] == verb:
                break
            idx = idx + 1

        return idx

    def set_role_token(self, tokcorpus_lst, idx_verb, args):
        """
        It sets the role position for each token in the tokcorpus_lst list

        @param [TokenCorpus]: a list of TokenCorpus objects
        @param int: the index of the current governing verb of the given argument
                  structure
        @param dict: The dictionary that stores the argument structure of an SRL 
                    labeling returned by the nlpnet API

        @return [TokenCorpus]: the list of TokenCorpus objects updated with 
                    governing verb and srl role
        """

        # TODO 1: It is possible implement a more efficient method?
        # TODO 2: Perform a more complete test to see what examples does not work
        # here

        for role in args:
            for tok in args[role]:
                # talvez começar analise APOS idx_verb e, tokcorpus_lst!
                # nao pode, pois tem argumentos antes de idx_verb
                # Verbo mais proximo? Como saber qual eh o mais proximo?
                #  guardar posicao do gov_verb
                for idx_corpus, tokcorpus in enumerate(tokcorpus_lst):
                    if tok == tokcorpus.text:

                        # if still there is no role
                        if tokcorpus.srl is None:
                            tokcorpus.gov_verb = tokcorpus_lst[idx_verb].text
                            tokcorpus.gov_verb_idx = idx_verb
                            tokcorpus.srl = role
                            break
                        else:
                            # if there is role, but there is a more specific governing verb
                            # then assign a new role and a new governing verb
                            if tokcorpus.gov_verb_idx is not None:
                                # the nearest verb is more probable to be the gpverning verb
                                # so, this kind a heuristic according to the 
                                # nlpnet output
                                dist_verb = abs(idx_corpus - idx_verb)
                                dist_gov_verb = abs(idx_corpus - tokcorpus.gov_verb_idx)
                                if tokcorpus.gov_verb != tokcorpus_lst[idx_verb].text and\
                                        dist_verb <= dist_gov_verb:
                                    tokcorpus.gov_verb = tokcorpus_lst[idx_verb].text
                                    tokcorpus.srl = role

        #for tokcorpus in tokcorpus_lst:
        #    print(tokcorpus.text, tokcorpus.srl, tokcorpus.gov_verb)
        return tokcorpus_lst

    def get_srl_tags(self, doc):


        tok_lst = [tok.text for tok in doc]
        tags = self.srl_tagger.tag_tokens(tok_lst)

        all_toks_lst = []

        # tracking position of the verbs in the argument 
        # structures
        last_verb_idx = 0
        new_arg_lst = []
        for (verb, args) in tags.arg_structures:
            idx_verb = self.get_verb_pos(verb, tags.tokens, last_verb_idx)
            last_verb_idx = idx_verb
            new_arg_lst.append((verb, args, idx_verb))

        # mapping roles to tokens, using the governing verb positio
        tokcorpus_lst = [TokenCorpus(tok) for tok in tags.tokens]
        for verb, args, idx_verb in new_arg_lst:
            tokcorpus_lst = self.set_role_token(tokcorpus_lst, idx_verb, args)
        all_toks_lst = all_toks_lst + tokcorpus_lst
        return all_toks_lst

    def doc2feature(self, doc, file_name):

        data_= {"token":[],"lemma":[],"head":[],"head_lemma":[],\
                "head_pos":[],"pos":[]}

        y_ = []

        srl_tags = self.get_srl_tags(doc)

        for tok in doc:

            data_["token"] = tok.text
            data_["lemma"] = tok.lemma
            data_["head"] = tok.head
            data_["head_lemma"] = tok.head_lemma
            data_["gov_verb"] = tok.gov_verb

            data_["head_pos"] = tok.head_pos
            data_["pos"] = tok.pos

            if tok.ann == "Event":
                y_.append("I")
            else:
                y_.append("O")

        return data_,y_


    def doc2featureCRF(self, doc, file_name):

        data_doc = []
        y_doc = []

        sent_lst = []
        sent = []
        for tok in doc:
            if  tok.text != ".":
                sent.append(tok)
            else:
                sent.append(tok)
                sent_lst.append(sent)
                sent = []

        if sent != []:
            sent_lst.append(sent)

        srl_toks = []
        for sent in sent_lst:
            srl_toks += self.get_srl_tags(sent)


        last_label = "O"
        for idx, tok in enumerate(doc):
            
            data_= {}

            data_["token"] = tok.text
            data_["lemma"] = tok.lemma
            data_["head"] = tok.head
            data_["head_lemma"] = tok.head_lemma
            #data_["gov_verb"] = tok.gov_verb

            data_["head_pos"] = tok.head_pos
            data_["pos"] = tok.pos

            if tok.ann == "Event" and \
                    (last_label == "I-EVENT" or last_label == "B-EVENT"):
                y_doc.append("I-EVENT")
                last_label = "I-EVENT"
            elif tok.ann == "Event" and last_label == "O":
                y_doc.append("B-EVENT")
                last_label = "B-EVENT"
            else:
                y_doc.append("O")
                last_label = "O"

            data_doc.append(data_)

        return data_doc, y_doc

    def _get_events_iob(self, doc, class_tokens):
        """
        Get a list of events from the labels of a iob labeling list
        """

        last_class = "O"
        count_event = 1
        current_event = {}
        event_lst = []

        symbols = string.punctuation + "\”\"\“"

        # get events

        for idx, tok in enumerate(doc):

            if class_tokens[idx] == "B-EVENT":

                current_event = {"id":count_event, "start":tok.offset, \
                        "end":tok.offset + len(tok.text),\
                        "value":tok.text}

                last_class = "B-EVENT"
                count_event += 1

                #print("T%d Event %d %d %s" % \
                #        (count_event, tok.offset, tok.offset + len(tok.text), tok.text))
            elif class_tokens[idx] == "I-EVENT":

                if last_class == "B-EVENT" or last_class == "I-EVENT":

                    id_ = current_event["id"]
                    start_ = current_event["start"]
                    end_ = tok.offset + len(tok.text)
                    if tok.text not in symbols:
                        value_ = current_event["value"] + " " + tok.text
                    else:
                        value_ = current_event["value"] + tok.text

                    current_event = {"id":id_,"start":start_,"end":end_, "value":value_}
                    last_class = "I-EVENT"
                else:
                    current_event = {"id":count_event, "start":tok.offset, \
                        "end":tok.offset + len(tok.text),\
                        "value":tok.text}

                    last_class = "I-EVENT"
                    count_event += 1
            else:
                if last_class == "B-EVENT" or last_class == "I-EVENT":
                    event_lst.append(current_event)
                    current_event = {}
                    last_class = "O"

        return event_lst

    def write2brat_file(self, doc, class_tokens, output_file):

        event_lst = self._get_events_iob(doc, class_tokens)

        with open(output_file, "w") as fd_output:
            for event in event_lst:
                fd_output.write("T%d\tEvent %d %d\t%s\n" % \
                        (event["id"],\
                        event["start"],\
                        event["end"],\
                        event["value"]))


    def write2brat(self, data_tokens, file_txt_lst, y_pred):

        # TODO: preciso do offset aqui tbm
        # coloquei offset.

        if not(os.path.exists("brat_output")):
            os.mkdir("brat_output")

        for idx, doc in enumerate(data_tokens):

            # copy the txt file
            shutil.copy2(file_txt_lst[idx], "brat_output")

            # write the annotation file
            ann_file = os.path.join("brat_output",Path(file_txt_lst[idx]).stem + ".ann")


            self.write2brat_file(doc, y_pred[idx], ann_file)

        

    def process_test(self, data_dir,model="crf",write_output=False):

        reader = ReadBrat()
        data_tokens = reader.process(data_dir)

        data_lst = []
        data_dct = {}
        y = []
        idx = 0

        for doc in data_tokens:
            if model == "crf":
                data_doc, y_doc = self.doc2featureCRF(doc, reader.file_lst[idx])
                data_lst.append(data_doc)
                y.append(y_doc)
            else:
                data_doc, y_doc = self.doc2feature(doc, reader.file_lst[idx])
                data_dct = {**data_doc, **data_dct}
            idx += 1

        clf = joblib.load("crf.joblib")
        y_pred = clf.predict(data_lst)

        print(metrics.classification_report(flatten(y), flatten(y_pred)))

        if write_output:
            self.write2brat(data_tokens, reader.file_lst, y_pred)

    def process(self, data_dir, model="crf"):
        """
        Read a data_dir (.ann and their .txt files)
        and returns a matrix of features, and labels

        @param string: data directory

        @return (numpy.array,numpy.array): Matrix of features and vector 
        of labels
        """

        # read file names .ann and .txt
        reader = ReadBrat()
        data_tokens = reader.process(data_dir)

        data_lst = []
        data_dct = {}
        y = []
        idx = 0
        for doc in data_tokens:
            if model == "crf":
                data_doc, y_doc = self.doc2featureCRF(doc, reader.file_lst[idx])
                data_lst.append(data_doc)
                y.append(y_doc)
            else:
                data_doc, y_doc = self.doc2feature(doc, reader.file_lst[idx])
                data_dct = {**data_doc, **data_dct}
            idx += 1


        if model != "crf":
            df = pd.DataFrame(data_dct)
            if os.path.exists("preprocessor.joblib"):
                print("Warning: Loading a TF-IDF local model called preprocessor.joblib.")
                self.preprocessor = joblib.load("preprocessor.joblib")
            else:
                print("Warning: Creating a TF-IDF new model from the given data.")
                self.preprocessor.fit(df)
                joblib.dump(self.preprocessor,"preprocessor.joblib")

            df_trans = self.preprocessor.transform(df)
            # TODO: finish model
        else:
            clf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                    c1=0.1, c2=0.1, max_iterations=100, 
                    all_possible_transitions=True)
            df_trans = data_lst

        clf.fit(df_trans, y)
        joblib.dump(clf, "crf.joblib")

if __name__ == "__main__":

    data_dir = "data/train/"
    test_dir = "data/test/"
    ftr_extractor = FeatureExtractor()

    ftr_extractor.process(data_dir)
    ftr_extractor.process_test(test_dir, write_output=True)

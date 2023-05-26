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
#import nlpnet

#from srl_bert_pt import tag_srl
import subprocess

import joblib

from readers.read_brat import ReadBrat
from readers.token_corpus import TokenCorpus


class TokSRL:
    def __init__(self):
        self.gov_verb = None
        self.arg = None
        self.tok = None

class EventModel:

    def __init__(self):

        current_path = os.path.dirname(__file__)
        srl_path = "srl_bert_pt/srl-enpt_xlmr-large/"
        full_srl_path = os.path.join(current_path, srl_path)

        subprocess.call("./srl_download.sh")
        #self.srl_tagger = tag_srl.TagSRL(full_srl_path)

        self.reader = ReadBrat()

        #self.srl_tagger = nlpnet.SRLTagger("srl-pt/",language="pt")
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
        tags = self.srl_tagger.tag(tok_lst)

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

    def get_idx_token_srl(self, file_name):
        """
        Get the SRL tag of a given token

        @param string: the file to be tagged

        @return [(string, string, int)]: return a list of tuples that stores 
        (word, SRL tag, index in the text)
        """

        with open(file_name, "r") as fd:
            txt_ = fd.read()
            doc = self.reader.nlp(txt_)

            current_offset = 0
            token_tags_lst = []

            for sent in doc.sents:
                # get srl tags by sentence
                result = self.srl_tagger.tag(sent.text)

                # if there is verb in the sentence, 
                # then a tagged result is possible
                if "verbs" in result:
                    srl_tags = [(t, result["verbs"][0]["verb"]) for t in result["verbs"][0]["tags"]]
                    gov_verb = result["verbs"][0]["verb"]


                    # if there is more than one verb, these loops try to
                    # merge the srl tags
                    for i in range(1,len(result["verbs"])):

                        for j in range(len(result["verbs"][i]["tags"])):
                            if result["verbs"][i]["tags"][j] != "O":
                                tag = result["verbs"][i]["tags"][j]
                                gov_verb = result["verbs"][i]["verb"]
                                srl_tags[j] = (tag, gov_verb)

                    words = result["words"]
                else:
                    srl_tags = None
                    words = result
                    gov_verb = "None"

                # grouping the words and tags,
                # and in the process find in the text 
                # the character offset
                for idx, w in enumerate(words):
                    # character offset of the word w
                    offset_w = txt_.find(w, current_offset)
 
                    # updating global char offset
                    current_offset += len(w)

                    if srl_tags is not None:
                        token_tags_lst.append((w, srl_tags[idx], offset_w))
                    else:
                        token_tags_lst.append((w, ("O", gov_verb), offset_w))

        return token_tags_lst

    def get_srl_tags2(self, doc, file_name):
        """
        Since the srl tagger and the reader (Current ReadBrat) own different 
        tokenizers. This method present an heuristic to assign srl tags to doc.
        """

        token_tags_lst = self.get_idx_token_srl(file_name)

        i = 0

        while i < len(doc):
            # stores the last distance between the token from SRL
            # and the token from doc. So, it starts with None value, since
            # there is still no distance computed
            dist_old = None
            j = i
            while j < len(token_tags_lst):
                offset1 = token_tags_lst[j][2]
                offset2 = doc[i].offset

                label_srl , gov_verb = token_tags_lst[j][1]

                # the concept of distance is the difference between 
                # the offset of the token from the SRL tokenizer and
                # the offset of the token from the Reader tokenizer.
                # the smallets the diffrence, the nearest the words, and 
                # there is high probability that they are the same tokens.
                # So, the method assigns the SRL tag to the doc token.
                dist = abs(offset1 - offset2)
                if dist_old is None:
                    dist_old = dist
                    doc[i].srl = label_srl
                    doc[i].gov_verb = gov_verb
                else:
                    if dist < dist_old:
                        dist_old = dist
                        doc[i].srl = label_srl
                        doc[i].gov_verb = gov_verb
                j += 1
                
            i += 1


    def doc2feature(self, doc, file_name):
        """
        Extract the features of a text document for a CRF algorithm
        """

        data_doc = []
        y_doc = []

        #self.get_srl_tags2(doc, file_name)

        last_label = "O"
        for idx, tok in enumerate(doc):
            
            data_= {}
            # print(tok.text, "-",res2["words"][idx])

            data_["token"] = tok.text
            data_["lemma"] = tok.lemma
            data_["head"] = tok.head
            data_["head_lemma"] = tok.head_lemma
            data_["gov_verb"] = tok.gov_verb if tok.gov_verb is not None else "None"
            data_["srl"] = tok.srl if tok.srl is not None else "O"

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

        

    def process_test(self, data_dir,model="crf",write_output=False):

        data_tokens = self.reader.process(data_dir)

        data_lst = []
        data_dct = {}
        y = []
        idx = 0

        for doc in data_tokens:
            data_doc, y_doc = self.doc2feature(doc, self.reader.file_lst[idx])
            data_lst.append(data_doc)
            y.append(y_doc)
            idx += 1

        clf = joblib.load("crf_v2.1.joblib")
        y_pred = clf.predict(data_lst)

        print(metrics.classification_report(flatten(y), flatten(y_pred)))


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
            data_doc, y_doc = self.doc2feature(doc, reader.file_lst[idx])
            data_lst.append(data_doc)
            y.append(y_doc)
            idx += 1


        clf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                    c1=0.1, c2=0.1, max_iterations=100, 
                    all_possible_transitions=True)
        df_trans = data_lst

        clf.fit(df_trans, y)
        joblib.dump(clf, "crf_v2.1.joblib")

if __name__ == "__main__":

    data_dir = "data/train/"
    test_dir = "data/test/"
    model = EventModel()

    model.process(data_dir)
    model.process_test(test_dir)

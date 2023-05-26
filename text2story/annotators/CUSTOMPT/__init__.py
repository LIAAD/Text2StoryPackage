"""
    A custom annotator for events in portuguese language

    The model is a random forest based on the lusa news labeled data
"""

# adding root directory
import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../../')

import spacy
import joblib
import numpy as np
import pandas as pd

from text2story.readers.read_brat import ReadBrat

import os
# adding root directory
import sys
import string

sys.path.insert(0, '..')
sys.path.insert(0, '../../')

from text2story.readers.token_corpus import TokenCorpus

from sklearn.ensemble import GradientBoostingClassifier

pipeline = {}

# mapping of pos tag to a integer number
pos2idx = {"<S>": 1, "PROPN": 2, "PUNCT": 3, "NUM": 4, \
           "ADP": 5, "SPACE": 6, "DET": 7, "NOUN": 8, \
           "CCONJ": 9, "ADJ": 10, "VERB": 11, "ADV": 12, \
           "SCONJ": 13, "AUX": 14, "PRON": 15, "SYM": 16, \
           "X": 17}

# the size of the window to the
# features of the model, currently the model was trained with 2
WINDOW_SIZE = 2


def load(lang):

    if lang == "pt":
        current_path = os.path.dirname(__file__)
        model_name = "srl-pt_bertimbau-base"
        full_srl_path = os.path.join(current_path, "srl_bert_pt")
        full_srl_path = os.path.join(full_srl_path, model_name)


        pipeline["event_tagger"] = joblib.load(os.path.join(current_path, "crf_v2.1.joblib"))
        pipeline["pos_tagger"] = spacy.load("pt_core_news_lg")
    #pipeline["srl_tagger"] = tag_srl.TagSRL(full_srl_path)


def _get_data(idx_lst, all_ftrs):
    """
    it gets the data in all_ftr lst that has the index in the idx_lst

    @param [integer]: a list of index
    @param [[float]]: a list of lists of features
    """

    X = []
    y = []
    for i in idx_lst:
        for data in all_ftrs[i]:
            y.append(data[-1])

            # concat all the tokens in only one vector
            X.append(data[:-1])

    return np.asarray(X), y


def _get_pos_ftr(idx, doc_lst):
    """
    mapping of pos tagging to integer

    @param integer: the index of the token to get the pos tag code
    @param [(string,string)]: a list of tuples of the format (token, pos_tag)

    @return integer: a integer code that represents a pos tag
    """

    if idx < 0 or idx >= len(doc_lst):
        return pos2idx["<S>"]
    else:
        if len(doc_lst[idx]) > 2:
            (tok, pos, ann) = doc_lst[idx]
        else:
            (tok, pos) = doc_lst[idx]

        return pos2idx[pos]  # pos tag


def _extract_features_train(doc):
    """
    build features for each sentence tokenized by spacy

    @param document: a sentence as document type (spacy)

    @return np.array: an array of pos tagging features of a given text
    """
    ftrs = []
    idx = 0
    window = WINDOW_SIZE

    tok_lst = [(tok.text, tok.pos, tok.ann) for tok in doc]

    for (tok, pos, ann) in tok_lst:
        tok_ftrs = []

        for i in range(idx - window, idx + window + 1):
            tok_ftrs.append(_get_pos_ftr(i, tok_lst))

        if ann == 'Event':
            tok_ftrs.append("I")
        else:
            tok_ftrs.append("O")

        idx += 1
        ftrs.append(tok_ftrs)

    return np.array(ftrs)


def _extract_features(doc):
    """
    build features for each sentence tokenized by spacy

    @param document: a sentence as document type (spacy)

    @return np.array: an array of pos tagging features of a given text
    """
    ftrs = []
    idx = 0
    window = WINDOW_SIZE

    tok_lst = [(tok.text, tok.pos_) for tok in doc]

    for (tok, pos) in tok_lst:
        tok_ftrs = []

        for i in range(idx - window, idx + window + 1):
            tok_ftrs.append(_get_pos_ftr(i, tok_lst))

        idx += 1
        ftrs.append(tok_ftrs)

    return np.array(ftrs)


def execute_train(data_dir, reader, output_model="gb_custom_event_pt.joblib"):
    """
    Method to train a model to classify events, given a dataset

    @param string: path to the train data
    @param ReadBrat: a reader of the corpus
    @param string: the name of the podel to persist

    @return None
    """

    # read file names .ann and .txt
    reader = ReadBrat()
    data_tokens = reader.process(data_dir)

    data_lst = []
    data_dct = {}
    y = []
    idx = 0
    for doc in data_tokens:
        data_doc, y_doc = doc2featureCRF(doc, reader, file_name=reader.file_lst[idx])
        data_lst.append(data_doc)
        y.append(y_doc)
        idx += 1


    clf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                c1=0.1, c2=0.1, max_iterations=100, 
                all_possible_transitions=True)
    df_trans = data_lst

    clf.fit(df_trans, y)
    joblib.dump(clf, output_model)


def _get_verb_pos(verb, tok_lst, idx_start):
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


def _set_role_token(tokcorpus_lst, idx_verb, args):
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
                            if tokcorpus.gov_verb != tokcorpus_lst[idx_verb].text and \
                                    dist_verb <= dist_gov_verb:
                                tokcorpus.gov_verb = tokcorpus_lst[idx_verb].text
                                tokcorpus.srl = role

    # for tokcorpus in tokcorpus_lst:
    #    print(tokcorpus.text, tokcorpus.srl, tokcorpus.gov_verb)
    return tokcorpus_lst


def get_idx_token_srl(reader, txt_):
    """
    Get the SRL tag of a given token

    @param string: the file to be tagged

    @return [(string, string, int)]: return a list of tuples that stores 
    (word, SRL tag, index in the text)
    """

    doc = reader.nlp(txt_)

    current_offset = 0
    token_tags_lst = []

    for sent in doc.sents:
        # get srl tags by sentence
        result = pipeline["srl_tagger"].tag(sent.text)

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

def set_srl_tags(doc, reader, txt):
    """
    Since the srl tagger and the reader (Current ReadBrat) own different 
    tokenizers. This method present an heuristic to assign srl tags to doc.
    """

    token_tags_lst = get_idx_token_srl(reader, txt)

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

def doc2featureCRF(doc, reader, file_name=None, text=None):

    data_doc = []
    y_doc = []

    if text is None:
        if file_name is None:
            raise Exception("CustomPT Model: Unknown source of the text. \
                    Please, provide a raw text for labeling.")
        else:
            with open(file_name, "r") as fd:
                text = fd.read()

    # TODO: SRL portuguese requires a strong hardware
    #set_srl_tags(doc, reader, text)

    idx = 0
    last_label = "O"
    sent = []

    for idx, tok in enumerate(doc):
        
        data_= {}
        # print(tok.text, "-",res2["words"][idx])

        data_["token"] = tok.text
        data_["head"] = tok.head.text

        # if it is Token object
        if hasattr(tok, "head_lemma"):
            data_["lemma"] = tok.lemma_
            data_["head_lemma"] = tok.head_lemma
            data_["head_pos"] = tok.head_pos
            data_["gov_verb"] = tok.gov_verb if tok.gov_verb is not None else "None"
            data_["srl"] = tok.srl if tok.srl is not None else "O"
            data_["pos"] = tok.pos
        else:
            # else it is a Spacy.Token object
            data_["lemma"] = tok.lemma_
            data_["head_lemma"] = tok.head.lemma_
            data_["head_pos"] = tok.head.pos_
            data_["gov_verb"] = "None"
            data_["srl"] = tok.dep_
            data_["pos"] = tok.pos_


        if hasattr(tok, "ann"):
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

        if hasattr(tok,"is_sent_start"):
            if tok.is_sent_start:
                if sent!= []:
                    data_doc.append(sent)
                sent = [data_]
            else:
                sent.append(data_)
        else:
            data_doc.append(data_)

    if sent != []:
        data_doc.append(sent)

    return data_doc, y_doc



def extract_events(lang, text):
    """
    Main function that applies the custom tagger to extract event entities from each sentence.

    @param lang: The language of the text
    @param text: The full text to be annotated

    @return: Pandas DataFrame with every event entity and their character span
    """
    # 1. use spacy to tokenize and to get pos tags
    doc = pipeline["pos_tagger"]((text))

    reader = ReadBrat()
    # FIND EVENTS#
    ftrs, _ = doc2featureCRF(doc,reader, text=text)

    
    clf = pipeline["event_tagger"]

    y = clf.predict(ftrs)

    result = {"actor": [], "char_span": []}
    symbols = string.punctuation + "\”\"\“"

    # colocar em outro formato aqui
    idx_sent = 0
    for sent in doc.sents:
        idx_tok = 0
        for tok in sent:
            if y[idx_sent][idx_tok] == "I-EVENT" or y[idx_sent][idx_tok] == "B-EVENT":
                # TODO: the model with punctuation seems to be 
                # better, but it includes punctuation as events. 
                # So, fix this. However, try to fix this in another way.
                if tok.text not in symbols:
                    result["actor"].append(tok.text)
                    result["char_span"].append((tok.idx, tok.idx + len(tok.text)))
            idx_tok += 1
        idx_sent += 1

    return pd.DataFrame(result)


if __name__ == "__main__":
    load()
    data_dir = "data/train"
    reader = ReadBrat()
    execute_train(data_dir, reader)
    # unit test
    # ans = extract_events("pt","O cavalo correu na pista de corrida. E tudo deu certo naquele dia. Depois o cavalo assaltou um casal.")
    # print(ans)


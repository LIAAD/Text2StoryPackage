
from typing import List, Dict, Tuple

from text2story.readers.token_corpus import TokenCorpus
from text2story.core.utils import join_tokens

def select_sentences(doc:List[TokenCorpus]) -> List[Tuple[int,List[TokenCorpus]]]:
    """
    select sentences from a document, i.e., a token corpus list
    @param doc: a token corpus list that represents a text document
    @return: return a list of tuples that maps each sentence id to its correspondent list of tokens
    (each one as a TokenCorpus object)
    """

    sentences = []
    current_sent_id = None
    current_sent = []

    for tok in doc:
        if current_sent_id != None and tok.sent_id == current_sent_id:
            current_sent.append(tok)
        else:
            if current_sent_id is None:
                current_sent.append(tok)
            else:
                sentences.append((current_sent_id, current_sent))
                current_sent = [tok]
            current_sent_id = tok.sent_id

    sentences.append((current_sent_id, current_sent))


    return sentences

def select_sentences_ann(sent_lst:List[Tuple[int,List[TokenCorpus]]], narrative_component:str,**kwargs)->List[Tuple[int,List[TokenCorpus]]]:
    """

    @param sent_lst: A sentence list (the output of the method select_sentences)
    @param narrative_component: 'event','participant', or 'time'
    @param kwargs: attribute list related to the specified component to filter annotations
    @return: a list of sentences with the specified annotations
    """

    #print(len(kwargs))
    sent_id_lst = set()
    for sent_id, sent in sent_lst:
        for tok in sent:

            if tok.attr != []:
                for ann in tok.attr:
                    ann_type, ann_attr = ann
                    if ann_type.lower() == narrative_component:
                        if len(kwargs) > 0:
                            requirements = True
                            for k in kwargs:
                                if k not in ann_attr:
                                    requirements = False
                                    break
                                else:
                                    if ann_attr[k] != kwargs[k]:
                                        requirements = False
                                        break
                            if requirements:
                                sent_id_lst.add(sent_id)
                        else:
                            sent_id_lst.add(sent_id)
    ans = []

    for (sent_id, sent) in sent_lst:
        if sent_id in sent_id_lst:
            ans.append((sent_id, sent))

    return ans

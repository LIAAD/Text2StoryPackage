"""
    text2story/core/utils module

    Functions
    ---------
    chunknize_actors(annotations)
        converts the result from the 'extract_actors' method, implemented by the annotators supported, to a list the chunknized list of actors
        to do this conversion, the IOB in the NE tag is used
"""

from itertools import tee

import warnings

import numpy as np
from typing import Tuple, List


def bsearch_tuplelist(x: int, xs: List[Tuple[int, int]]) -> int:
    """
    Use binary search to check if x is whithin some interval of a tuple list
    xs.

    if x is not in none of intervals, it return -1, otherwise returns de
    position of the interval.
    @type xs: object

    """
    b = 0
    e = len(xs) - 1
    m = (b + e) // 2
    pos = -1

    while b <= e:
        (x_b, x_e) = xs[m]

        if x>= x_b and x<= x_e:
            return m
        elif x < x_b:
            e = m - 1
        elif x > x_e:
            b = m + 1

        m = (b + e) // 2

    return pos



def join_tokens(tok_lst):
    pronomes = ["me", "te", "lhe", "o", "a", "la", "lo", "lho", "lha", "nos"]
    parentheses = ["(", "{"]
    punctuation = [",", ".", ":", ";", "?", "!", ")", "}"]

    # first, find all hyphen in the token list
    array_tok_lst = np.array(tok_lst)
    indices = np.where(array_tok_lst == "-")[0]
    indices = list(indices)

    if indices != []:
        tok_join = ""
        if len(tok_lst) > 0:
            i = 0
            while i < len(tok_lst):
                if i in indices:
                    if i + 1 < len(tok_lst) and \
                            tok_lst[i + 1].lower() in pronomes:
                        tok_join = tok_join.strip() + tok_lst[i]
                        i = i + 1

                if tok_lst[i] in parentheses:
                    tok_join = tok_join + tok_lst[i]
                else:
                    if tok_lst[i] in punctuation:
                        tok_join = tok_join.rstrip() + tok_lst[i] + " "
                    else:
                        tok_join = tok_join + tok_lst[i] + " "
                i = i + 1

        return tok_join.strip()
    else:
        tok_join = ""
        if len(tok_lst) > 0:
            i = 0
            while i < len(tok_lst):

                if tok_lst[i] in parentheses:
                    tok_join = tok_join + tok_lst[i]
                else:
                    if tok_lst[i] in punctuation:
                        tok_join = tok_join.rstrip() + tok_lst[i] + " "
                    else:
                        tok_join = tok_join + tok_lst[i] + " "
                i = i + 1

        return tok_join.strip()


def chunknize_actors(annotations):
    """
    Parameters
    ----------
    annotations : list[tuple[tuple[int, int], str, str]]
        list of annotations made by some tool for each token

    Returns
    -------
    list[tuple[tuple[int, int], str, str]]
        the list of actors identified where each actor is represented by a tuple
    """

    actors = []

    ready_to_add = False

    prev_ne_tag = ''

    for ann in annotations:
        token_character_span, token_pos_tag, token_ne_iob_tag = ann

        # token_ne_iob_tag = 'I-PER', then token_ne_iob_tag[2:] == 'PER'
        if token_ne_iob_tag.startswith("B") or (
                token_ne_iob_tag.startswith("I") and token_ne_iob_tag[2:] != prev_ne_tag):
            # Case we start a new chunk, after finishing another, for instance: Case B-Per, I-Per, B(or I)-Org, then we add the finished actor
            if ready_to_add:
                actor = ((actor_start_offset, actor_end_offset), actor_lexical_head, actor_actor_type)
                actors.append(actor)

            ready_to_add = True
            actor_start_offset = token_character_span[0]
            actor_end_offset = token_character_span[1]
            actor_lexical_head = token_pos_tag if token_pos_tag in ['Noun', 'Pronoun'] else 'UNDEF'
            actor_actor_type = token_ne_iob_tag[2:]

        elif token_ne_iob_tag.startswith("I"):
            # actor_start_offset it's always the same, since it's defined by the first token of the actor
            actor_end_offset = token_character_span[1]
            actor_lexical_head = actor_lexical_head if actor_lexical_head != 'UNDEF' else token_pos_tag if token_pos_tag in [
                'Noun', 'Pronoun'] else 'UNDEF'
            # actor_actor_type it's the same for all tokens that constitute the actor and it's already defined by the first token of the actor

        elif token_ne_iob_tag.startswith("O") and ready_to_add:
            actor = ((actor_start_offset, actor_end_offset), actor_lexical_head, actor_actor_type)
            actors.append(actor)
            ready_to_add = False
            # No need to reset the variables, since the first update writes over

        prev_ne_tag = token_ne_iob_tag[2:]

    if ready_to_add:  # If the last token still makes part of the actor
        actor = ((actor_start_offset, actor_end_offset), actor_lexical_head, actor_actor_type)
        actors.append(actor)

    return actors


def pairwise(iterable):
    """
    Iterate through some iterable with a lookahead.
    From the itertools docs recipes - https://docs.python.org/3/library/itertools.html
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def merge_dicts(d1, d2):
    for k2 in d2:
        for attr in d2[k2]:
            if k2 in d1:
                if attr in d1[k2]:
                    d1[k2][attr] += d2[k2][attr]
                else:
                    d1[k2][attr] = d2[k2][attr]

            else:
                d1[k2] = {attr:d2[k2][attr]}

def capfirst(s):
    return s[:1].upper() + s[1:]

def reset_offset_ann(doc, txt, start_offset=0, start_txt=0):
    """
    Given a document in tokencorpus format, looks for the spans in txt
    and then reset the span
    @param doc: A TokenCorpus LIst
    @param txt: a string
    @return: a token corpus list with new spans
    """
    start = start_txt
    for tok in doc:
        if tok.offset >= start_offset:
            start_tok = txt.find(tok.text, start)
            if start_tok == -1:
                warnings.warn("reset_offset_ann: The token %s was not found in given text." % tok.text)
            else:
                #print(">>", tok.text, tok.offset, start_tok)
                tok.offset = start_tok
                start = start_tok + len(tok.text)

    return doc

def diff_ann(doc1, doc2):
    ann1 = {}
    for tok in doc1:
        if tok.attr is not None:
            for id in tok.id_ann:
                if id in ann1:
                    ann1[id] = ann1[id] + " " + tok.text
                else:
                    ann1[id] = tok.text
    ann2 = {}
    for tok in doc2:
        if tok.attr is not None:
            for id in tok.id_ann:
                if id in ann2:
                    ann2[id] = ann2[id] + " " + tok.text
                else:
                    ann2[id] = tok.text




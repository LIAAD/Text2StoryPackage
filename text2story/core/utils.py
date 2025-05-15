"""
    text2story/core/utils module

    Functions
    ---------
    chunknize_actors(annotations)
        converts the result from the 'extract_actors' method, implemented by the annotators supported, to a list the chunknized list of actors
        to do this conversion, the IOB in the NE tag is used
"""

from itertools import tee
import subprocess
import sys

import os

import warnings

import importlib_metadata
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

    special_suffix = ["se","me", "te", "lhe", "o", "a", "la", "lo", "lho", "lha", "nos","feira"]
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
                            tok_lst[i + 1].lower() in special_suffix:
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


def normalize_tag(label):
    """
    Parameters
    ----------
    label : str

    Returns
    -------
    str
        the label normalized
    """

    mapping = {
        # POS tags
        # Universal POS Tags
        # http://universaldependencies.org/u/pos/

        # "ADJ": "adjective",
        # "ADP": "adposition",
        # "ADV": "adverb",
        # "AUX": "auxiliary",
        # "CONJ": "conjunction",
        # "CCONJ": "coordinating conjunction",
        # "DET": "determiner",
        # "INTJ": "interjection",
        "NOUN": "Noun",
        # "NUM": "numeral",
        # "PART": "particle",
        "PRON": "Pronoun",
        "PROPN": "Noun",
        "NPROP":"Noun",
        # "PUNCT": "punctuation",
        # "SCONJ": "subordinating conjunction",
        # "SYM": "symbol",
        # "VERB": "verb",
        # "X": "other",
        # "EOL": "end of line",
        # "SPACE": "space",

        # NE
        # en
        'CARDINAL': 'Other',  # 'Numerals that do not fall under another type'
        'DATE': 'Date',  # 'Absolute or relative dates or periods'
        'EVENT': 'Other',  # 'Named hurricanes, battles, wars, sports events, etc.'
        'FAC': 'Loc',  # 'Buildings, airports, highways, bridges, etc.'
        'GPE': 'Loc',  # 'Countries, cities, states'
        'LANGUAGE': 'Other',  # 'Any named language'
        'LAW': 'Other',  # 'Named documents made into laws.'
        'LOC': 'Loc',  # 'Non-GPE locations, mountain ranges, bodies of water'
        'MONEY': 'Other',  # 'Monetary values, including unit'
        'NORP': 'Other',  # 'Nationalities or religious or political groups'
        'ORDINAL': 'Other',  # '"first", "second", etc.'
        'ORG': 'Org',  # 'Companies, agencies, institutions, etc.'
        'PERCENT': 'Other',  # 'Percentage, including "%"'
        'PERSON': 'Per',  # 'People, including fictional'
        'PRODUCT': 'Obj',  # 'Objects, vehicles, foods, etc. (not services)'
        'QUANTITY': 'Other',  # 'Measurements, as of weight or distance'
        'TIME': 'Time',  # 'Times smaller than a day'
        'WORK_OF_ART': 'Other',  # 'Titles of books, songs, etc.'

        # pt
        # 'LOC'
        'MISC': 'Other',  # 'Miscellaneous entities, e.g. events, nationalities, products or works of art'
        # 'ORG'
        'PER': 'Per'  # 'People, including fictional'
    }

    return mapping.get(label, 'UNDEF')


def install(pkg,path):
    return subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,"--target={}".format(path)])


def is_library_installed(library_name, target_directory):
    try:
        installed_distributions = importlib_metadata.distributions(path=[target_directory])

        for distribution in installed_distributions:
            if distribution.metadata['Name'] == library_name:
                return True

        return False
    except Exception:
        return False

def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

def find_target_dir(pathname, target):
    subfolders = fast_scandir(pathname)
    for p in subfolders:
        if p.endswith(target):
            return p

def map_pos2head(pos_tag):
    p = pos_tag.lower()

    map = {"noun":"Noun","propn":"Noun","pron":"Pronoun","det":"Noun"}
    if p in map:
        return map[p]
    else:
        return "None"

def find_first_non_space(s, p):
    # Check if p is out of range
    if p < 0 or p >= len(s):
        return -1  # Invalid position

    # Start from position p and iterate until a non-space character is found
    while p < len(s) and s[p].isspace():
        p += 1

    # Check if we reached the end of the string without finding a non-space character
    if p >= len(s):
        return -1  # No non-space character found

    return p  # Return the position of the first non-space character

def find_substring_match(matches, position):

    for match in matches:
        # Get the matched substring
        matched_substring = match.group(0)

        # Get the start and end positions of the matched substring
        start_pos = match.start()
        end_pos = match.end()

        # Check if the matched substring is within the desired range (starting from x)
        if start_pos >= position:
            return start_pos

def update_offsets(old_text, new_text, annotation_file):
    """
    Update character offsets in annotation file based on new text positions.

    Args:
        old_text (str): Original text content
        new_text (str): Modified text content
        annotation_file (str): Content of the annotation file

    Returns:
        str: Updated annotation file content
    """
    # Read annotations line by line
    updated_annotations = []
    annotations = annotation_file.strip().split('\n')

    # Create a mapping of old expressions to their new positions
    expression_mapping = {}

    for line in annotations:
        parts = line.split()
        if len(parts) >= 4 and parts[0].startswith('T'):  # Only process T lines
            # Extract the expression from old text using offsets
            start, end = int(parts[2]), int(parts[3])
            expression = old_text[start:end].strip()

            # Find new position of expression in new text
            # Start searching from beginning to maintain order
            search_start = 0
            while True:
                new_start = new_text.find(expression, search_start)
                if new_start == -1:
                    break

                # Check if this position maintains relative ordering
                valid_position = True
                for old_expr, (prev_start, _) in expression_mapping.items():
                    old_pos = old_text.find(old_expr)
                    if (old_pos < start and new_start < prev_start) or \
                            (old_pos > start and new_start > prev_start):
                        continue
                    valid_position = False
                    break

                if valid_position:
                    new_end = new_start + len(expression)
                    expression_mapping[expression] = (new_start, new_end)
                    break
                search_start = new_start + 1

    # Update annotations with new offsets
    for line in annotations:
        parts = line.split()
        if len(parts) >= 4 and parts[0].startswith('T'):
            # Extract the expression from old text
            start, end = int(parts[2]), int(parts[3])
            expression = old_text[start:end].strip()

            if expression in expression_mapping:
                new_start, new_end = expression_mapping[expression]
                # Replace old offsets with new ones
                parts[2] = str(new_start)
                parts[3] = str(new_end)

                updated_line = parts[0] + '\t' + ' '.join(parts[1:4]) + '\t' + ' '.join(parts[4:])
            else:
                updated_line = line  # Keep original if expression not found
        else:
            updated_line = line  # Keep non-T lines unchanged

        updated_annotations.append(updated_line)

    return '\n'.join(updated_annotations)
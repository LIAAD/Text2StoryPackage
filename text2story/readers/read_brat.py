import os
import sys
from pathlib import Path

from . import read
from . import token_corpus

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS, HYPHENS
from spacy.util import compile_infix_regex

import re
import copy
import numpy as np


# regular expression of an integer number
# NUMBER_RE = re.compile("^[0-9]*[1-9][0-9]*$")
NUMBER_RE = re.compile("^[0-9]+$")

LINK_TYPES = ["SEMROLE", "SRLINK"]

LINK_TYPES = ["SEMROLE","SRLINK"]

infix_re = re.compile(r'[-]')


def custom_tokenizer(nlp):
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )

    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

class ReadBrat(read.Read):
    """
      Reader to brat file annotations and their text files
    """

    def __init__(self, lang="en"):
        """
        Load spacy model to read process brat files. Also
        store the files processed.
        """
        if lang == "pt":
            if not(spacy.util.is_package('pt_core_news_lg')):
                spacy.cli.download('pt_core_news_lg')


            self.nlp = spacy.load("pt_core_news_lg")


            self.nlp.tokenizer = custom_tokenizer(self.nlp)
        else:
            if not (spacy.util.is_package('en_core_web_lg')):
                spacy.cli.download('en_core_web_lg')

            self.nlp = spacy.load('en_core_web_lg')

        self.file_lst = []
        self.ann_ref = {}

    def toColumn(self, data_dir, output_dir):
        """
        Convert a set of files to column format, similar to Conll

        @param string: path of data to gather and process
        @param string: path of the ouput directory

        @return None
        """

        for dirpath, dirnames, filenames in os.walk(data_dir):
            for f in filenames:
                if f.endswith(".ann"):
                    p = Path(f)
                    fullname = os.path.join(data_dir, p.stem)
                    output_file = os.path.join(output_dir, "%s.conll" % p.stem)
                    self.fileToColumnFormat(fullname, output_file)

    def _is_adjacent_offset(self, i1, i2):
        """
        test if two intervals are adjacent

        @param (int,int): a tuple of integers
        @param (int,int): a tuple of integers

        @return int: 1 if i1 is adjacent, but before i2, -1 if i1 is adjacent
        , but after i2, and 0 otherwise
        """

        s1, e1 = i1
        s2, e2 = i2

        # if the interval i1 is adjacent, but it is placed before
        # the interval i2
        if e1 + 1 == s2:
            return 1
        elif e2 + 1 == s1:
            # if the interval i1 is adjacent, but it is placed after
            # the interval i2
            return -1
        else:
            return 0

    def _get_left_span(self, ann):
        """
        Given an annotation element (dictionary), it takes the left most span interval,

        """

        # (start, end)
        return (ann["offset1"][0], ann["offset1"][1])

    def _get_right_span(self, ann):
        """
        Given an annotation element (dictionary), it takes the right most span interval,

        """
        if "offset2" in ann:
            return (ann["offset2"][0], ann["offset2"][1])
        else:
            return (ann["offset1"][0], ann["offset1"][1])

    def _is_adjacent(self, el1, el2):
        """
        Check if two annotation elements (dictionaries) are adjacent to each other
        1, if el1 < el2, -1, if el1 > el2, and 0 if el1 is not adjacent to el2
        """

        if "offset2" in el1 and "offset2" in el2:
            return 0

        (left1_start, left1_end) = self._get_left_span(el1)
        (left2_start, left2_end) = self._get_left_span(el2)


        (right1_start, right1_end) = self._get_right_span(el1)
        (right2_start, right2_end) = self._get_right_span(el2)

        ans = self._is_adjacent_offset((left1_start, left1_end), (left2_start, left2_end))
        if ans != 0: return ans

        ans = self._is_adjacent_offset((left1_start, left1_end), (right2_start, right2_end))
        if ans != 0: return ans

        ans = self._is_adjacent_offset((right1_start, right1_end), (left2_start, left2_end))
        if ans != 0: return ans

        ans = self._is_adjacent_offset((right1_start, right1_end), (right2_start, right2_end))
        if ans != 0: return ans

        return 0

    def _build_merged_element(self, el1, el2):

        new_el = {}
        if "offset2" in el1:

            new_el["offset1"] = (el1["offset1"][0], el1["offset1"][1])

            # the union if make in the second offset
            new_el["offset2"] = (el1["offset2"][0], el2["offset2"][1])

        else:
            # the union if make in the first offset
            new_el["offset1"] = (el1["offset1"][0], el2["offset1"][1])
            if "offset2" in el2:
                new_el["offset2"] = (el2["offset2"][0], el2["offset2"][1])

        new_el["value"] = el1["value"] + " " + el2["value"]
        new_el["id"] = el1["id"]
        return new_el

    def merge_span(self, ann_entity):
        """
        It merge spans of annotations

        @param [dict]: a list of annotations as dictionaries

        @param [dict]:a list of annotations as dictionaries that were merged
        if they are in the same span text
        """

        # TODO: a recursive solution will be more suitable solution

        new_ann_type = []
        merged_indexes = []  # register elements that were already merged

        for idx_el, el in enumerate(ann_entity):

            # search for an element that is inside the
            # current span or contains the current span
            ans = 0
            for idx in range(idx_el + 1, len(ann_entity)):

                ans = self._is_adjacent(el, ann_entity[idx])

                if ans > 0:
                    new_el = self._build_merged_element(el, ann_entity[idx])
                    new_ann_type.append(new_el)
                    merged_indexes.append(idx)
                    break
                elif ans < 0:
                    new_el = self._build_merged_element(ann_entity[idx], el)
                    new_ann_type.append(new_el)
                    merged_indexes.append(idx)
                    break

            if ans == 0:
                if idx_el not in merged_indexes:
                    new_ann_type.append(el)

        return new_ann_type

    def merge_entity_span(self, ann):
        """
        Given a dictionary returned by read_annotation_file method,
        it merge span of entities

        @param dict: a dictionary of different entities that contains their
        annotation

        @return dict: a dictionary of different entities that contains their
        annotation merged if they are in the same text span
        """

        new_ann = {}
        for ent_type in ann:
            if ent_type in LINK_TYPES:
                new_ann[ent_type] = ann[ent_type]
                continue # if it is a link, just ignore it
            #if "offset2_end" not in ann[ent_type]:
            #    print("-->", ann[ent_type])
            new_ann_ent = self.merge_span(ann[ent_type])
            new_ann[ent_type] = new_ann_ent

        return new_ann

    def get_attribute_value(self, element_id, attr_name):
        if attr_name in self.ann_ref:
            if element_id in self.ann_ref[attr_name]:
                return self.ann_ref[attr_name][element_id]

    def get_relation(self, element_a, element_b):
        """
        Get the relation name between two entities (their ids)
        @param element_a: String with the id of an entity
        @param element_b: String with the id of an entity
        @return: String with relation name, NOne if there is no relation between
        elements A and B
        """

        annotation_types = list(self.ann_ref.keys())
        annotation_links_types = [ann_type for ann_type in annotation_types if "LINK" in ann_type]

        for relation_name in  annotation_links_types:
            if element_a in self.ann_ref[relation_name]:
                if element_b in self.ann_ref[relation_name][element_a]:
                    return relation_name



    def read_annotation_file(self, file_ann, merge_entities=False):
        """
        It reads only the annotation file, then returns
        the processed tokens as TokenCorpus type

        @param string: path of data to gather and process

        @return dictionary: a dictionary of annotations
        """

        ann = {"Event": [], "Actor": [], "Time": [], "TIME_X3": [], \
                "ACTOR": [], "Participant": [],"SRLINK":[],"OLINK":[],"TLINK":[]}
        ann_ref = {}  # annotation by reference
        # the pattern of a reference for an argument of a relation
        ARG_REL = re.compile("T\d+")

        with open(file_ann, "r") as fd:
            last_line_type = None
            for line in fd:
                ann_type = None
                line_toks = line.split()
                if line[0] != '#':

                    if len(line_toks) < 3:
                        if line[0] != 'T' and line[0] != 'A' and line[0] != 'R' and last_line_type in ann.keys():
                            ann[last_line_type][-1]["value"] += " " + line.strip()
                        else:
                            raise Exception(f"Invalid standoff line:{line}")
                        continue
                    ann_type = line_toks[1]
                    last_line_type = ann_type

                    if line[0] == 'T':
                        # TODO: catalogar aqui todos as entidades em um mapeamento

                        if ann_type not in ann:
                            ann[ann_type] = []

                        if ';' in line_toks[3]:

                            start_range = int(line_toks[2])
                            offset_lst = []

                            idx_toks = 3
                            while ';' in line_toks[idx_toks]:
                                tmp = line_toks[idx_toks].split(';')
                                end_range = int(tmp[0])
                                offset_lst.append((start_range, end_range))

                                start_range = int(tmp[1])
                                idx_toks += 1

                            end_range = int(line_toks[idx_toks])
                            offset_lst.append((start_range, end_range))

                            segment_ann = {"id":line_toks[0]} # annotation multi-segment annotation
                            for idx_offset, (start_range, end_range) in enumerate(offset_lst):
                                segment_ann[f"offset{idx_offset}"] = (start_range, end_range)

                            value = " ".join(line_toks[idx_toks + 1:])
                            segment_ann["value"] = value
                            ann[ann_type].append(segment_ann)

                        else:
                            offset1_start = int(line_toks[2])
                            offset1_end = int(line_toks[3])

                            value = " ".join(line_toks[4:])

                            ann[ann_type].append({"id":line_toks[0],"offset1": (offset1_start, offset1_end), \
                                              "value": value})
                    elif line[0] == 'R':
                        e1 = line_toks[2].split(":")[1]  # entity 1
                        e2 = line_toks[3].split(":")[1]  # entity 2
                        rel_id = line_toks[0]
                        if ann_type in ann:
                            ann[ann_type].append({"id":rel_id,"args":(e1, e2)})
                        else:
                            ann[ann_type] = [{"id": rel_id, "args": (e1, e2)}]


                        if ann_type in self.ann_ref:
                            if e1 in self.ann_ref[ann_type]:
                                self.ann_ref[ann_type][e1].append(e2)
                            else:
                                self.ann_ref[ann_type][e1] = [e2]
                        else:
                            self.ann_ref[ann_type] = {e1: [e2]}

                    elif line[0] == 'A':

                        ref = line_toks[2]
                        if ref in ann_ref:
                            # ann_ref[ref][1].append((line[1], line[3]))
                            if len(line_toks) > 3:
                                ann_ref[ref][line_toks[1]] = line_toks[3]
                            else:
                                ann_ref[ref][line_toks[1]] = line_toks[1]

                        else:
                            ann_ref[ref] = {line_toks[1]: line_toks[3]}
                    elif line[0] == '#':
                            # it is a note about the annotation,
                            # insert it as an attribute
                            ann_type = line_toks[1]
                            ref = line_toks[2]
                            ann_info = " ".join(line_toks[3:])

                            if ref in ann_ref:
                                if ann_type in ann_ref[ref]:
                                    ann_ref[ref][ann_type].append(ann_info)
                                else:
                                    ann_ref[ref][ann_type] = [ann_info]
                            else:
                                ann_ref[ref] = {ann_type: [ann_info]}

        if merge_entities:
            return self.merge_entity_span(ann)
        else:
            return ann

    def process(self, data_dir, split=None):
        """
        It reads a set of files of annotations and text, then returns
        the processed tokens as TokenCorpus type

        @param string: path of data to gather and process
        @param string: a file name that contains a list of the files in data dir that
        should be processed by this reader

        @return [[TokenCorpus]]: a list of lists of tokens
        """
        # process the data corpus
        # and return a list of tokens
        split_lst = []
        if split is not None:
            with open(split, "r") as fd:
                split_lst = fd.readlines()
                split_lst = [line.replace("\n","") for line in split_lst]

        data_tokens = []

        for dirpath, dirnames, filenames in os.walk(data_dir):
            for f in filenames:

                if f.endswith(".ann"):

                    p = Path(f)
                    if split_lst != [] and not(p.stem + ".txt" in split_lst):
                        continue
                 
                    fullname = os.path.join(data_dir, p.stem)
                    #print("-->", fullname)
                    token_lst = self.process_file(fullname)
                    self.file_lst.append(fullname + ".txt")

                    if len(token_lst) > 0:
                        data_tokens.append(token_lst)

        return data_tokens

    def extract_token_corpus(self, doc, ann_idx, ann, ann_ref):
        """
        Build a list of tokens using TokenCorpus object. It uses 
        a document of spacy.Document type (doc) and a list of indexes.

        ann_idx is a maps a 2-tuple integer to a list of annotation. Regarfing, 
        the 2-tuple, the first element  is the start character offset of 
        an annotated token in the 
        raw text of the document, and the second element is end of the character 
        offset of an annotated token.

        ann is a dictionary that maps a 2-tuple of indexes to a list of 
        the annotation of the given span indexes. 

        ann_ref is a dictionary that maps a reference (i.e. an id of an annotation 
        assigned by BRAT) to the attribute list of an annotation span
        """
        count = 0
        token_lst = []
        ref2tok = {}

    
        sent_id = -1
        clause_id = 0
        for tok_idx, tok in enumerate(doc):
            if tok.is_sent_start:
                sent_id += 1

            # Check if the token marks the end of a clause
            if tok.dep_ in ["punct", "mark"]:
                clause_id += 1

            
            mytok = token_corpus.TokenCorpus()
            mytok.id = count
            mytok.text = tok.text
            mytok.lemma = tok.lemma_
            mytok.pos = tok.pos_
            mytok.dep = tok.dep_
            mytok.head = tok.head.text
            mytok.head_pos = tok.head.pos_
            mytok.head_lemma = tok.head.lemma_
            mytok.offset = tok.idx
            mytok.sent_id = sent_id
            mytok.clause_id = clause_id
            #print("-->",clause_id, sent_id)


            ans = self.search_all_idx(tok.idx, tok.idx + len(tok.text), ann_idx)

            # TODO: it is necessary perform a more efficient search
            # a possible subtoken annotation
            # perform more than one search, and build a list of id's
            if len(tok.text.strip()) != 0 and tok.text not in ".!?-,":
                ans_sub = self.search_subtoken(tok.idx, tok.idx + len(tok.text) - 1, \
                        ann_idx)

                ans = ans.union(ans_sub)

            if len(ans) != 0:
                # annotations in token
                # a token can be annotated twice, and besides that,
                # can be a part of a span annotation..The iteration in ans
                # tries to get the span annotations in one token, and the 
                # iteration in ann[(t0,t1)] bellow tries to get several annotations
                # in only one token
                ref_lst_tok = set()
                for idx in ans:
                    (t0, t1) = ann_idx[idx]


                    for ref, ann_type, elems in ann[(t0, t1)]:

                        mytok.attr.append((ann_type,ann_ref[ref]))
                        mytok.ann_offset.append((t0,t1))
                        mytok.id_ann.append(ref) 
                        # all the annotations ids that a token can be assigned
                        ref_lst_tok.add(ref)


                # now update the mapping of references to tokens with
                # the token object that was build in this iteration
                for ref in ref_lst_tok:
                    if ref in ref2tok:
                        ref2tok[ref].append(mytok)
                    else:
                        ref2tok[ref] = [mytok]

            token_lst.append(mytok)

            count += 1

        return token_lst, ref2tok

    # TODO: refactor this method. It is a very long and confusing method
    def process_file(self, data_file):
        """
        It reads only one file of annotation and text, then returns
        the processed tokens as TokenCorpus type

        @param string: path of data to gather and process

        @return [TokenCorpus]: a list of tokens
        """

        file_ann = "%s.ann" % data_file
        file_txt = "%s.txt" % data_file


        if not os.path.exists(file_txt) or not os.path.exists(file_ann):
            return []

        ann = {}  # the index is the offset of the annotation
        ann_ref = {}  # annotation by reference
        ann_rel = {} # annotation by relation

        # the pattern of a reference for an argument of a relation
        ARG_REL = re.compile("T\d+") 

        with open(file_ann, "r") as fd:

            for line in fd:
                line_toks = line.split()
                if line[0] != '#' and line[0] == 'T':

                    # ann[offset start] = (ref, ann_type, value_str)
                    if ';' in line_toks[3]:
                        # this situation is when the annotation of th event
                        # is two segments not adjacents
                        start_range = int(line_toks[2])
                        offset_lst = []

                        idx_toks = 3
                        while ';' in line_toks[idx_toks]:

                            tmp = line_toks[idx_toks].split(';')
                            end_range = int(tmp[0])
                            offset_lst.append((start_range, end_range))

                            start_range = int(tmp[1])
                            idx_toks += 1

                        end_range = int(line_toks[idx_toks])
                        offset_lst.append((start_range, end_range))

                        for start_range, end_range in offset_lst:
                            if (start_range, end_range) in ann:
                                ann[(start_range, end_range)].append((line_toks[0], line_toks[1], line_toks[idx_toks + 1:]))
                            else:
                                ann[(start_range, end_range)] = [(line_toks[0], line_toks[1], line_toks[idx_toks + 1:])]

                        #ann[(int(line_toks[2]), int(tmp[0]))] = [(line_toks[0], line_toks[1], line_toks[5:])]
                        #ann[(int(tmp[1]), int(line_toks[4]))] = [(line_toks[0], line_toks[1], line_toks[5:])]
                        #ann_ref[line[0]] = (
                        #    [(int(line_toks[2]), int(tmp[0])), (int(tmp[1]), int(line_toks[4]))], [])
                        ann_ref[line[0]] = (offset_lst, [])


                        ann_ref[line_toks[0]] = {}
                    else:
                        if NUMBER_RE.match(line_toks[2]):

                            offset_start, offset_end = int(line_toks[2]), int(line_toks[3])

                            if (offset_start, offset_end) in ann:
                                ann[(offset_start, offset_end)].append((line_toks[0], line_toks[1], line_toks[4:]))
                            else:
                                ann[(offset_start, offset_end)] = [(line_toks[0], line_toks[1], line_toks[4:])]
                                
                            ann_ref[line_toks[0]] = {}
                elif line[0] == 'R':
                    rel_id = line_toks[0]

                    rel_type = line_toks[1]
                    ref_lst = ARG_REL.findall(line)

                    if len(ref_lst) > 1:

                        ref1 = ref_lst[0]
                        ref2 = ref_lst[1]

                        if ref1 in ann_rel:
                            ann_rel[ref1].append((rel_id, rel_type, ref2, "arg2"))
                        else:
                            ann_rel[ref1] = [(rel_id, rel_type, ref2,"arg2")]

                        if ref2 in ann_rel:
                            ann_rel[ref2].append((rel_id, rel_type, ref1,"arg1"))
                        else:
                            ann_rel[ref2] = [(rel_id, rel_type, ref1,"arg1")]

                    else:
                        print("Warning: There is a relation with only one argument.")



                elif line[0] == 'A':

                    ref = line_toks[2]
                    if ref in ann_ref:
                        #ann_ref[ref][1].append((line[1], line[3]))
                        if len(line_toks) > 3:
                            ann_ref[ref][line_toks[1]] = line_toks[3]
                        else:
                            ann_ref[ref][line_toks[1]] = line_toks[1]

                    else:
                        ann_ref[ref] = {line_toks[1]:line_toks[3]}
                elif line[0] == '#':
                    # it is a note about the annotation,
                    # insert it as an attribute
                    ann_type = line_toks[1]
                    ref = line_toks[2]
                    ann_info = " ".join(line_toks[3:])

                    if ref in ann_ref:
                        if ann_type in ann_ref[ref]:
                            ann_ref[ref][ann_type].append(ann_info)
                        else:
                            ann_ref[ref][ann_type] = [ann_info]
                    else:
                        ann_ref[ref] = {ann_type: [ann_info]}

        idx_lst = ann.keys()

        idx_lst = sorted(idx_lst, key=lambda elem: elem[0])

        # a mapping between ref and Tok object
        # the use it to point to the tokens that it own relations
        ref2tok = {} 
        token_lst = []

        with open(file_txt, "r") as fd:

            doc = self.nlp(fd.read().strip())
            token_lst, ref2tok = self.extract_token_corpus(doc, idx_lst, ann, ann_ref)

        # update relation field for each token
        # each relation field is a list of TokenRelation object, which 
        # points to a TokenCorpus and specifies the type of relation
        for mytok in token_lst:

            if mytok.id_ann != []:

                for ref  in mytok.id_ann:

                    if ref in ann_rel: # if the current token has any relation, then...

                        # the relation for the reference ref
                        for rel_id, rel_type, ref_arg,argn in ann_rel[ref]:
                            try:
                                rel_obj = token_corpus.TokenRelation(rel_id, ref2tok[ref_arg], rel_type, argn, ref_arg)
                                mytok.relations.append(rel_obj)
                            except KeyError:
                                print(f"Warning: The {rel_id} was not included in the doc since {ref_arg} was not found as annotation.")

        return token_lst

    def search_all_idx(self, idx, idx_end, idx_lst):
        """
        Since a token can be annotated with multiple id's, it is necessary 
        to perform multiple search for each one of these id's.
        """

        all_idx_lst = set()

        copy_idx_lst = [e for e in idx_lst]
        
        while True:

            ans = self.search_idx(idx, copy_idx_lst)
            if ans == -1:
                return all_idx_lst
            else:
                # update index according to the position
                # in the idx_lst 
                elem = copy_idx_lst[ans]
                copy_idx_lst.pop(ans)
                if len(idx_lst) > 0:
                    # can I afford such expensive operation
                    # since few of them are possible performed?
                    ans = idx_lst.index(elem)

                all_idx_lst.add(ans)


    def search_subtoken(self, tok_start, tok_end, idx_lst):

        ans = set()

        for pos, (start, end) in enumerate(idx_lst):
            
            #if start >= tok_start and end <= tok_end:
            if not (tok_end < start or tok_start > end):
                ans.add(pos)

        return ans

    def search_idx(self, idx, idx_lst):
        """
        it searches for tuples (t0, t1) in idx_lst where idx >= t0 and idx <= t1

        @param integer: an index
        @param [(integer,integer)]: a list of index

        @return integer: the position of the tuple or -1 if none is found
        """

        b = 0
        e = len(idx_lst) - 1

        m = int((b + e) / 2)
        pos = -1

        while b <= e:
            (t0, t1) = idx_lst[m]

            if idx == t0:
                pos = m
                break
            elif idx > t0:
                if idx >= t1:
                    b = m + 1
                else:
                    pos = m
                    break
            else:
                e = m - 1

            m = int((b + e) / 2)

        return pos

    def __process_annotations(self, file_ann, ann, ann_ref):
        pass


    def fileToColumnFormat(self, ann_file, output_file):
        """
        Convert only one file to the column format.

        @param string: path of annotation file
        @param string: path of output file

        @return None
        """
        print("Processing %s..." % ann_file)

        tok_lst = self.process_file(ann_file)
        with open(output_file, "w") as fd:

            for tok in tok_lst:

                if tok.ann == 'Event':
                    ann = 'I'
                else:
                    ann = 'O'

                fd.write("%d %s %s %s\n" % (tok.id, tok.text.strip(), tok.pos, ann))
        print("Output %s" % output_file)


if __name__ == "__main__":
    # only to unit  tests
    data_dir = os.environ.get("DATA_DIR")
    output_dir = os.environ.get("COLUMN_DIR")

    if data_dir is None:
        print("Please, set DATA_DIR enviroment variable.")
        sys.exit(0)
    if output_dir is None:
        print("Please, set COLUMN_DIR enviroment variable.")
        sys.exit(0)

    r = ReadBrat()
    # r.process(data_dir) # read and return a list of list of tokens
    r.toColumn(data_dir, output_dir)

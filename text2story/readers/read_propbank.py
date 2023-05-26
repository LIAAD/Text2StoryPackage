import os
import nltk
import argparse

from nltk.corpus import treebank, propbank
from nltk.chunk import tree2conlltags
from nltk.corpus.reader.propbank import PropbankChainTreePointer, PropbankSplitTreePointer

from text2story.core.narrative import Narrative
from text2story.readers.utils import convert_role_propbank_to_lirics, get_index

POS_NOUN = ["NN","NNP","NNS","NNP","NNPS","DT","JJ","JJS","JJR","PRP"]
TIME_EX = ["RB","RBR","RBS","IN","BY","CD"]

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


class ReadPropBank:

    def __init__(self):
        nltk.download("propbank")
        nltk.download("treebank")


    def convert_to_brat(self, lang, outputdir):

        self.build_txt_files(outputdir)
        self.build_ann_files(lang, outputdir)


    def get_index_pointer(self, pb_pointer):

        index_lst = []

        if isinstance(pb_pointer, PropbankChainTreePointer) or\
                isinstance(pb_pointer, PropbankSplitTreePointer):
            # a pointer can have discontinues elements, so it returns a list
            pieces_index = []

            # What to do with a chain?
            # iterate in the pieces and add to a list
            for piece in pb_pointer.pieces:
                pieces_index = pieces_index + self.get_index_pointer(piece)
            return pieces_index
        else:
            return [pb_pointer.wordnum]

    def count_offset(self, wordnum, sent_lst):

        count_offset = 0
        idx = 0
        while idx < wordnum:
            tok, pos, ne = sent_lst[idx]
            count_offset = count_offset + len(tok)
            idx = idx + 1

        return count_offset

    def get_time_args(self, argloc, ne_sent):
        index_lst = self.get_index_pointer(argloc)
        start = index_lst[0]

        if len(index_lst) > 1:
            end = index_lst[-1]
        else:
            end = len(ne_sent)

        count_offset = 0
        idx = 0
        # in this methodology, we have roughly the 
        # position of the token in text, in characters
        while idx < start:
            tok, pos, ne = ne_sent[idx]
            count_offset = count_offset + len(tok) 
            idx = idx + 1

        res = []
        while start < end:
            tok, pos, ne = ne_sent[start]
            if pos in TIME_EX:
                start = start + 1
            else:
                break


        return index_lst[0], start, count_offset

    def get_noun_args(self, argloc, ne_sent):
        index_lst = self.get_index_pointer(argloc)
        start = index_lst[0]

        if len(index_lst) > 1:
            end = index_lst[-1]
        else:
            end = len(ne_sent)

        count_offset = 0
        idx = 0
        # in this methodology, we have roughly the 
        # position of the token in text, in characters
        while idx < start:
            tok, pos, ne = ne_sent[idx]
            count_offset = count_offset + len(tok) 
            idx = idx + 1

        res = []
        while start < end:
            tok, pos, ne = ne_sent[start]
            if pos in POS_NOUN:
                start = start + 1
            else:
                break


        return index_lst[0], start, count_offset


    def _get_pointer_content(self, pointer, sent):


        if isinstance(pointer, PropbankChainTreePointer) or\
                isinstance(pointer, PropbankSplitTreePointer):
            # a pointer can have discontinues elements, so it returns a list
            pieces_txt = ""

            # What to do with a chain?
            # iterate in the pieces and add to a list
            for piece in pointer.pieces:
                pieces_txt = pieces_txt + self._get_pointer_content(piece, sent) + " "
            return pieces_txt.rstrip()
        else:
            tok_id = pointer.wordnum 
            tok_txt = sent[tok_id]
            return tok_txt


    def add_participant_narrative(self, narrative, pointer_content, offset, participant_set):

            offset_start = narrative.text.find(pointer_content, offset)
            offset_end = offset_start + len(pointer_content)

            if (offset_start, offset_end) not in participant_set:
                actor_id =  narrative._add_actor((offset_start, offset_end))
                participant_set.add((offset_start, offset_end))

                return actor_id, offset_end
            else:
                return None, -1

    def add_time_narrative(self, narrative, pointer_content, offset, time_set):

            offset_start = narrative.text.find(pointer_content, offset)
            offset_end = offset_start + len(pointer_content)

            if (offset_start, offset_end) not in time_set:
                time_id =  narrative._add_time((offset_start, offset_end))
                time_set.add((offset_start, offset_end))

                return time_id, offset_end
            else:
                return None, -1


    def add_arg_to_narrative(self, narrative, pointer_content, offset):

        if isinstance(pointer_content, tuple):
            tok_txt, ne_label = pointer_content

            offset_start = narrative.text.find(tok_txt, offset)
            offset_end = offset_start + len(tok_txt)
            actor_id = None
            if ne_label != 'O':
                actor_id = narrative._add_actor((offset_start, offset_end))

            return actor_id, offset_end
        else:
            # it is a list of tuples
            actor_lst = []
            current_offset = 0
            for content in pointer_content:
                actor_id, current_offset = self.add_arg_to_narrative(narrative, content, \
                        current_offset)
                if actor_id is not None:
                    actor_lst.append(actor_id)

            return actor_lst, current_offset

    def build_ann_files(self, lang, outputdir):

        narrative_instances = {}
        # the heuristic to map participants can add 
        # duplicate participants/events. Then this map helps to eliminate the 
        # repetead participants/events
        participants_instances = {} 
        events_instances = {} 
        times_instances = {} 

        for inst in propbank.instances():


            nameid = os.path.splitext(inst.fileid)[0]
            fname = os.path.join(outputdir, "%s.txt" % nameid)
            if not(os.path.exists(fname)):
                continue

            if nameid not in narrative_instances:
                with open(fname, "r") as fd:
                    txt = fd.read()
                    narrative_instances[nameid] = Narrative(lang, txt,"0000-00-00")
            narrative_inst = narrative_instances[nameid]
            # now, it is going to look for the arguments in the narrative text
            # to add events, actors, and srlinks 

            # id sentence of this propbank instance
            sent_id = inst.sentnum 
            # tokenized sentences from the txt file associated with this sentence
            sent_lst = treebank.sents(inst.fileid)  
            # pos tagged tokenized sentences from the txt file associated with this sentence
            tag_sent_lst = treebank.tagged_sents(inst.fileid)
            # entities of the tokenized sentences from the txt file associated with this sentence
            ne_sent = tree2conlltags(nltk.ne_chunk(tag_sent_lst[sent_id]))
            # the whole narrative text associated with the sentence sent_id
            narrative_txt = narrative_instances[nameid].text


            # add th ehead of the srl as event. Is that ok?
            event_current_offset = self.count_offset(inst.wordnum, ne_sent)
            event_txt = sent_lst[inst.sentnum][inst.wordnum]
            offset_event_start = narrative_txt.find(event_txt, event_current_offset)
            offset_event_end = offset_event_start + len(event_txt)

            if inst.fileid not in participants_instances:
                participants_instances[inst.fileid] = set()

            if inst.fileid not in events_instances:
                events_instances[inst.fileid] = set()

            if inst.fileid not in times_instances:
                times_instances[inst.fileid] = set()

            if (offset_event_start, offset_event_end) not in events_instances[inst.fileid]:
                event_id = narrative_instances[nameid]._add_event((offset_event_start, offset_event_end))
                events_instances[inst.fileid].add((offset_event_start, offset_event_end))
            else:
                event_id = None


            for argloc, argid in inst.arguments:

                lirics_role = convert_role_propbank_to_lirics(inst.roleset, argid)

                # start token index, end token index and
                # the position (char) in the file to start looking for the participants,
                # and events
                if argid == "ARGM-TMP": # it is a time expression
                    start, end, current_offset = self.get_time_args(argloc, ne_sent)
                    pointer_content = " ".join(sent_lst[sent_id][start:end])
                    time_id, time_offset = self.add_time_narrative(narrative_instances[nameid],\
                                    pointer_content,\
                                    0,\
                                    times_instances[inst.fileid])
                else:
                    start, end, current_offset = self.get_noun_args(argloc, ne_sent)
                    pointer_content = " ".join(sent_lst[sent_id][start:end])


                    if len(pointer_content) > 0:


                        actor_id, current_offset = self.add_participant_narrative(narrative_instances[nameid],\
                                pointer_content,\
                                current_offset,\
                                participants_instances[inst.fileid])

                        if actor_id is not None:
 
                            if lirics_role is not None and event_id is not None:
                                narrative_instances[nameid]._add_srlink(actor_id, event_id, "%s" % lirics_role)


        for nameid in narrative_instances:
            ann_fname = os.path.join(outputdir,"%s.ann" % nameid)

            with open(ann_fname, "w") as fd_ann:
                ann_txt = narrative_instances[nameid].ISO_annotation()
                fd_ann.write(ann_txt)



    def build_txt_files(self, outputdir):

        for fid in treebank.fileids():

            nameid = os.path.splitext(fid)[0]
            fname = os.path.join(outputdir, "%s.txt" % nameid)

            with open(fname, "w") as fd:

                raw_str = ""

                for sent in treebank.sents(fid):

                    for tok in sent:
                        if tok in ".,?!;":
                            raw_str = raw_str.rstrip()
                        raw_str = raw_str + tok + " "

                raw_str = raw_str.rstrip()

                fd.write(raw_str)

def main(lang, outputdir):

    reader = ReadPropBank()
    reader.convert_to_brat(lang, outputdir)

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Convert PropBank corpus to brat format')

    my_parser.add_argument("language", action='store', type=str, help="The language to convert the propbank (pt or en).")
    my_parser.add_argument("datadir", action='store', type=dir_path, help="The directory where the brat files will be stored.")

    args = my_parser.parse_args()
    main(args.language, args.datadir)

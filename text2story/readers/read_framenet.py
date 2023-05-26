import os
import nltk
import argparse
import time
import json

from nltk.corpus import framenet

from text2story.core.narrative import Narrative

PARTICIPANT_TYPES = ["focal_participant","participant","participant_1",\
        "participant_2","protagonist","victim","agent","instrument",\
        "theme","leader","name","member","creator","buyer",\
        "individuals","author","conqueror","sleeper","goal",\
        "weapon","resource","sent_item","owner","possessor",\
        "experiencer","user","student","substance"]

OTHER_ELEMENTS_TYPE = ["location"]

#POSSIBLE_PARTIPANTS = ["conqueror","medium","figure","cognizer","sleeper"]

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

class ReadFramenet:

    def __init__(self):
        nltk.download("framenet_v17")

    def convert_to_brat(self, lang, outputdir):

        self.build_txt_files(outputdir)
        self.build_ann_files(lang, outputdir)

    def add_arg_to_narrative(self, arg, narrative):

        # events
        event_offset_set = set()


        # it gets the event of the current frame
        event_txt = ""
        for (start, end) in arg.Target:

            event_offset_set.add((start, end))
            tok_txt = arg.text[start:end]
            event_txt = event_txt + tok_txt + " "

        event_txt = event_txt.rstrip()
        offset_start = narrative.text.find(event_txt, 0)
        event_id = None
        if offset_start < 0:
            # probably it is a non continous event
            print("Warning: Event index not located.")
        else:
            offset_end = offset_start + len(event_txt)
            event_id = narrative._add_event((offset_start, offset_end))


        # coletar aqui as relacoes, existe o type_relation
        # o temos acima o evento que conecta os atores abaixo. certo?
        # participants (FE)
        with open("fn-lirics.json","r") as fd:
            map_fn_lirics = json.load(fd)

        current_offset = 0
        for (start, end, type_relation) in arg.FE[0]:

            # it is an event already
            if (start,end) in event_offset_set:
                continue

            tok_txt = arg.text[start:end]
            offset_start = narrative.text.find(tok_txt, 0)
            offset_end = offset_start + len(tok_txt)
            current_offset = offset_end

            type_relation_low = type_relation.lower()


            # it is a type of event
            if type_relation_low.endswith("event") or\
                type_relation_low.startswith("event")    :
                event_id = narrative._add_event((offset_start, offset_end))
            else:

                if type_relation_low == "time":
                    time_id = narrative._add_time((offset_start, offset_end))

                if type_relation_low in OTHER_ELEMENTS_TYPE:
                    continue

                actor_id = None
                # it is an participant that is the participant type group
                if type_relation_low in PARTICIPANT_TYPES:
                    actor_id = narrative._add_actor((offset_start, offset_end))

                    # it is participant that can be mapped to the lirics type
                    # nao necessariamente, a condicao acima e a condicao abaixo devem ocorrer. Certo?
                    if type_relation_low in map_fn_lirics:
                        narrative._add_srlink(actor_id, event_id, type_relation_low)



    def build_ann_files(self, lang, outputdir):

        # ignore the annotations of exemplars since they are just 
        # examples, and we want to extract narrative from full text files
        ann_lst = framenet.annotations(exemplars=False)
        doc_lst = framenet.docs()

        # iterate through annotations, and map each document to a 
        # a narrative object
        map_narrative_doc = {}
        for ann in ann_lst:

            # get the document associated with  this annotation
            docid = ann.sent.docID
            doc = framenet.doc(docid)

            filename = os.path.splitext(doc.filename)[0] + ".txt"


            # it gets or creates the narrative object associated with this 
            # annotation 
            if filename in map_narrative_doc:
                narrative_inst = map_narrative_doc[filename]
            else:
                full_filename = os.path.join(outputdir, filename)

                txt = ""
                with open(full_filename, "r") as fd:
                    txt = fd.read()

                narrative_inst = Narrative(lang, txt,"0000-00-00")
                narrative_inst.text = txt

                map_narrative_doc[filename] = narrative_inst


            self.add_arg_to_narrative(ann, narrative_inst)

            filename_ann = os.path.splitext(doc.filename)[0] + ".ann"
            full_filename_ann = os.path.join(outputdir, filename_ann)

            narrative_ann_str = narrative_inst.ISO_annotation()

            with open(full_filename_ann, "w") as fd:
                fd.write(narrative_ann_str)
                


    def build_txt_files(self, outputdir):

        doc_lst = framenet.docs()

        for doc in doc_lst:

            doc_txt = ""
            for sent in doc.sentence:
                doc_txt = doc_txt + sent.text

            filename = os.path.splitext(doc.filename)[0] + ".txt"
            filename = os.path.join(outputdir, filename)

            with open(filename, "w") as fd:
                fd.write(doc_txt)


def main(lang, outputdir):

    reader = ReadFramenet()
    reader.convert_to_brat(lang, outputdir)

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Convert Framenet corpus to brat format')

    my_parser.add_argument("language", action='store', type=str, help="The language to convert the framenet (pt or en).")
    my_parser.add_argument("datadir", action='store', type=dir_path, help="The directory where the brat files will be stored.")

    args = my_parser.parse_args()
    main(args.language, args.datadir)

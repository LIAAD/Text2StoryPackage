import os
import sys
import xml.etree.ElementTree as ET
import argparse
import re
import string

from text2story.readers.read import Read
from text2story.readers.token_corpus import TokenCorpus

from text2story.core.narrative import Narrative
from text2story.core.entity_structures import ActorEntity, EventEntity, TimeEntity
from text2story.readers.read_brat import ReadBrat

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

class ReadECB(Read):
    def __init__(self) -> None:
        self.root = None
        self.tokens = []
        self.actors = []
        self.events = []
        self.times = []
        self.srlinks = []

        # given an id of a sentence, maps to the entities (event, actors, times)
        # of that sentence
        self.sentences_entities = {} 

        self.narrative = Narrative("en", "", "2021-08-08")
        self.read_brat = ReadBrat()

        self.idx_actor = 0
        self.idx_event = 0
        self.idx_time = 0

    def process(self, data_dir):
        return self.read_brat.process(data_dir)

    def process_file(self, data_file):
        return self.read_brat.process_file(data_file)

    def convert_xml_to_brat(self, xml_path, data_dir):
        files_list = []

        for dir_path, dirs, files in os.walk(xml_path):
            for f in files:
                files_list.append({
                    'file': f.replace('.xml', ''),
                    'path': dir_path
                })

        for index, file_dict in enumerate(files_list):
            print(f'Processing file {file_dict["file"]} No: {index + 1} out of {len(files_list)}')
            self.generate_ann_txt_file(file_dict['path'], file_dict['file'], data_dir)

    def generate_ann_txt_file(self, file_path, file_name, data_dir):
        tree = ET.parse(f'{file_path}/{file_name}.xml')
        self.root = tree.getroot()

        txt_file = open(f'{data_dir}/{file_name}.txt', 'w')
        ann_file = open(f'{data_dir}/{file_name}.ann', 'w')

        self.erase_data_structures()

        narrative_text = self.reconstruct_narrative(txt_file)
        self.findall_actors(narrative_text)
        self.findall_events(narrative_text)
        self.findall_time(narrative_text)

        self.findall_relations(narrative_text)

        self.fill_narrative()
        self.construct_ann(self.narrative.ISO_annotation(), ann_file)

    def reconstruct_narrative(self, txt_file):


        for token in self.root.iter('token'):
            self.tokens.append(TokenCorpus(token.text.strip(), token.attrib['t_id'], token.attrib['sentence']))

        current_sentence = 0
        char_counter = 0
        string_to_file = ""

        for token in self.tokens:

            if int(token.sentence) > current_sentence:
                current_sentence = current_sentence + 1
                string_to_file += "\n"

            if token.text != ";" and token.text != "," and token.text != "'" and token.text != ".":
                string_to_file += ' '
                char_counter += 1

            string_to_file += token.text
            # TODO: I dont understand why we need +1 but we need it
            token.offset = char_counter + 1
            char_counter += len(token.text)


        txt_file.write(string_to_file)
        txt_file.flush()
        txt_file.close()

        return string_to_file

    def findall_actors(self, narrative_text):

        type_actor = [("HUMAN_PART_PER","Noun","Per"),\
                ("HUMAN_PART_ORG","Noun","Org"),\
                ("HUMAN_PART_GPE","Noun","Loc"),\
                ("HUMAN_PART_VEH","Noun","Obj"),\
                ("NON_HUMAN_PART","Noun","Obj"),\
                ("NON_HUMAN_PART_GENERIC","Noun","Obj"),\
                ("HUMAN_PART_FAC","Noun","Other"),\
                ("HUMAN_PART_MET","Noun","Other")]

        for atype, apos, aclass in type_actor:
            self.idx_actor = 0
            self.findall_actor_type(narrative_text, atype, apos, aclass)

    def findall_actor_type(self, narrative_text, atype, apos, aclass):
        actor_lst = []
        for actor in self.root.iter(atype):

            actor_token_index_list = []

            for anchor in actor.findall('token_anchor'):
                t_id = anchor.attrib['t_id']
                actor_token_index_list.append(int(t_id))

            if len(actor_token_index_list) > 0:
                actor_lst.append(actor_token_index_list)

        actor_lst.sort(key=lambda x: int(x[0]))
        #import pdb
        #pdb.set_trace()

        for actor_token_index_list in actor_lst:
            self.actors.append(self.process_actor_token_list(actor_token_index_list, apos, aclass, narrative_text))


    def tokenlist2text(self, token_lst, idx_start, text):
        """
        Given a list of tokens, find the string of these tokens in the text parameter.
        """
        if len(token_lst) > 0:
            white_space_re = re.compile("\s")
            fst_token = token_lst[0].text

            # it going to start the search by the first token
            offset = text.find(fst_token, idx_start)
            idx_text = offset + len(fst_token)

            idx_tok = 1

            while idx_tok < len(token_lst):
                current_token = token_lst[idx_tok].text

                # ignore punctuation and every kind of space (tab, new line, etc)
                while idx_text < len(text):
                    m = white_space_re.search(text[idx_text])
                    if m is None and text[idx_text] not in string.punctuation:
                        break
                    idx_text = idx_text + 1

                j = 0    
                # ignore punctuation and every kind of space (tab, new line, etc)
                while j < len(current_token):
                    m = white_space_re.search(current_token)
                    if m is None and current_token not in string.punctuation:
                        break
                    j = j + 1
 
                k = idx_text
                # compare the current token to the current token in the text
                while j < len(current_token) and current_token[j] == text[k]:
                    j = j + 1
                    k = k + 1

                # if there is no match between one of the tokens in the token
                # list, return None
                if j != len(current_token):
                    return
                else:
                    idx_text = k


                idx_tok = idx_tok + 1

            return text[offset:idx_text]
 
    def findall_relations(self, narrative_text):

        if len(self.sentences_entities) == 0:
            print("Warning: There is no entity found to build relations.")
            return
        else:
            for sent_id in self.sentences_entities:
                if "actor" in self.sentences_entities[sent_id]:
                    actor_lst = self.sentences_entities[sent_id]["actor"]
                    event_lst = self.sentences_entities[sent_id]["event"]

                    actor_txt = ""
                    for a in actor_lst:
                        actor_txt = actor_txt + " " + a.text

                    event_txt = ""
                    for a in event_lst:
                        event_txt = event_txt + " " + a.text

                    print("--- SENT %s ---" % sent_id)
                    print("ACTORS %s" % actor_txt)
                    print("EVENT %s" % event_txt)
                    print()



    def process_actor_token_list(self, index_list, lexical_head, actor_type, narrative_text):
        raw_str = ""
        tok_lst = []

        for index in index_list:

            # collecting entities per sentence in order to 
            # detect relation between them
            sent_id = self.tokens[index - 1].sentence
            if sent_id in self.sentences_entities:
                if "actor" in self.sentences_entities[sent_id]:
                    self.sentences_entities[sent_id]["actor"].append(self.tokens[index - 1])
                else:
                    self.sentences_entities[sent_id]["actor"] = [self.tokens[index - 1]]
            else:
                self.sentences_entities[sent_id] = {"actor":[self.tokens[index - 1]]}

            if self.tokens[index - 1].text in ".,;?!":
                raw_str = raw_str.rstrip()
            raw_str += self.tokens[index - 1].text + " "
            tok_lst.append(self.tokens[index - 1])

        raw_str = raw_str.rstrip()


        offset_start = narrative_text.find(raw_str, self.idx_actor)
        if offset_start < 0:
            token_txt = self.tokenlist2text(tok_lst, self.idx_actor, narrative_text)
            if token_txt is not None:
                offset_start = narrative_text.find(token_txt, self.idx_actor)


        self.idx_actor = offset_start + len(raw_str)
        #return ActorEntity(raw_str, (
        #    self.tokens[index_list[0] - 1].offset, self.tokens[index_list[0] - 1].offset + len(raw_str)),
        #                   lexical_head,
        #                   actor_type)
        return ActorEntity(raw_str, (
                           offset_start, offset_start + len(raw_str)),
                           lexical_head,
                           actor_type)

    def findall_events(self, narrative_text):

        type_events = [('ACTION_OCCURRENCE',"Occurrence","Pos"),\
                ('NON_ACTION_OCCURRENCE',"Occurrence","Neg"),\
                ('ACTION_PERCEPTION','Perception','Pos'),\
                ('NON_ACTION_PERCEPTION','Perception','Neg'),\
                ('ACTION_REPORTING',"Reporting","Pos"),\
                ('NON_ACTION_REPORTING','Reporting','Neg'),\
                ('ACTION_ASPECTUAL','Aspectual','Pos'),\
                ('NON_ACTION_ASPECTUAL','Aspectual','Neg'),\
                ('ACTION_STATE','State','Pos'),\
                ('NON_ACTION_STATE','State','Neg'),\
                ('ACTION_CAUSATIVE','Causative','Pos'),\
                ('NON_ACTION_CAUSATIVE','Causative','Neg'),\
                ('ACTION_GENERIC','Generic','Pos'),\
                ('NON_ACTION_GENERIC','Generic','Neg')]

        for etype, eclass, polarity in type_events:
            self.idx_event = 0
            self.findall_event_type(narrative_text, etype, eclass, polarity)

    def findall_event_type(self, narrative_text, etype, eclass, polarity):

        event_action_lst = []
        for event in self.root.iter(etype):

            event_token_index_list = []

            for anchor in event.findall('token_anchor'):
                t_id = anchor.attrib['t_id']
                event_token_index_list.append(int(t_id))

            if len(event_token_index_list) > 0:
                event_action_lst.append(event_token_index_list)

        event_action_lst.sort(key=lambda x:x[0])
        for event_token_index_list in event_action_lst:
            event_obj = self.process_event_token_list(event_token_index_list, eclass, polarity, narrative_text)
            # this only happens when there are two events with the same token
            # so to avoid it, ignore one of the events
            if event_obj.character_span[0] > 0:
                self.events.append(event_obj)


    def process_event_token_list(self, index_list, event_class, polarity, narrative_text):

        # TODO: create a solution when the event has not adjacent tokens
        raw_str = ""
        tok_lst = []
        for index in range(index_list[0], index_list[-1] + 1):

            # collecting entities per sentence in order to 
            # detect relation between them
            sent_id = self.tokens[index - 1].sentence
            if sent_id in self.sentences_entities:
                if "event" in self.sentences_entities[sent_id]:
                    self.sentences_entities[sent_id]["event"].append(self.tokens[index - 1])
                else:
                    self.sentences_entities[sent_id]["event"] = [self.tokens[index - 1]]
            else:
                self.sentences_entities[sent_id] = {"event":[self.tokens[index - 1]]}

            if self.tokens[index - 1].text in ".,;?!":
                raw_str = raw_str.rstrip()
            raw_str += self.tokens[index - 1].text + " "
            tok_lst.append(self.tokens[index - 1])

        raw_str = raw_str.rstrip()

        offset_start = narrative_text.find(raw_str, self.idx_event)
        if offset_start < 0:
            token_txt = self.tokenlist2text(tok_lst, self.idx_event, narrative_text)
            if token_txt is not None:
                offset_start = narrative_text.find(token_txt, self.idx_event)

        self.idx_event = offset_start + len(raw_str)

        return EventEntity(raw_str,
                           (offset_start,
                            offset_start + len(raw_str)),
                           event_class, polarity)

    def findall_time(self, narrative_text):
        time_types = [("TIME_DATE","Date","Date"),\
                ("TIME_THE_DAY","Time","Time"),\
                ("TIME_DURATION","Duration","Duration"),\
                ("TIME_REPETITION","Time","Set")]

        for type_t, value, timex in time_types:
            self.idx_time = 0
            self.findall_time_type(type_t, value, timex, narrative_text)

    def findall_time_type(self,time_t, value, timex, narrative_text):

        time_lst = []
        for time in self.root.iter(time_t):

            time_token_index_list = []

            for anchor in time.findall('token_anchor'):
                t_id = anchor.attrib['t_id']
                time_token_index_list.append(int(t_id))

            if len(time_token_index_list) > 0:
                time_lst.append(time_token_index_list)

        time_lst.sort(key=lambda x:x[0])

        for time_token_index_list in time_lst:
            self.times.append(self.process_time_token_list(time_token_index_list, value, timex, narrative_text))

    def process_time_token_list(self, index_list, value, timex_type, narrative_text):
        raw_str = ""

        # SOmetimes the tokens are not connected by space. Usually, 
        # this happens with punctuation. Then, it checks if the token is 
        # a punctuation symbol, then remove the space before the punctuation.

        for index in range(index_list[0], index_list[-1] + 1):

            # collecting entities per sentence in order to 
            # detect relation between them
            sent_id = self.tokens[index - 1].sentence
            if sent_id in self.sentences_entities:
                if "time" in self.sentences_entities[sent_id]:
                    self.sentences_entities[sent_id]["time"].append(self.tokens[index - 1])
                else:
                    self.sentences_entities[sent_id]["time"] = [self.tokens[index - 1]]
            else:
                self.sentences_entities[sent_id] = {"time":[self.tokens[index - 1]]}

            if self.tokens[index - 1].text in ".,;?!":
                raw_str = raw_str.rstrip()
            raw_str += self.tokens[index - 1].text + " "

        raw_str = raw_str.rstrip()

        offset_start = narrative_text.find(raw_str, self.idx_time)
        self.idx_time = offset_start + len(raw_str)

        return TimeEntity(raw_str,
                (offset_start, offset_start + len(raw_str)), value,
                          timex_type)

    def fill_narrative(self):
        actors_dict = {}
        events_dict = {}
        times_dict = {}

        index = 0

        for actor in self.actors:
            actors_dict['T' + str(index)] = actor
            index = index + 1

        for event in self.events:
            events_dict['T' + str(index)] = event
            index = index + 1

        for time in self.times:
            times_dict['T' + str(index)] = time
            index = index + 1

        self.narrative.actors = actors_dict
        self.narrative.events = events_dict
        self.narrative.times = times_dict

    def construct_ann(self, iso_annotation, ann_file):
        ann_file.write(iso_annotation)
        ann_file.flush()
        ann_file.close()

    def to_column(self, data_dir, output_dir):
        return self.read_brat.toColumn(data_dir, output_dir)

    def erase_data_structures(self):
        self.tokens = []
        self.actors = []
        self.events = []
        self.times = []
        self.narrative = Narrative("en", "", "2021-08-08")


def main(xml_dir: str = None, data_dir: str = None, output_dir: str = None):

    r = ReadECB()
    r.convert_xml_to_brat(xml_dir, data_dir)
    # r.to_column(data_dir, output_dir)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Read ECB+ corpus')

    my_parser.add_argument("xmldir", action='store', type=dir_path, help="The directory that contains the xml file (ECB+ format).")
    my_parser.add_argument("datadir", action='store', type=dir_path, help="The directory where the brat files will be stored.")

    args = my_parser.parse_args()
    main(args.xmldir, args.datadir)



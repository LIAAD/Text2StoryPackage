import re
import sys

from text2story.readers.token_corpus import TokenCorpus
from text2story.core.utils import join_tokens

class NodeEvent:

    def __init__(self, event_txt, event_class, event_idx, event_arg, event_sent, id_ann = None):
        self.txt = event_txt
        self.class_type = event_class
        self.idx = event_idx 
        self.argn = event_arg
        self.sent_id = event_sent
        self.id_ann = id_ann

class EdgeEvent:

	def __init__(self, edge_type):
		self.edge_type = edge_type
		self.nodes = []

	def add_node(self, event_txt, event_class, event_idx, event_arg, event_sent, id_ann = None):
		self.nodes.append(NodeEvent(event_txt, event_class, event_idx, event_arg, event_sent, id_ann))

def sieve_bbubbles_events(event_lst, type_big_bubble):
    """
    A sieve for events that has temporal links of identity types
    """

    ans = {}
    not_ans = {}

    for id_ann in event_lst:
        
        for tok in event_lst[id_ann]:


            for attr_item in tok.attr:
 
                ann_type = attr_item[0]
                attr_map = attr_item[1]

                if "Class" not in attr_map:
                    continue

                if attr_map["Class"] == type_big_bubble:
                    is_big_bubble = False

                    for r in tok.relations:
                        #print(r.rel_type, id_ann, tok.text)
                        if is_event(r.toks[0]):
                            if r.rel_type == "TLINK_identity" and\
                                id_ann not in ans and type_big_bubble in get_event_type(r.toks[0]):

                                #toks_ids = [rel_toks.id_ann for rel_toks in r.toks]
                                #print("id_ann: ",id_ann, toks_ids)
                                ans[id_ann] = event_lst[id_ann]
                                is_big_bubble = True
                                break

                    if not(is_big_bubble):
                        if id_ann in not_ans:
                            not_ans[id_ann].append(tok)
                        else: 
                            not_ans[id_ann] = [tok]
                    
                else:
                    if id_ann in not_ans:
                        not_ans[id_ann].append(tok)
                    else: 
                        not_ans[id_ann] = [tok]

    return ans, not_ans


def get_reporting_events(tok_lst):

    event_lst = {}


    for i in range(len(tok_lst)):
        tok = tok_lst[i]


        for attr_item in tok.attr:

            ann_type = attr_item[0]
            attr_map = attr_item[1]

            if "Class" in attr_map and attr_map["Class"] == "Reporting": # it is an event token

                for id_ann in tok.id_ann:

                    if id_ann in event_lst:
                        event_lst[id_ann].append(tok)
                    else:
                        event_lst[id_ann] = [tok]
                break
    return event_lst

def get_offset_lst(tok_lst):
    res = []
    for tok in tok_lst:
        res.append((tok.offset, tok.offset + len(tok.text), tok.id_ann))
    return res
def get_first_event(tok_lst):
    offset_lst = get_offset_lst(tok_lst)
    offset_lst.sort(key=lambda x:x[0])

    return offset_lst[0]
def get_last_event(tok_lst):
    offset_lst = get_offset_lst(tok_lst)
    offset_lst.sort(key=lambda x: x[0])
    return offset_lst[-1]

def get_nested_events_nonlinked(event_lst, nested_events_ids):
    """
    Get nested events in reporting events that are not directly linked to it
    @param event_lst: the list of events that are nested in a given reporting event
    @param nested_events_ids: the ids of events that are directly connected to reporting events
    @return:
    """
    # For each event E in event_lst, check if there is an Et that
    # it is the same sentence that E, and is linked with TLINK_identity
    # Also, Et should not be a reporting event
    X =[]
    for E in event_lst:
        for rel in E.relations:
            if rel.rel_type == "TLINK_identity":
                Et = rel.toks[0]
                if Et.sent_id == E.sent_id and\
                        Et.get_attr_value('Class') != 'Reporting' and\
                        Et.id_ann[0] not in nested_events_ids:
                    X.append(Et)
                    nested_events_ids.add(Et.id_ann[0])
    return X

def get_embedded_events(doc):

    event_lst = get_all_events_sent(doc)

    embedded_events_lst = {}
    embedded_events_ids = set()
    count_supressed_reporting_events = 0

    for sent_id in event_lst:

        reporting_events = []
        embedded_events = []

        for event in event_lst[sent_id]:
            if event.get_attr_value("Class") == "Reporting":
                reporting_events.append(event)
            else:
                embedded_events.append(event)

        if len(reporting_events) > 1:
            # in this situation there is a chain of reporting events
            # some of them are concerning of story, and others are
            # related to
            print(f"{len(reporting_events)} reporting events in {sent_id}")
            count_supressed_reporting_events += len(reporting_events) - 1

            # TODO: for now just ignoring other reporting events and considering only one
            # considering the ROOT
            for e in embedded_events:
                embedded_events_ids.add(e.id_ann[0])
            embedded_events_lst[sent_id] = (reporting_events[0], embedded_events)


        elif len(reporting_events) == 1:
            for e in embedded_events:
                embedded_events_ids.add(e.id_ann[0])
            embedded_events_lst[sent_id] = (reporting_events.pop(), embedded_events)

    return embedded_events_lst, embedded_events_ids, count_supressed_reporting_events


def get_nested_events_reporting(doc):
    # for each id annotation get the related events
    event_lst = get_all_events(doc)
    nested_events = {}
    nested_events_ids = set()

    for event_id in event_lst:

        for event in event_lst[event_id]:
            event_class = event.get_attr_value('Class', 'Event')

            if event_class == 'Reporting':
                sent_id = event.sent_id

                # devo olhar somente para quem ele se relaciona? E se eu fizer a coleta por sentença?
                # fazer um get_all_events por sentença?
                for r in event.relations:
                    tok_rel = r.toks[0]
                    # print("-->", tok_rel.id_ann)
                    if tok_rel.is_type("Event"):
                        if tok_rel.get_attr_value('Class') == 'Reporting' and r.rel_type == 'TLINK_identity':
                            continue

                        rel_sent_id = tok_rel.sent_id
                        if rel_sent_id == sent_id and tok_rel.id_ann[0] not in nested_events_ids:
                            if event_id in nested_events:
                                nested_events[event_id].append(tok_rel)
                            else:
                                nested_events[event_id] = [tok_rel]
                            nested_events_ids.add(tok_rel.id_ann[0])

    # some events are nested in a reporting event, but are not directly linked to it
    for event_id in nested_events:
        non_linked = get_nested_events_nonlinked(nested_events[event_id], nested_events_ids)
        nested_events[event_id] += non_linked

    return event_lst, nested_events, nested_events_ids


def get_all_events_sent(tok_lst):
    """
    Get all events from a document and index them by the sentence number
    @param tok_lst:
    @return:
    """

    event_lst = {}
    event_id_lst = set()

    for i in range(len(tok_lst)):
        tok = tok_lst[i]

        if tok.is_type("Event") and tok.id_ann[0] not in event_id_lst:
            if tok.sent_id in event_lst:
                event_lst[tok.sent_id].append(tok)
            else:
                event_lst[tok.sent_id] = [tok]
            event_id_lst.add(tok.id_ann[0])

    return event_lst

def get_all_events(tok_lst):

    event_lst = {}


    for i in range(len(tok_lst)):
        tok = tok_lst[i]


        for attr_item in tok.attr:

            ann_type = attr_item[0]
            attr_map = attr_item[1]

            if "event" in ann_type.lower():

                for id_ann in tok.id_ann:

                    if id_ann in event_lst:
                        event_lst[id_ann].append(tok)
                    else:
                        event_lst[id_ann] = [tok]
                break
            	

    return event_lst

def get_txt_events(event_lst):

    event_txt_lst = []

    for id_ann in event_lst:
        text_tokens = [e.text for e in event_lst[id_ann]]
        event_txt = " ".join(text_tokens)
        event_txt_lst.append(event_txt)

    return event_txt_lst

        
def get_hist_report_events(tok_lst):

    event_lst = get_all_events(tok_lst)
    hist_reporting = {}
    for e in event_lst:

        rel_type = get_event_type(e)
        if "Reporting" in rel_type:
            tok_event = e.text.lower()
            if tok_event in hist_reporting:
                hist_reporting[tok_event] += 1
            else:
                hist_reporting[tok_event] = 1
    return hist_reporting

def map_events_bysent(tok_lst):

    event_lst = get_all_events(tok_lst)
    map_sent = {}
    for e in event_lst:

        primary_type = get_event_type(e)[0]

        if e.sent_id in map_sent:
            if primary_type in map_sent[e.sent_id]:
                map_sent[e.sent_id][primary_type] += 1
            else:
                map_sent[e.sent_id][primary_type] = 1
        else:
            map_sent[e.sent_id] = {primary_type:1}

    return map_sent

def is_event(event):

    for attr_item in event.attr:

        ann_type = attr_item[0]
        attr_map = attr_item[1]

        if ann_type.lower() == "event":
            return True
    return False


def get_event_type(event):

    event_type_lst = []

    
    for attr_item in event.attr:

        ann_type = attr_item[0]
        attr_map = attr_item[1]

        if "Class" in attr_map: 
            event_type_lst.append(attr_map["Class"])
                
    #print("-->", event_type_lst)
    return event_type_lst

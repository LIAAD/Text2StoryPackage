import re

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


def get_all_events(tok_lst):

    event_lst = {}


    for i in range(len(tok_lst)):
        tok = tok_lst[i]


        for attr_item in tok.attr:

            ann_type = attr_item[0]
            attr_map = attr_item[1]

            if "Class" in attr_map: # it is an event token

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

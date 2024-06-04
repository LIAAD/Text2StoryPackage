import json

from dataclasses import dataclass, field
from typing import List

from text2story.readers.token_corpus import TokenCorpus
from text2story.select.event import get_all_events, get_event_type
from text2story.select.event import sieve_bbubbles_events, is_event


def is_type_rel(type_rel, type_rel_lst):
    for t in type_rel_lst:
        if type_rel.startswith(t):
            return True
    return False


class Bubble:

    def __init__(self, event=None):
        # print("-->", event)
        self.event_chain = []
        if isinstance(event, list):

            self.event = event[0]
            if len(event) > 1:
                self.event_chain = event
        else:
            self.event = event  # event trigger as TokenCorpus
        # if event is not None:
        #    print("-->", event.id_ann)
        self.relations = []  # the edges to other bubbles (BubbleEdge)

        self.name = ""
        # a text span (type:[TokenCorpus]) that is connected 
        # by an SRLINK_agent to this event
        self.agent = Agent()

    def get_event_text(self):
        if self.event_chain == []:
            return self.event.text
        else:
            tok_txt = [tok.text for tok in self.event_chain]
            return " ".join(tok_txt)

    def get_id_ann(self):
        return self.event.id_ann

    def get_hist_relations(self):

        hist_rel = {}

        for bubble_rel in self.relations:
            if bubble_rel.edge_type in hist_rel:
                hist_rel[bubble_rel.edge_type] += 1
            else:
                hist_rel[bubble_rel.edge_type] = 1
        return hist_rel

    def add_agent_relations(self, map_ann, relations_set, type_rel_lst=[]):

        # this bubble does not have an agent object
        if len(self.agent.span) == 0:
            return

        # iterate through the relations of the agent of
        # the event in the bubble
        current_agent = self.agent
        while current_agent != None:

            for rel in current_agent.span[0].relations:

                for id_ann_rel in rel.toks[0].id_ann:

                    if rel.rel_type in type_rel_lst and rel.rel_id not in relations_set:

                        if id_ann_rel in map_ann:
                            object_rel = map_ann[id_ann_rel]

                            if isinstance(object_rel, Agent):

                                relation = RelationsAgent(object_rel,
                                                          rel.rel_type,
                                                          rel.rel_id)
                                relation.set_arg(rel.argn)

                                current_agent.relations_agent.append(relation)
                            else:
                                relation = RelationsBubbleAgent(object_rel,
                                                                rel.rel_type,
                                                                rel.rel_id)
                                relation.set_arg(rel.argn)
                                current_agent.relations_bubbles.append(relation)

                            relations_set.add(rel.rel_id)
            current_agent = current_agent.next_agent

    def add_relations(self, map_ann, relations_set, type_rel_lst=[]):

        for rel in self.event.relations:
            self_event_type = get_event_type(self.event)

            for idx_rel, id_ann_rel in enumerate(rel.toks[0].id_ann):
                ann_type_rel, attr_rel = rel.toks[0].attr[idx_rel]

                # getting the participant connected to this bubble
                if "Reporting" in self_event_type and \
                        (rel.rel_type.startswith("SRLINK")) and \
                        ann_type_rel == "Participant" and \
                        id_ann_rel == rel.arg_id and \
                        rel.toks[0].sent_id == int(self.event.sent_id):
                    # the last condition test if the id of this token chain if related
                    # to the current relation

                    agent_pointer = self.agent.add_agent(rel.toks)
                    id_ann_agent = rel.toks[0].id_ann[0]
                    map_ann[id_ann_agent] = agent_pointer

                    # print("Agent %s Event %s ID REL %s %d %s" % (id_ann_agent, self.get_id_ann(), rel.rel_id, idx_rel, rel.arg_id))
                # if it is a link between two reporting events, it ignores this link
                event_arg2_class = get_event_type(rel.toks[0])
                if "Reporting" in event_arg2_class and "Reporting" in self_event_type:
                    continue

                if id_ann_rel in map_ann and \
                        rel.rel_id not in relations_set and \
                        is_type_rel(rel.rel_type, type_rel_lst):

                    if not (isinstance(map_ann[id_ann_rel], Agent)):
                        #print("Event %s ID REL %s %d %s %s" % (self.get_id_ann(), rel.rel_id, idx_rel, rel.arg_id, rel.rel_type))
                        bubble_rel = map_ann[id_ann_rel]
                        relation = BubbleRelation(rel.rel_type, \
                                                  bubble_rel, \
                                                  rel.argn)

                        self.relations.append(relation)
                        relations_set.add(rel.rel_id)

                # prnt("==>", rel.rel_type)
                # print("-->", rel.rel_id)
                # print("-->",rel.argn)


@dataclass
class Agent:
    span: [TokenCorpus] = field(default_factory=list)
    relations_bubbles: ['RelationsBubbleAgent'] = field(default_factory=list)
    relations_agent: List['RelationsAgent'] = field(default_factory=list)
    name: str = ""
    next_agent: 'Agent' = None

    def add_agent(self, toks: [TokenCorpus]):
        if self.span != []:
            current_agent = self
            while current_agent.next_agent != None:
                current_agent = current_agent.next_agent

            current_agent.next_agent = Agent(span=toks)
            return current_agent.next_agent

        else:
            self.span = toks
            return self


@dataclass
class RelationsBubbleAgent:
    bubble: Bubble
    rel_type: str
    rel_id: str

    def set_arg(self, argn):

        # if the relation starts with the agent pointer,
        # then its type is out
        if argn == "arg1":
            self.out = True
        else:
            self.out = False


@dataclass
class RelationsAgent:
    agent: Agent
    rel_type: str
    rel_id: str

    def set_arg(self, argn):

        # if the relation starts with the agent pointer,
        # then its type is out
        if argn == "arg1":
            self.out = True
        else:
            self.out = False


class BigBubble:

    def __init__(self, event=None):
        # print("BigBubble")
        self.bubble_ = Bubble(event)
        self.little_bubbles = []
        self.idx = 0

    def add_bubble(self, event):
        # print("add_bubble")
        self.bubble_ = Bubble(event)

        return self.bubble_

    def add_little_bubble(self, event_lst):
        # print("add_little_bubble")
        little_bubble = Bubble(event_lst)
        self.little_bubbles.append(little_bubble)

        return little_bubble

    def sort_by_offset(self):
        # sort little bubbles by offset
        tmp_bubbles = [(b, b.event.offset) for b in self.little_bubbles]
        tmp_bubbles = sorted(tmp_bubbles, key=lambda x: x[1])
        self.little_bubbles = [b for (b, offset) in tmp_bubbles]

    def create_relations(self, map_ann, relations_set, type_rel_lst=[]):
        # add big bubble relations
        self.bubble_.add_relations(map_ann, relations_set, type_rel_lst)

        # create relations for the little bubbles
        for bubble in self.little_bubbles:
            bubble.add_relations(map_ann, relations_set, type_rel_lst)


class BubbleRelation:
    def __init__(self, edge_type, bubble, argn):
        self.edge_type = edge_type
        self.bubble_pointer = bubble

        # if the relation starts with bubble pointer,
        # then its type is out
        if argn == "arg1":
            self.out = True
        else:
            self.out = False


class BubbleMap:

    def __init__(self):
        self.map = {}  # sent_id -> BigBubble

    def collect_bubble_tokens(self, tok_lst, sent_id, map_ann):

        bubble_toks = []

        for tok in tok_lst:

            # only events
            if is_event(tok):

                # if this is a new relation, but with an already existing bubble
                # then we dont need to create a new bubble
                if tok.id_ann[0] in map_ann:
                    continue

                # it is a little bubble of this big bubble, if 
                # it is in the same sentence
                if tok.sent_id == sent_id:
                    bubble_toks.append(tok)

        return bubble_toks

    def build_map(self, tok_lst: [TokenCorpus], type_big_bubble: str, type_rel_lst: [str]) -> None:

        # for each id annotation get the related events
        all_event_lst = get_all_events(tok_lst)

        # get only eligible big bubble events (with temporal links identity events)
        event_lst, not_big_bubble_events = sieve_bbubbles_events(all_event_lst, type_big_bubble)

        event_lst = [event_lst[id_ann] for id_ann in event_lst]

        # map the bubble by the id of annotations, 
        # then, after create the bubble, we create the bubble relations
        map_ann = {}
        bubbles_ids = set()

        # create the big bubbles
        for event in event_lst:

            big_bubble = BigBubble(event)
            sent_id = event[0].sent_id
            id_ann = event[0].id_ann[0]

            if sent_id in self.map:
                new_sent_id = sent_id + 0.1
                while (new_sent_id in self.map):
                    new_sent_id += 0.1

                big_bubble.bubble_.event.sent_id = new_sent_id
                for tok in big_bubble.bubble_.event_chain:
                    tok.sent_id = new_sent_id

                self.map[new_sent_id] = big_bubble
            else:
                self.map[sent_id] = big_bubble

            map_ann[id_ann] = big_bubble
            bubbles_ids.add(id_ann)

        for event in event_lst:
            id_ann = event[0].id_ann[0]
            sent_id = event[0].sent_id

            for r in event[0].relations:
                rel_sent_id = r.toks[0].sent_id
                rel_id_ann = r.toks[0].id_ann[0]

                # if the event related to the current event is in the same sentence,
                # then add it as a little bubble. Otherwise, it is an event that is big bubble,
                # which was already created before or will be created as a little bubble
                # of another big bubble
                if int(rel_sent_id) == int(sent_id) and is_type_rel(r.rel_type, type_rel_lst):

                    # add little bubbles (only events)
                    bubble_toks = self.collect_bubble_tokens(r.toks, sent_id, map_ann)

                    if len(bubble_toks) > 0:
                        bbubble = self.map[sent_id]
                        little_bubble = bbubble.add_little_bubble(bubble_toks)
                        lbubble_id_ann = bubble_toks[0].id_ann[0]
                        bubbles_ids.add(lbubble_id_ann)

                        map_ann[lbubble_id_ann] = little_bubble
                        # print()

        # events that can be little bubbles, but are not linked to big bubbles
        # TODO: mas nao deveriam so eventos que estivessem ligados por tlink?
        for id_ann in not_big_bubble_events:

            sent_id = not_big_bubble_events[id_ann][0].sent_id
            # it is an already created bubble AND
            if id_ann in map_ann:
                continue

            # ...and there is no big bubble container, ignore this event
            if sent_id not in self.map:
                continue

            # print("-->", sent_id, id_ann, not_big_bubble_events[id_ann])

            bbubble = self.map[sent_id]

            little_bubble = bbubble.add_little_bubble(not_big_bubble_events[id_ann])
            map_ann[id_ann] = little_bubble
            bubbles_ids.add(id_ann)

        # keep track of the relations (id) already created
        relations_set = set()
        for sent_id in self.map:
            bbubble = self.map[sent_id]
            bbubble.create_relations(map_ann, relations_set, type_rel_lst)
            bbubble.sort_by_offset()

        # now, add the relations regarding the agents
        type_agent_rel = ["OLINK_objIdentity", "OLINK_partOf", "OLINK_memberOf", "SRLINK_agent", "SRLINK_medium"]
        for sent_id in self.map:
            bbubble = self.map[sent_id]
            bbubble.bubble_.add_agent_relations(map_ann, relations_set, type_agent_rel)
        return  all_event_lst, bubbles_ids

    def to_json(self, output_file_name):

        data = {}

        for sent_id in self.map:
            data[sent_id] = {}
            big_bubble = self.map[sent_id]

            # the info about the big bubble
            data[sent_id]["bubble"] = {}
            data[sent_id]["bubble"]["id"] = big_bubble.bubble_.event.id_ann[0]
            data[sent_id]["bubble"]["name"] = big_bubble.bubble_.name
            data[sent_id]["bubble"]["value"] = big_bubble.bubble_.get_event_text()

            # little bubbles information
            data[sent_id]["bubbles"] = {}

            for lbubble in big_bubble.little_bubbles:
                data[sent_id]["bubbles"]["bubble"] = {}
                data[sent_id]["bubbles"]["bubble"]["id"] = lbubble.event.id_ann[0]
                data[sent_id]["bubbles"]["bubble"]["name"] = lbubble.name
                data[sent_id]["bubbles"]["bubble"]["value"] = lbubble.get_event_text()

                for rel in lbubble.relations:
                    data[sent_id]["bubbles"]["bubble"]["relations"] = {}
                    if isinstance(rel.bubble_pointer, Bubble):
                        data[sent_id]["bubbles"]["bubble"]["relations"]["next"] = rel.bubble_pointer.event.id_ann[0]
                    else:
                        data[sent_id]["bubbles"]["bubble"]["relations"]["next"] = \
                        rel.bubble_pointer.bubble_.event.id_ann[0]
                    data[sent_id]["bubbles"]["bubble"]["relations"]["type"] = rel.edge_type
                    data[sent_id]["bubbles"]["bubble"]["relations"]["out"] = rel.out

            # relations
            for rel in big_bubble.bubble_.relations:
                data[sent_id]["relations"] = {}

                if isinstance(rel.bubble_pointer, Bubble):
                    data[sent_id]["relations"]["next"] = rel.bubble_pointer.event.id_ann[0]
                else:
                    data[sent_id]["relations"]["next"] = rel.bubble_pointer.bubble_.event.id_ann[0]

                data[sent_id]["relations"]["type"] = rel.edge_type
                data[sent_id]["relations"]["out"] = rel.out

        with open(output_file_name, "w") as fd:
            json.dump(data, fd, indent=4)

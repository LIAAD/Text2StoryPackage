# gather corpus stats
import os
import sys

from pathlib import Path

from markdown_it.common.entities import entities
from typer.models import NoneType

from text2story.readers.read_brat import  ReadBrat
from nltk import word_tokenize

import text2story as t2s
from text2story.core.exceptions import *
from text2story.core.entity_structures import *
from text2story.select.event import get_nested_events_reporting, get_embedded_events


def count_tokens(fd):
    return len(word_tokenize(fd.read()))

def count_elements(fd, element_type):

    count = 0
    for line in fd:
        if line[0] == "T":
            elements = line.split()
            if elements[1] == element_type:
                count += 1
        elif line[0] == "R":
            elements = line.split()
            if elements[1].startswith(element_type):
                count += 1

    return count

def get_attr_repr(attr_value):
    if type(attr_value) == list:
        return repr(attr_value)
    else:
        return attr_value

def build_stats_byelement(doc,  element_type="event", attr_filter={}, allowed_by_id=[]):

    attr_count = {}
    event_id_lst = set()
    allowed_id_lst = [id for (id, _) in allowed_by_id]

    for tok in doc:

        if tok.is_type(element_type):

            for attr_type, attr_map in tok.attr:
                if tok.id_ann[0] not in event_id_lst:
                    filter_flag = False

                    # checking if the annotation should be filtered of the statiscs
                    filtered_attrs = set(attr_filter.keys()) & set(attr_map.keys())
                    # if has some of filtered attributes, check if has the filtered value
                    for attr in filtered_attrs:
                        # all values should be filtered or the specified value only
                        if attr_filter[attr] is None or \
                                (attr_map[attr] == attr_filter[attr]) or \
                                (attr_map[attr] in attr_filter[attr]): # list of options
                            filter_flag = True
                            break
                    # if this annotation should not be filtered, lets counting its numbers
                    if not(filter_flag):
                        for attr in attr_map:
                            attr_map_value = get_attr_repr(attr_map[attr])


                            if attr in attr_count:
                                attr_count[attr][attr_map_value] = attr_count[attr].get(attr_map_value, 0) + 1
                            else:
                                attr_count[attr] = {attr_map_value:1}

                        event_id_lst.add(tok.id_ann[0])
                        break



    return attr_count, len(event_id_lst)

def build_stats_entity_attributes(doc,type="event",filter={},attr_lst=["Class","Event_Type",\
                                                               "Pos","Tense","Aspect",\
                                                               "VForm","Mood","Modality",\
                                                               "Polarity","Movement","Factuality"]):
    """
    It counts the combination of attributes of a given entity type
    @param doc: a document as list of TokenCorpus
    @param type: the of entity as string
    @param attr: a list of string of name of attributes to be counted
    @return:
    """
    attr_counts = {}
    entity_set = set()
    for tok in doc:
        for idx, (ann_type, attr_map) in enumerate(tok.attr):
            if ann_type.lower() == type and tok.id_ann[idx] not in entity_set:

                # filter the entity given an attribute name and value
                filtered_attrs = set(filter.keys()) & set(attr_map.keys())
                filter_entity_flag = False
                if len(filtered_attrs) != 0:
                    for fattr in filtered_attrs:
                        if attr_map[fattr] == filter[fattr] or\
                                attr_map[fattr] in filter[fattr] :
                            filter_entity_flag = True
                else:
                    filter_entity_flag = True # this is the None case, should be filtered as well
                if filter_entity_flag:
                    continue

                attr_combination = tuple([attr_map.get(attr_name, None) for attr_name in attr_lst])
                if attr_combination in attr_counts:
                    attr_counts[attr_combination] += 1
                else:
                    attr_counts[attr_combination] = 1
                entity_set.add(tok.id_ann[idx])

    return attr_counts



def build_stats_byrelation_tuple(doc,  type1="event", type2="participant"):
    """
    Given two type of entities, build the stats of their relation
    @param doc:
    @param element_type:
    @param attr_filter:
    @param allowed_by_id:
    @return:
    """

    rel_count = {}
    relation_id_lst = set()
    entity1_set = set()
    entity2_set = set()

    for tok1 in doc:

        for ann_type, attr_map in tok1.attr:
            if ann_type.lower() == type1:

                # if has attributes, has annotation, and thus relations
                for rel in tok1.relations:
                    if rel.rel_id not in relation_id_lst:
                        for tok2 in rel.toks:
                            for ann_type2, attr_map2 in tok2.attr:
                                if ann_type2.lower() == type2:
                                    if rel.rel_type in rel_count:
                                        rel_count[rel.rel_type] += 1
                                    else:
                                        rel_count[rel.rel_type] = 1
                                    entity1_set.add(tok1.id_ann[0])
                                    entity2_set.add(tok2.id_ann[0])
                                    relation_id_lst.add(rel.rel_id)


    return rel_count, len(entity1_set), len(entity2_set)

def build_stats_byrelation(doc):
    rel_stats = {}
    rel_set = set()

    for tok in doc:
        for rel in tok.relations:
            if rel.rel_id not in rel_set:

                rel_type = rel.rel_type.split("_")
                maintype = rel_type[0]
                subtype = rel_type[1]

                rel_set.add(rel.rel_id)

                if maintype in rel_stats:
                    if subtype in rel_stats[maintype]:
                        rel_stats[maintype][subtype] += 1
                    else:
                        rel_stats[maintype][subtype] = 1
                else:
                    rel_stats[maintype] = {subtype:1}

    return rel_stats

def build_stats_byrelationtype_entity(doc, relation_type,attr_keys):
    """
    Given the relation type, it counts the type of the entities according to an attribute this relation connects
    @param doc:
    @return:
    """
    rel_stats = {}
    rel_set = set()
    entities_stats = {}

    for tok in doc:
        for rel in tok.relations:
            if rel.rel_id not in rel_set and\
                rel.rel_type.lower().startswith(relation_type):
                rel_set.add(rel.rel_id)

                rel_type = rel.rel_type.split("_")
                maintype = rel_type[0]
                subtype = rel_type[1]

                if maintype in rel_stats:
                    if subtype in rel_stats[maintype]:
                        rel_stats[maintype][subtype] += 1
                    else:
                        rel_stats[maintype][subtype] = 1
                else:
                    rel_stats[maintype] = {subtype:1}
                try:
                    attr_map = tok.attr[0][1]
                    attr_map_arg = rel.toks[0].attr[0][1]

                    attr_value = set(attr_map.keys()) & attr_keys
                    attr_value_arg = set(attr_map_arg.keys()) & attr_keys
                    if len(attr_value) == 0:
                        attr_value = None
                    else:
                        attr_value = attr_value.pop()
                    if len(attr_value_arg) == 0:
                        attr_value_arg = None
                    else:
                        attr_value_arg = attr_value_arg.pop()

                    if rel.argn == "arg1":
                        relation_triple = (attr_value_arg, attr_value, rel.rel_type)

                        if attr_value_arg == "Evento_Tipo":
                            print("-->",relation_triple, tok.text, tok.id_ann, rel.rel_id)
                    else:
                        relation_triple =  (attr_value, attr_value_arg, rel.rel_type)

                        if attr_value == "Evento_Tipo":
                            print("-->",relation_triple, tok.text, tok.id_ann, rel.rel_id)

                    if relation_triple in entities_stats:
                        entities_stats[relation_triple] += 1
                    else:
                        entities_stats[relation_triple] = 1
                except KeyError:
                    continue

    return rel_stats, entities_stats

def build_stats_byrelationentity(doc, relation_type,attr_type,attr_type_arg=None):
    """
    Given the relation type, it counts the type of the entities according to an attribute this relation connects
    @param doc:
    @return:
    """
    rel_stats = {}
    rel_set = set()
    entities_stats = {}

    for tok in doc:
        for rel in tok.relations:
            if rel.rel_id not in rel_set and\
                rel.rel_type.lower().startswith(relation_type):

                rel_type = rel.rel_type.split("_")
                maintype = rel_type[0]
                subtype = rel_type[1]

                rel_set.add(rel.rel_id)

                if maintype in rel_stats:
                    if subtype in rel_stats[maintype]:
                        rel_stats[maintype][subtype] += 1
                    else:
                        rel_stats[maintype][subtype] = 1
                else:
                    rel_stats[maintype] = {subtype:1}
                try:
                    if attr_type in tok.attr[0][1]:
                        attr_map = tok.attr[0][1][attr_type]
                    else:
                        attr_map = tok.attr[0][1][attr_type_arg]

                    if attr_type in rel.toks[0].attr[0][1]:
                        attr_map_arg = rel.toks[0].attr[0][1][attr_type]
                    else:
                        attr_map_arg = rel.toks[0].attr[0][1][attr_type_arg]

                    if (attr_map, attr_map_arg,rel.rel_type) in entities_stats:
                        entities_stats[(attr_map, attr_map_arg,rel.rel_type)] += 1
                    else:
                        if (attr_map_arg, attr_map,rel.rel_type) in entities_stats:
                            entities_stats[(attr_map_arg, attr_map,rel.rel_type)] += 1
                        else:
                            entities_stats[(attr_map, attr_map_arg,rel.rel_type)] = 1
                except KeyError:
                    continue

    return rel_stats, entities_stats


def build_stats_byelement_batch(data_dir, element_type="event", dataset_type="BRAT"):
    """

    Count a given type element

    @param data_dir: A string of data directory
    @param dataset_type: default type assumes the BRAT standoff format
    @return: dict of {attribute:frequency}
    """
    reader = ReadBrat()
    doc_lst = reader.process(data_dir)
    attr_count = {}


    for doc in doc_lst:
        attr_count.update(build_stats_byelement(doc, element_type, dataset_type))

    return attr_count

def get_entity(narrative, id_entity):

    if id_entity in narrative.events:
        return narrative.events[id_entity]
    elif id_entity in narrative.participants:
        return narrative.participants[id_entity]
    elif id_entity in narrative.times:
        return narrative.times[id_entity]
    else:
        raise InvalidIDAnn(id_entity)


def build_stats(data_dir):


    counts = {"tokens":[],"events":[],"participants":[],"time":[], "olink":[],
              "srlink":[],"tlink":[],"slink":[],"alink":[],"qslink":[],"movelink":[]}

    count_docs = 0

    for dirpath, dirnames, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith(".ann"):

                full_name = os.path.join(dirpath, f)
                p = Path(f)
                txt_file = os.path.join(data_dir,p.stem + ".txt")

                with open(txt_file, "r") as fd:
                    counts["tokens"].append(count_tokens(fd))

                with open(full_name, "r") as fd:
                    counts["events"].append(count_elements(fd, "Event"))

                with open(full_name, "r") as fd:

                    counts["participants"].append(count_elements(fd, "Participant"))
                with open(full_name, "r") as fd:
                    counts["time"].append(count_elements(fd, "Time"))

                with open(full_name, "r") as fd:
                    counts["olink"].append(count_elements(fd, "OLINK"))

                with open(full_name, "r") as fd:
                    counts["srlink"].append(count_elements(fd, "SRLINK"))

                with open(full_name, "r") as fd:
                    counts["tlink"].append(count_elements(fd, "TLINK"))

                with open(full_name, "r") as fd:
                    counts["slink"].append(count_elements(fd, "SLINK"))

                with open(full_name, "r") as fd:
                    counts["alink"].append(count_elements(fd, "ALINK"))

                with open(full_name, "r") as fd:
                    counts["qslink"].append(count_elements(fd, "QSLINK"))

                with open(full_name, "r") as fd:
                    counts["movelink"].append(count_elements(fd, "MOVELINK"))

                count_docs += 1

    print("Total #docs:", count_docs)
    print("Total #tokens:", sum(counts["tokens"]))
    print("Average #tokens:", sum(counts["tokens"])/len(counts["tokens"]))
    print("Total #events:", sum(counts["events"]))
    print("Total #participants:", sum(counts["participants"]))
    print("Total #time:", sum(counts["time"]))
    print("Total #olink:", sum(counts["olink"]))
    print("Total #srlink:", sum(counts["srlink"]))
    print("Total #tlink:", sum(counts["tlink"]))
    print("Total #slink:", sum(counts["slink"]))
    print("Total #alink:", sum(counts["alink"]))
    print("Total #qslink:", sum(counts["qslink"]))
    print("Total #movelink:", sum(counts["movelink"]))

def count_tlinks_embedded_events_sent(doc):
    """
        This method count all tlinks in the subset of embedded events in
        reporting events
        @param doc: a doc as TokenCorpus list
        @return: a dictionary representing a histogram
        """

    embedded_events_lst, embedded_events_ids, count_supressed_reporting = get_embedded_events(doc)

    hist = {}  # histogram of tlinks betwen reporting events and nested events
    relation_id_lst = set()


    # events nested into sent_id
    for sent_id in embedded_events_lst:
        reporting_event, embedded_events = embedded_events_lst[sent_id]
        for ee in embedded_events:
            for rel in ee.relations:

                # we are collecting relations in which:
                # 1) The type is TLINK
                # 2) the relation isn't analyzed yet
                # 3) the another argument is also an embedded event
                if rel.rel_type.startswith("TLINK") and\
                        rel.rel_id not in relation_id_lst and\
                        rel.toks[0].id_ann[0] in embedded_events_ids:
                    #print("-->", rel.rel_id, rel.rel_type, rel.toks[0].id_ann[0], ee.id_ann[0])

                    if rel.rel_type in hist:
                        hist[rel.rel_type] += 1
                    else:
                        hist[rel.rel_type] = 1
                    relation_id_lst.add(rel.rel_id)
    return hist, count_supressed_reporting


def count_tlinks_embbeded_events(doc):
    """
    This method count all tlinks in the subset of embedded events in
    reporting events
    @param doc: a doc as TokenCorpus list
    @return: a dictionary representing a histogram
    """

    #for tok in doc:
    #    print("-->", tok.text, tok.dep)

    event_lst, nested_events, nested_events_ids = get_nested_events_reporting(doc)
    hist = {}  # histogram of tlinks betwen reporting events and nested events
    event_id_lst = set()

    # events nested into event_id
    for event_id in nested_events:
        for nest_event in nested_events[event_id]:

            for rel in nest_event.relations:

                # we are collecting relations in which:
                # 1) The type is TLINK
                # 2) Its id isn't already checked
                # 3) Its second argument it is not a reporting event (represented by the keys of nested_events)
                # 4) Its second argument is also a nested event
                if rel.rel_type.startswith("TLINK") and \
                        rel.rel_id not in event_id_lst and \
                        rel.toks[0].id_ann[0] not in nested_events.keys() and \
                        rel.toks[0].id_ann[0] in nested_events_ids:

                    if rel.rel_type in hist:
                        hist[rel.rel_type] += 1
                    else:
                        hist[rel.rel_type] = 1
                    event_id_lst.add(rel.rel_id)
    return(hist)

if __name__ == "__main__":
    data_dir = sys.argv[1]
    option = sys.argv[2]
    if option == "-s":
        build_stats(data_dir)
    #elif option == "-tlinks":
    #    count_tlinks_byevent(data_dir)
    else:
        print("Use:")
        print("python stats.py data_dir -s")
        print("python stats.oy data_dir -tlinks")

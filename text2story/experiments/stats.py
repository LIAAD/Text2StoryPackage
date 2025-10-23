# gather corpus stats
import os
import sys

from pathlib import Path
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

        for ann_type, attr_map in tok.attr:


            if ann_type.lower() == element_type:
                if tok.id_ann[0] not in event_id_lst:
                    filter_flag = False
                    for attr in attr_map:
                        # check if the attribute satistifies a given filter
                        if attr in attr_filter and attr_map[attr] == attr_filter[attr]:
                            if tok.id_ann[0] not in allowed_id_lst:
                                filter_flag = True
                                break # ignore this element if it satisfies the requirement

                        attr_map_value = get_attr_repr(attr_map[attr])

                        if attr in attr_count:
                            attr_count[attr][attr_map_value] = attr_count[attr].get(attr_map_value, 0) + 1
                        else:
                            attr_count[attr] = {attr_map_value:1}

                    if not(filter_flag):
                       # if element_type == "quantificacao" and ann_type.lower() == "quantificacao":
                        #    print("-->", tok.id_ann[0], tok.text, tok.attr)
                        event_id_lst.add(tok.id_ann[0])


    return attr_count, len(event_id_lst)

def build_stats_byrelation(doc):
    rel_stats = {}
    rel_set = set()

    for tok in doc:
        for rel in tok.relations:
            if rel.rel_id not in rel_set:

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

    return rel_stats

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

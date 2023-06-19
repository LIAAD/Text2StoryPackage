# gather corpus stats
import os
import sys

from pathlib import Path
from text2story.readers.read_brat import  ReadBrat
from nltk import word_tokenize


def count_tokens(fd):
    return len(word_tokenize(fd.read()))

def count_elements(fd, element_type):

    count = 0
    for line in fd:
        if line[0] == "T":
            elements = line.split()
            if elements[1] == element_type:
                count += 1

    return count

def build_stats_byelement(doc,  element_type="event", dataset_type="BRAT"):

    attr_count = {}
    nelements = 0

    for tok in doc:
        for ann_type, attr_map in tok.attr:
            if ann_type.lower() == element_type:
                nelements += 1
                for attr in attr_map:
                    if attr in attr_count:
                        if attr_map[attr] in attr_count[attr]:
                            attr_count[attr][attr_map[attr]] +=  1
                        else:
                            attr_count[attr][attr_map[attr]] = 1
                    else:
                        attr_count[attr] = {attr_map[attr]:1}

    return attr_count, nelements

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


def build_stats(data_dir):


    counts = {"tokens":[],"events":[],"participants":[],"time":[]}

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

    print("Total #tokens:", sum(counts["tokens"]))
    print("Average #tokens:", sum(counts["tokens"])/len(counts["tokens"]))
    print("Total #events:", sum(counts["events"]))
    print("Total #participants:", sum(counts["participants"]))
    print("Total #time:", sum(counts["time"]))

if __name__ == "__main__":
    data_dir = sys.argv[1]
    build_stats(data_dir)

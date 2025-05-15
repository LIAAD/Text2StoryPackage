"""
This is a script to generate visualization of
brat files in a batch way.
"""
import plantuml
import matplotlib.pyplot as plt

import networkx as nx
from netgraph import Graph, ArcDiagram, InteractiveGraph
import warnings

import argparse
import os

import text2story.brat2viz.drs2viz.parser as parser
from text2story.brat2viz.brat2drs import brat2drs

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def list_ann_files(inputdir):
    """
    It collects the names of .ann files in the inputdir directory
    """

    lst_files = []
    for dirpath, dirnames, filenames in os.walk(inputdir):
        for f in filenames:
            if f.endswith(".ann"):
                lst_files.append(os.path.join(dirpath,f))

    return lst_files

def build_drs(file_name_ann, file_name_drs):

    filecontent = brat2drs.read_file(file_name_ann)
    f_parser = brat2drs.file_parser(filecontent)
    dexpr_list, f_parser = brat2drs.assign_variable(f_parser)
    dr_set, dexpr_list = brat2drs.attributes_events(dexpr_list, f_parser)
    dexpr_list = brat2drs.event_event_relation(f_parser, dexpr_list)

    actors, actors_sr = brat2drs.actors_relation(f_parser)

    relations, dexpr_list = brat2drs.TE_relations(f_parser, dexpr_list)
    relations, dexpr_list = brat2drs.TT_relations(relations, f_parser, dexpr_list)

    actors = brat2drs.update_actors(relations, actors, f_parser)

    brat2drs.write_output(dexpr_list, dr_set, actors, relations, file_name_drs)

def create_plantuml(actors_dict, events_dict, events_relations, non_event_relations):

    msc_str = ""
    for x in non_event_relations:
        ref1 = x[0]
        ref2 = x[2]
        rel_type = x[1]
        msc_str = msc_str + "\"%s\"->\"%s\":%s\n" % (actors_dict[ref1], actors_dict[ref2], rel_type)

    for e in events_relations:
        for (ref1,rel_type,ref2) in events_relations[e]:

            # check for redundandt actor inside string and displays only the first one
            actors_ref1_lst = actors_dict[ref1].split(",")
            actors_ref2_lst = actors_dict[ref2].split(",")

            if len(actors_ref1_lst) > 1:
                participant1 = "%s (%d)" % (actors_ref1_lst[0], len(actors_ref1_lst))
            else:
                participant1 = "%s" % (actors_ref1_lst[0])

            if len(actors_ref2_lst) > 1:
                participant2 = "%s (%d)" % (actors_ref2_lst[0], len(actors_ref2_lst))
            else:
                participant2 = "%s" % (actors_ref2_lst[0])

            msc_str = msc_str + "\"%s\"->\"%s\":%s (%s)\n" % (participant1, participant2, events_dict[e],rel_type)
        #print("-->",e,events_relations[e])

    return msc_str

def build_graph(actors_dict, events_dict, events_relations, non_event_relations):

    #G = nx.MultiDiGraph()
    G = nx.DiGraph()

    labels = {}
    key2labels = {}
    for e in events_relations:
        for (ref1,rel_type,ref2) in events_relations[e]:

            # check for redundandt actor inside string and displays only the first one
            actors_ref1_lst = actors_dict[ref1].split(",")
            actors_ref2_lst = actors_dict[ref2].split(",")

            if len(actors_ref1_lst) > 1:
                participant1 = "%s (%d)" % (actors_ref1_lst[0], len(actors_ref1_lst))
            else:
                participant1 = "%s" % (actors_ref1_lst[0])

            if len(actors_ref2_lst) > 1:
                participant2 = "%s (%d)" % (actors_ref2_lst[0], len(actors_ref2_lst))
            else:
                participant2 = "%s" % (actors_ref2_lst[0])

            if participant1 not in key2labels:
                G.add_node(len(key2labels))
                key2labels[participant1] = len(key2labels.keys())

            node1 = key2labels[participant1]

            if participant2 not in key2labels:
                G.add_node(len(key2labels))
                key2labels[participant2] = len(key2labels.keys())

            node2 = key2labels[participant2]

            label_tokens = events_dict[e].split()
            if len(label_tokens) < 4:
                label_edge = "%s" % (events_dict[e])
            else:
                label_edge = " ".join(label_tokens[:3]) + "..."

            G.add_edge(node1, node2, label=label_edge)

            # print("node1: %d node2: %d label_edge:%s" % (node1, node2, label_edge))

    for k in key2labels:
        labels[key2labels[k]] = k

    return G, labels

def process_edge_labels(edge_labels):
    """
    It process edge labels to find if there are more than one label per node, then
    join the labels in one label

    @param edge_labels: a dictionary that contains (tuple):string as a map for edges and their labels.
    @return: Dict
    """
    new_edge_labels = {}

    for edge in edge_labels:
        n0 = edge[0]
        n1 = edge[1]

        if (n0, n1) in new_edge_labels:
            new_edge_labels[(n0, n1)] = new_edge_labels[(n0, n1)] + " |\n " + edge_labels[edge]
        else:
            new_edge_labels[(n0, n1)] = edge_labels[edge]

    return new_edge_labels

def drs2kg(drs_file, output=None):
    #plt.tight_layout()
    #plt.figure(figsize=(9, 9))

    if output is None:
        file_name_out = os.path.basename(drs_file)
        file_name_out = os.path.splitext(drs_file)[0]
    else:
        file_name_out = output

    # get entities
    actors_dict, events_dict, events_relations, non_event_relations = \
            parser.parse_drs(drs_file)

    file_name_graph_fig = "%s_graph.png" % (file_name_out)

    G, labels = build_graph(actors_dict, events_dict, \
                    events_relations, non_event_relations)
    edge_labels = nx.get_edge_attributes(G, "label")

    edge_labels = process_edge_labels(edge_labels)

    node_size = {}
    for nnode in labels:
        if len(labels[nnode]) > 5:
            node_size[nnode] = 7
        else:
            node_size[nnode] = len(labels[nnode])

    fig, ax = plt.subplots(figsize=(9, 9))
    plot_instance = ArcDiagram(G, edge_layout_kwargs=dict(rad=0.5), node_labels=labels,\
                          node_label_fontdict=dict(size=10),\
                          node_size=node_size, edge_labels=edge_labels, node_layout="circular",\
                               edge_label_fontdict=dict(fontsize=9))


    fig.savefig(file_name_graph_fig, format='png')

def drs2kginteractive(drs_file, output=None):
    #plt.tight_layout()
    #plt.figure(figsize=(9, 9))

    if output is None:
        file_name_out = os.path.basename(drs_file)
        file_name_out = os.path.splitext(drs_file)[0]
    else:
        file_name_out = output

    # get entities
    actors_dict, events_dict, events_relations, non_event_relations = \
            parser.parse_drs(drs_file)

    file_name_graph_fig = "%s_graph.png" % (file_name_out)

    G, labels = build_graph(actors_dict, events_dict, \
                    events_relations, non_event_relations)
    edge_labels = nx.get_edge_attributes(G, "label")

    edge_labels = process_edge_labels(edge_labels)

    node_size = {}
    for nnode in labels:
        if len(labels[nnode]) > 5:
            node_size[nnode] = 7
        else:
            node_size[nnode] = len(labels[nnode])


    fig, ax = plt.subplots(figsize=(9, 9))
    plot_instance = InteractiveGraph(G, edge_layout_kwargs=dict(rad=0.5), node_labels=labels,\
                          node_label_fontdict=dict(size=10),\
                          node_size=node_size, edge_labels=edge_labels, node_layout="spring",\
                               edge_label_fontdict=dict(fontsize=6))

    plt.show()

def drs2msc(drs_file, output=None):
    # url server to generate the figures
    url = "http://www.plantuml.com/plantuml/img/"
    pl = plantuml.PlantUML(url)

    if output is None:
        file_name_out = os.path.basename(drs_file)
        file_name_out = os.path.splitext(drs_file)[0]
    else:
        file_name_out = output

    # get entities
    actors_dict, events_dict, events_relations, non_event_relations = \
        parser.parse_drs(drs_file)

    # build plantuml txt file
    msc_str = create_plantuml(actors_dict, events_dict, \
                              events_relations, non_event_relations)

    with open(file_name_out, "w") as fd:
        fd.write(msc_str)

    file_name_fig = "%s.png" % (file_name_out)
    success = pl.processes_file(file_name_out, outfile=file_name_fig)



def drs2vis(drs_file):
    warnings.warn("drs2viz is deprecated, use drs2msc or drs2kg", DeprecationWarning)

    # url server to generate the figures
    url = "http://www.plantuml.com/plantuml/img/"
    pl = plantuml.PlantUML(url)

    file_name_out = os.path.basename(drs_file)
    file_name_out = os.path.splitext(drs_file)[0]

    # get entities
    actors_dict, events_dict, events_relations, non_event_relations = \
            parser.parse_drs(drs_file)

    # build plantuml txt file
    msc_str = create_plantuml(actors_dict, events_dict, \
                                events_relations, non_event_relations)


    with open(file_name_out,"w") as fd:
        fd.write(msc_str)

    file_name_fig = "%s.png" % (file_name_out)

    success = pl.processes_file(file_name_out, outfile=file_name_fig)




def msc_vis(lst_files, outputdir):

    # url server to generate the figures
    url = "http://www.plantuml.com/plantuml/img/"
    pl = plantuml.PlantUML(url)

    for f in lst_files:

        file_name_out = os.path.basename(f)
        file_name_out = os.path.splitext(file_name_out)[0]

        # build drs
        file_name_drs = os.path.join(outputdir, "%s.drs" % (file_name_out))
        build_drs(f, file_name_drs)

        # get entities
        actors_dict, events_dict, events_relations, non_event_relations = \
                parser.parse_drs(file_name_drs)

        # build plantuml txt file
        msc_str = create_plantuml(actors_dict, events_dict, \
                                    events_relations, non_event_relations)
        #print(msc_str)
        file_name_plantuml = os.path.join(outputdir,"%s.txt" % (file_name_out))
        with open(file_name_plantuml,"w") as fd:
            fd.write(msc_str)

        file_name_fig = os.path.join(outputdir, "%s.png" % (file_name_out))

        success = pl.processes_file(file_name_plantuml, outfile=file_name_fig)

#if __name__ == "__main__":

#    my_parser = argparse.ArgumentParser(description='This is a script to generate visualization of\
#brat files in a batch way.')

#    my_parser.add_argument("inputdir", action='store', type=dir_path, help="The directory that \
#            contains the target files (brat format) and the txt narrative files.")
#    my_parser.add_argument("outputdir", action='store', type=dir_path, help="The directory that \
#            will be written the MSC files.")

    # TODO: visualization in graph format

#    args = my_parser.parse_args()

#    ann_files = list_ann_files(args.inputdir)
#    msc_vis(ann_files, args.outputdir)

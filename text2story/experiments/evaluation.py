import os
import sys

from pathlib import Path

import text2story as t2s  # Import the package
from text2story.readers import read_brat

import argparse
import functools
from datetime import datetime

from text2story.experiments.metrics import *


def start(lang):
    t2s.start(lang)  # Load the pipelines


##########################
### Auxiliary methods ####
##########################

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")



#########################################
### Prediction and evaluation methods ###
#########################################

def extract_element(doc, el, tool):
    """
    It extracts a given element (participant, event or time) using a given
    tool.

    @param doc: a Narrative object
    @param string: a string specifying the type of element to be extracted
    @param string: a string specifying the type of tool to apply in the document

    @return None
    """

    if el == 'participant':
        doc.extract_actors(tool)
    elif el == 'event':
        doc.extract_events(tool)
    elif el == 'time':
        doc.extract_times(tool)
    elif el == 'srlink':
        doc.extract_semantic_role_links(tool)
    else:
        raise Exception("extract_element: Unrecognize element %s" % el)


def evaluate_element(pred_file, target_file, el):
    """
    It evaluates a given element (participant, event or time) .

    @param string: path of prediction file (.ann)
    @param string: path of target annotated file (.ann)
    @param string: a string specifying the type of elment to evaluate

    @return (dict, dict) the relaxed and strict results
    """

    if el == 'participant':
        return evaluate_actor(pred_file, target_file)
    elif el == 'event':
        return evaluate_event(pred_file, target_file)
    elif el == 'time':
        return evaluate_time(pred_file, target_file)
    elif el == 'srlink':
        return evaluate_srlink(pred_file, target_file)
    else:
        raise Exception("evaluate_element: Unrecognize element %s" % el)


def get_element(elem, ann_pred, ann_target):

    if elem == "participant":
        return get_element_actor(ann_pred, ann_target)
    elif elem == "event":
        return get_element_event(ann_pred, ann_target)
    elif elem == "time":
        return get_element_time(ann_pred, ann_target)
    elif elem == "srlink":
        return get_element_srlink(ann_pred,ann_target)
    else:
        raise Exception("get_element: Unrecognize element %s" % el)

def get_element_srlink(ann_pred, ann_target):

    # aqui deveria retorna um dicionario  e nao uma lista, visto que existem varios tipos 
    # de links. Por outro lado, para cada tipo, devem existir em pouca quantidade. 
    # o ideal é quantificar cada tipo de srlink. Dai analisar se vale a pena fazer a análise por
    # tipo aqui
    #r += (f"{sem_link_id}\tSEMROLE_{sem_link.type} Arg1:{sem_link.event} Arg2:{sem_link.actor}\n")
    #r += (f"{sem_link_id}\tSRLINK_{sem_link.type} Arg1:{sem_link.event} Arg2:{sem_link.actor}\n")
    srlink_pred = []
    srlink_target = []

    for k in ann_pred:
        if k.startswith("SEMROLE") or k.startswith("SRLINK"):
            srlink_pred = ann_pred[k]

    for k in ann_target:
        if k.startswith("SEMROLE") or k.startswith("SRLINK"):
            srlink_target = ann_target[k]

    actors_pred, actors_target = get_element_actor(ann_pred, ann_target) 
    event_pred, event_target = get_element_event(ann_pred, ann_target) 

    pred = (actors_pred, event_pred, srlink_pred)
    target = (actors_target, event_target, srlink_target)
    return pred, target


def get_element_event(ann_pred, ann_target):

    # compute accuracy of the exacttly same span
    event_pred = ann_pred["Event"]
    event_target = ann_target["Event"]

    return event_pred, event_target

def get_element_actor(ann_pred, ann_target):

    actor_pred = ann_pred["Actor"]
    # conditions to preserve compatibility between anotation versions
    if len(actor_pred) == 0:
        actor_pred = ann_pred["ACTOR"]
    if len(actor_pred) == 0:
        actor_pred = ann_pred["Participant"]

    actor_target = ann_target["Actor"]
    if len(actor_target) == 0:
        actor_target = ann_target["ACTOR"]
    if len(actor_target) == 0:
        actor_target = ann_target["Participant"]

    return actor_pred, actor_target

def get_element_time(ann_pred, ann_target):

    time_pred = ann_pred["Time"]
    if len(time_pred) == 0:
        time_pred = ann_pred["TIME_X3"]
    time_target = ann_target["Time"]

    return time_pred, time_target


def evaluate_event(pred_file, target_file):
    """
    Implements token event precision/recall/f1 and span event precision/recall/f1

    @param string: predict file in the brat .ann format
    @param string: target file (human labeled) in the brat .ann format

    @return a tuple of dictionaries
    """

    reader = read_brat.ReadBrat()

    ann_pred = reader.read_annotation_file(pred_file)
    ann_target = reader.read_annotation_file(target_file)

    # compute accuracy of the exacttly same span
    event_pred = ann_pred["Event"]
    event_target = ann_target["Event"]

    scores_relax = compute_relax_scores("event",event_pred, event_target)
    scores = compute_strict_scores(event_pred, event_target)
    return scores_relax, scores


def evaluate_actor(pred_file, target_file):
    """
    Implements token actor precision/recall/f1 and span actor precision/recall/f1

    @param string: predict file in the brat .ann format
    @param string: target file (human labeled) in the brat .ann format

    @return a tuple of dictionaries
    """

    reader = read_brat.ReadBrat()

    ann_pred = reader.read_annotation_file(pred_file)
    ann_target = reader.read_annotation_file(target_file)

    # compute accuracy of the exacttly same span

    actor_pred = ann_pred["Actor"]
    # conditions to preserve compatibility between anotation versions
    if len(actor_pred) == 0:
        actor_pred = ann_pred["ACTOR"]
    if len(actor_pred) == 0:
        actor_pred = ann_pred["Participant"]

    actor_target = ann_target["Actor"]
    if len(actor_target) == 0:
        actor_target = ann_target["ACTOR"]
    if len(actor_target) == 0:
        actor_target = ann_target["Participant"]

    scores_relax = compute_relax_scores("participant",actor_pred, actor_target)
    scores = compute_strict_scores("participant", actor_pred, actor_target)
    return scores_relax, scores


def evaluate_time(pred_file, target_file):
    """
    Implements token time precision/recall/f1 and span time precision/recall/f1

    @param string: predict file in the brat .ann format
    @param string: target file (human labeled) in the brat .ann format

    @return a tuple of dictionaries
    """

    reader = read_brat.ReadBrat()
    ann_pred = reader.read_annotation_file(pred_file)
    ann_target = reader.read_annotation_file(target_file)

    # compute accuracy of the exacttly same span
    time_pred = ann_pred["Time"]
    if len(time_pred) == 0:
        time_pred = ann_pred["TIME_X3"]
    time_target = ann_target["Time"]

    scores_relax = compute_relax_scores("time",time_pred, time_target)
    scores = compute_strict_scores(time_pred, time_target)
    return scores_relax, scores

def evaluate_srlink(pred_file, target_file):
    """
    Implements token srlink precision/recall/f1 and span time precision/recall/f1

    @param string: predict file in the brat .ann format
    @param string: target file (human labeled) in the brat .ann format

    @return a tuple of dictionaries
    """

    reader = read_brat.ReadBrat()
    ann_pred = reader.read_annotation_file(pred_file)
    ann_target = reader.read_annotation_file(target_file)


    # compute accuracy of the exacttly same span
    pred, target = get_element("srlink",ann_pred, ann_target)


    scores_relax = compute_relax_scores("srlink", pred, target)
    scores = compute_strict_scores("srlink", pred, target)
    return scores_relax, scores

def prediction(input_dir, results_dir, narrative_elements, language):
    """
    Read brat data (.ann and .txt files) in the input directory,
    and write the results (columns files: token, pred_label, target_label)
    in the results dir

    @param string: input directory with ann and txt files
    @param string: directory with results file for each document file in
    the input_dir

    @return [(string,string)]: a tuple file list to compare
    """

    reader = read_brat.ReadBrat()

    doc_lst = reader.process(input_dir)

    doc_pred_target = []  #

    for idx_doc, doc in enumerate(doc_lst):
        text_ = ""
        with open(reader.file_lst[idx_doc], "r") as fd:
            text_ = fd.read()

        narrative_doc = t2s.Narrative(language, text_, "2020-10-11")


        # extract the element in the given tool
        for el in narrative_elements:
            print("%s extracting from file %s" % (narrative_elements[el], reader.file_lst[idx_doc]))
            extract_element(narrative_doc, el, narrative_elements[el])

        ann_filename = os.path.basename(reader.file_lst[idx_doc])
        ann_filename = os.path.join(results_dir, ann_filename)

        iso_str = narrative_doc.ISO_annotation()
        with open(ann_filename, "w") as fd:
            fd.write(iso_str)

        target_file = Path(reader.file_lst[idx_doc]).stem + ".ann"
        target_file = os.path.join(input_dir, target_file)

        doc_pred_target.append((ann_filename, target_file))

    return doc_pred_target


def process_evaluation(narrative_elements, doc_lst, merge_entities=True):
    """
    Process evaluation for a given element (time, actors or event), in a
    given tool (spacy, spacy, py_heideltime, custompt, etc).

    You should set DATA_DIR and RESULTS_DIR in your PATH enviroment
    vari

    @param string: the element to be extracted
    @param string: the annotator tool to be employed in the extraction

    @param dictionary: a dictionary with the results
    """

    res = {}
    metrics = ["precision_relax","recall_relax", "f1_relax", \
            "precision", "recall", "f1"]

    for elem in narrative_elements:
        for m in metrics:
            res[m + "_" + elem] = []

    for pred_file, target_file in doc_lst:
        print("Evaluating %s and %s" % (pred_file, target_file))
        reader = read_brat.ReadBrat()

        ann_pred = reader.read_annotation_file(pred_file, merge_entities)
        ann_target = reader.read_annotation_file(target_file, merge_entities)

        for elem in narrative_elements:
            pred, target = get_element(elem, ann_pred, ann_target)

            # TODO: esse e so para o srlink. Depois posso querer
            # fazer a divisao por tipos
            #if len(pred) > 0 and len(target) > 0 and \
            #        type(pred[0]) == type(tuple()):
            #    pred = functools.reduce(lambda x,y:x+y,\
            #            [score for (type_pred, score) in pred])
            #    target = functools.reduce(lambda x,y:x+y,\
            #            [score for (type_target, score) in target])

            try:
                precision_relax, recall_relax, f1_relax = compute_relax_scores(elem, pred, target)
                precision, recall, f1  = compute_strict_scores(elem, pred, target)

                res["precision_relax_" + elem].append(precision_relax)
                res["recall_relax_" + elem].append(recall_relax)
                res["f1_relax_" + elem].append(f1_relax)

                res["precision_" + elem].append(precision)
                res["recall_" + elem].append(recall)
                res["f1_" + elem].append(f1)
            except:
                e = sys.exc_info()[0]
                with open("log_evaluation","a") as fd_log:
                    now = datetime.now()
                    time_str = now.strftime("%m/%d/%Y, %H:%M:%S")
                    fd_log.write( "[%s] <p>Error: %s</p>\n" % (time_str, e))
                print("Warning: Some error computing score of %s file" % pred_file)

    return res


def build_evaluation(narrative_elements, language, data_dir: str, results_dir: str):
    """
    Process the evaluation of DATA_DIR (enviroment variable) and put
    the extracted elements in the RESULTS_DIR (enviroment variable)

    @param dict: A dictionary with the elements (actor, time, event) with the
    the tool list to be employed
    @param string: the language to be evaluated (pt or en)
    @param: The System Path where data is stored
    @parm: The System Path where results will be output
    @return None
    """

    doc_pred_lst = prediction(data_dir, results_dir, narrative_elements, language)
    return process_evaluation(narrative_elements, doc_pred_lst)



def print_metrics_result(res: dict):
    print(f"\n-------Metrics Results-------")

    for key_tool in res.keys():
        avg_value = calculate_average_result(res[key_tool])
        print(f"Average Value for Metric {key_tool} is: {avg_value}")

def write_metrics_result(res, fd):
    fd.write(f"\n-------Metrics Results-------\n")

    for key_tool in res.keys():
        avg_value = calculate_average_result(res[key_tool])
        fd.write(f"Average Value for Metric {key_tool} is: {avg_value}\n")


def calculate_average_result(result) -> float:
    counter = 0

    for elem in result:
        counter += elem / len(result)

    return counter


def main(narrative_elements: dict, language: str, data_dir: str = None, results_dir: str = None):

    start(language)

    res = build_evaluation(narrative_elements=narrative_elements, language=language, data_dir=data_dir,
                           results_dir=results_dir)

    print_metrics_result(res)

if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(description='Evaluation of a give dataset according to standard metrics')

    my_parser.add_argument("inputdir", action='store', type=dir_path, help="The directory that contains the target files (brat format) and the txt narrative files.")
    my_parser.add_argument("resultsdir", action='store', type=dir_path, help="The directory where are the files with the extracted entities.")

    my_parser.add_argument("--language", action='store', type=str, help="Current support en (English) and pt (Portuguese. Default: en.")

    my_parser.add_argument("--participant", action='store', type=str, help="The tools to extract participants from narratives. Default: spacy.")
    my_parser.add_argument("--time", action='store', type=str, help="The tools to extract time from narratives. Default: py_heideltime.")
    my_parser.add_argument("--event", action='store', type=str, help="The tools to extract event from narratives. Default: allennlp.")

    my_parser.add_argument("--srlink", action='store', type=str, help="The tools to extract srlinks from narratives. Default: allennlp.")

    args = my_parser.parse_args()

    language = 'en'
    if args.language is not None:
        if args.language != 'en' and args.language != 'pt':
            print("Language option not recognized: %s." % args.language)
            sys.exit()
        language = args.language

    participant_tool = 'spacy'
    if args.participant is not None:
        if args.participant not in t2s.participant_tool_names():
            print("Participant tool not recognized: %s." % args.participant)
            sys.exit()
        participant_tool = args.participant

    time_tool = 'py_heideltime'
    if args.time is not None:
        if args.time not in t2s.time_tool_names():
            print("Time tool not recognized: %s." % args.time)
            sys.exit()
        time_tool = args.time

    event_tool = 'allennlp'
    if args.event is not None:
        if args.event not in t2s.event_tool_names():
            print("Event tool not recognized: %s." % args.event)
            sys.exit()
        event_tool = args.event

    srlink_tool = 'allennlp'
    if args.srlink is not None:
        if args.srlink not in t2s.srlink_tool_names():
            print("Event tool not recognized: %s." % args.event)
            sys.exit()
        srlink_tool = args.srlink

    narrative_elements = {"participant":participant_tool,\
                          "time":time_tool,\
                          "event":event_tool,\
                          "srlink":srlink_tool}

    main(narrative_elements, language,\
            args.inputdir, args.resultsdir)
     
     
     


from collections import Counter

import nltk
from sklearn.metrics import cohen_kappa_score


def partial_match(b1, e1, b2, e2):
    """
    Check if the interval (b1,e1) intersects
    with the interval (b2, e2)

    @param int: beginning of a interval
    @param int: ending of a interval
    @param int: beginning of a interval
    @param int: ending of a interval

    @return bool: if there is an intersection between an interval
     it returns true, otherwise returns false
    """

    if (b2 <= e1) and (b2 >= b1):
        return True
    if (e2 <= e1) and (e2 >= b1):
        return True

    if (b1 >= b2) and (e1 <= e2):
        return True
    return False


def partial_search_annotation(ann, ann_lst):
    """
    given  an annotation (dictionary), do a binary search in a list of annotations
    if the tokens are in the annotation list. The search looks for only
    if there is in intersection between ann and any of the intervals in
    ann_lst

    @param (int, int): a tuple of integers that indicates an intervals
    @param [(int,int, int)]: a list of intervals

    @return int: return a integer that is the index position of the element, or -1
    if the interval is absence of ann_lst

    """

    ans = -1
    # ans = len(word_tokenize(ann["value"]))
    start, end = ann["offset1"]
    b = 0
    e = len(ann_lst) - 1
    m = int((b + e) / 2)
    idkey = None

    while (b <= e):
        start_search, end_search, idkey = ann_lst[m]

        if end < start_search:
            e = m - 1
            m = int((b + e) / 2)
        else:
            if start > end_search:
                b = m + 1
                m = int((b + e) / 2)
            else:
                if partial_match(start, end, start_search, end_search):
                    ans = m
                    break

    return ans, idkey


def search_annotation(ann, ann_lst):
    """
    given  an annotation (dictionary), do a binary search in a list of annotations
    if the tokens are in the annotation list

    @param (int, int): a tuple of integers that indicates an intervals
    @param [(int,int, int)]: a list of intervals

    @return int: return a integer that is the index position of the element, or -1
    if the interval is absence of ann_lst
    """

    ans = -1
    # ans = len(word_tokenize(ann["value"]))
    start, end = ann["offset1"]
    b = 0
    e = len(ann_lst) - 1
    m = int((b + e) / 2)
    idkey = None

    while (b <= e):
        start_search, end_search, idkey = ann_lst[m]
        if end < start_search:
            e = m - 1
            m = int((b + e) / 2)
        else:
            if start > end_search:
                b = m + 1
                m = int((b + e) / 2)
            else:
                if start == start_search and end == end_search:
                    ans = m
                break

    return ans, idkey


def get_intervals(ann_lst):
    """
    get the intervals of annotations as a sorted tuple list

    @param dictionary: dictionary of annotations

    @return tuple list of integers
    """

    interval_lst = []

    for el in ann_lst:
        s1, e1 = el["offset1"]
        interval_lst.append((s1, e1, el["id"]))
        if "offset2" in el:
            s2, e2 = el["offset2"]
            interval_lst.append((s2, e2, el["id"]))

    sorted(interval_lst, key=lambda elem: elem[0])

    return interval_lst

def get_map_list(ann_link):

    arg_map = dict()
    for ann in ann_link:
        arg_map[ann["args"]] = ann["id"]

    return arg_map

def matrix_confusion(res_pred, res_target):


    tp = 0
    fp = 0
    for ans in res_pred:
        if res_pred[ans] is not None:
            # ann_target
            tp += 1  # true positive
        else:
            # false positive
            fp += 1

    tn = 0
    fn = 0
    for ans in res_target:
        if res_target[ans] is None:
            fn += 1
        else:
            tn += 1

    return tp, tn, fp, fn

def compute_precision(tp, fp):

    if tp == 0 and fp == 0:
        return 0
    else:
        return tp / (tp + fp)

def compute_recall(tp, fn):
    if tp == 0 and fn == 0:
        return 0
    else:
        return tp / (tp + fn)

def compute_f1(res_pred, res_target):

    tp, tn, fp, fn = matrix_confusion(res_pred, res_target)
    precision = compute_precision(tp, fp)
    recall = compute_recall(tp, fn)

    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def compute_relax_scores(elem, ann_pred, ann_target):
    if elem == "participant" or elem == "event" or elem == "time":
        match_pred, match_target = compute_relax_scores_entity(ann_pred, ann_target)
        return compute_f1(match_pred, match_target)
    else:
        match_pred, match_target = compute_relax_scores_srlink(ann_pred, ann_target)
        return compute_f1(match_pred, match_target)

def compute_relax_scores_entity(ann_pred, ann_target):
    """
    it computes the  relaxed scores (tokens in common is a match)
    for two annotations

    @param dictionary: annotations of the prediction
    @param dictionary: annotations of the target/human-labeled

    @return match_pred, match_target
    """

    interval_pred_lst = get_intervals(ann_pred)
    interval_target_lst = get_intervals(ann_target)

    interval_pred_lst.sort(key=lambda x: int(x[0]))
    interval_target_lst.sort(key=lambda x: int(x[0]))

    search_pred = {}
    for pred in ann_pred:
        ans, id_ans = partial_search_annotation(pred, interval_target_lst)
        if ans == -1:
            search_pred[pred["id"]] =  None
        else:
            search_pred[pred["id"]] =  id_ans


    fn = 0  # false negative
    search_target = {}
    for target in ann_target:
        ans, id_ans = partial_search_annotation(target, interval_pred_lst)
        if ans == -1:
            search_target[target["id"]] = None
        else:
            search_target[target["id"]] = id_ans

    return search_pred, search_target


def compute_relax_scores_srlink(ann_pred, ann_target):
    #srlink_pred, srlink_target = get_element("srlink",ann_pred, ann_target)
    #participant_pred, participant_target = get_element("participant", ann_pred, ann_target)
    #event_pred, event_target = get_element("event", ann_pred, ann_target)

    participant_pred, event_pred, srlink_pred = ann_pred
    participant_target, event_target, srlink_target = ann_target

    # preciso do indice dos atores e eventos, assim posso conferir
    # o srlink
    search_part_pred, search_part_target = compute_relax_scores_entity(participant_pred, participant_target)

    search_event_pred, search_event_target = compute_relax_scores_entity(event_pred, event_target)

    arg_map_target = get_map_list(srlink_target)
    search_pred = {}
    for srlink in srlink_pred:

        e1, e2 = srlink["args"]
        id_srlink = srlink["id"]

        # if element 1 is a partial match(id) or not (None)
        ans_e1 = None
        if e1 in search_part_pred:
            ans_e1 = search_part_pred[e1]
        else:
            if e1 in search_event_pred:
                ans_e1 = search_event_pred[e1]

        ans_e2 = None
        if e2 in search_part_pred:
            ans_e2 = search_part_pred[e2]
        else:
            if e2 in search_event_pred:
                ans_e2 = search_event_pred[e2]

        # if both match partially in the target, and
        # there is really a link between them in the 
        # srlink_target.. Then get the mapping between the 
        # event/participants from prediction to target
        # check if there is a link between them in the target 
        if ans_e1 is not None and ans_e2 is not None:
             # ans_e1 and ans_e2 are the id of the elements in the 
             # target file
             if (ans_e1, ans_e2) in arg_map_target:
                 search_pred[id_srlink] =  arg_map_target[(ans_e1, ans_e2)]
             elif (ans_e2, ans_e1) in arg_map_target:
                 search_pred[id_srlink] =  arg_map_target[(ans_e2, ans_e1)]
             else:
                 search_pred[id_srlink] = None
        else:
            search_pred[id_srlink] = None

    arg_map_pred = get_map_list(srlink_pred)
    search_target = {}
    for srlink in srlink_target:

        e1, e2 = srlink["args"]
        id_srlink = srlink["id"]

        # if element 1 is a partial match(1) or not (-1)
        ans_e1 = -1
        if e1 in search_part_target:
            ans_e1 = search_part_target[e1]
        else:
            if e1 in search_event_target:
                ans_e1 = search_event_target[e1]

        ans_e2 = -1
        if e2 in search_part_target:
            ans_e2 = search_part_target[e2]
        else:
            if e2 in search_event_pred:
                ans_e2 = search_event_target[e2]

        # if both match partially in the target, and
        # there is really a link between them in the 
        # srlink_target.. Then get the mapping between the 
        # event/participants from prediction to target
        # check if there is a link between them in the target 
        if ans_e1 is not None and ans_e2 is not None:
             # ans_e1 and ans_e2 are the id of the elements in the 
             # target file
             if (ans_e1, ans_e2) in arg_map_pred:
                 search_target[id_srlink] =  arg_map_pred[(ans_e1, ans_e2)]
             elif (ans_e2, ans_e1) in arg_map_pred:
                 search_target[id_srlink] =  arg_map_pred[(ans_e2, ans_e1)]
             else:
                 search_target[id_srlink] = None
        else:
            search_target[id_srlink] = None

    return search_pred, search_target


def compute_strict_scores(elem, ann_pred, ann_target):

    if elem == "participant" or elem == "event" or elem == "time":
        match_pred, match_target = compute_strict_scores_entity(ann_pred, ann_target)
        return compute_f1(match_pred, match_target)
    else:
        match_pred, match_target = compute_strict_scores_srlink(ann_pred, ann_target)
        return compute_f1(match_pred, match_target)

def compute_strict_scores_entity(ann_pred, ann_target):
    """
    it computes the strict scores for two annotations

    @param dictionary: annotations of the prediction
    @param dictionary: annotations of the target/human-labeled

    @return dictionary: scores (precision, recall, and f1)
    computed in a strict manner
    """

    interval_pred_lst = get_intervals(ann_pred)
    interval_target_lst = get_intervals(ann_target)

    interval_pred_lst.sort(key=lambda x: int(x[0]))
    interval_target_lst.sort(key=lambda x: int(x[0]))


    search_pred = {}
    for pred in ann_pred:
        ans, idkey = search_annotation(pred, interval_target_lst)
        if ans == -1:
            search_pred[pred["id"]] = None
        else:
            search_pred[pred["id"]] = idkey

    search_target = {}
    for target in ann_target:
        ans, idkey = search_annotation(target, interval_pred_lst)
        if ans == -1:
            search_target[target["id"]] = None
        else:
            search_target[target["id"]] = idkey

    return search_pred, search_target

def compute_strict_scores_srlink(ann_pred, ann_target):

    participant_pred, event_pred, srlink_pred = ann_pred
    participant_target, event_target, srlink_target = ann_target

    # preciso do indice dos atores e eventos, assim posso conferir
    # o srlink
    search_part_pred, search_part_target = compute_strict_scores_entity(participant_pred, participant_target)

    search_event_pred, search_event_target = compute_strict_scores_entity(event_pred, event_target)

    arg_map_target = get_map_list(srlink_target)
    search_pred = {}
    for srlink in srlink_pred:

        e1, e2 = srlink["args"]
        id_srlink = srlink["id"]

        # if element 1 is a strict match(id) or not (None)
        ans_e1 = None
        if e1 in search_part_pred:
            ans_e1 = search_part_pred[e1]
        else:
            if e1 in search_event_pred:
                ans_e1 = search_event_pred[e1]

        ans_e2 = None
        if e2 in search_part_pred:
            ans_e2 = search_part_pred[e2]
        else:
            if e2 in search_event_pred:
                ans_e2 = search_event_pred[e2]

        # if both match partially in the target, and
        # there is really a link between them in the 
        # srlink_target.. Then get the mapping between the 
        # event/participants from prediction to target
        # check if there is a link between them in the target 
        if ans_e1 is not None and ans_e2 is not None:
             # ans_e1 and ans_e2 are the id of the elements in the 
             # target file
             if (ans_e1, ans_e2) in arg_map_target:
                 search_pred[id_srlink] =  arg_map_target[(ans_e1, ans_e2)]
             elif (ans_e2, ans_e1) in arg_map_target:
                 search_pred[id_srlink] =  arg_map_target[(ans_e2, ans_e1)]
             else:
                 search_pred[id_srlink] = None
        else:
            search_pred[id_srlink] = None

    arg_map_pred = get_map_list(srlink_pred)
    search_target = {}
    for srlink in srlink_target:

        e1, e2 = srlink["args"]
        id_srlink = srlink["id"]

        # if element 1 is a partial match(1) or not (-1)
        ans_e1 = -1
        if e1 in search_part_target:
            ans_e1 = search_part_target[e1]
        else:
            if e1 in search_event_target:
                ans_e1 = search_event_target[e1]

        ans_e2 = -1
        if e2 in search_part_target:
            ans_e2 = search_part_target[e2]
        else:
            if e2 in search_event_pred:
                ans_e2 = search_event_target[e2]

        # if both match partially in the target, and
        # there is really a link between them in the 
        # srlink_target.. Then get the mapping between the 
        # event/participants from prediction to target
        # check if there is a link between them in the target 
        if ans_e1 is not None and ans_e2 is not None:
             # ans_e1 and ans_e2 are the id of the elements in the 
             # target file
             if (ans_e1, ans_e2) in arg_map_pred:
                 search_target[id_srlink] =  arg_map_pred[(ans_e1, ans_e2)]
             elif (ans_e2, ans_e1) in arg_map_pred:
                 search_target[id_srlink] =  arg_map_pred[(ans_e2, ans_e1)]
             else:
                 search_target[id_srlink] = None
        else:
            search_target[id_srlink] = None

    return search_pred, search_target

def compute_bleu_lines(data_lines1, data_lines2):

    text_span_map = {}

    for d1 in data_lines1:
        t1 = nltk.word_tokenize(d1['text_span'])
        span_id1 = d1['span_id']

        for d2 in data_lines2:

            t2 = nltk.word_tokenize(d2['text_span'])

            bleu_score = nltk.translate.bleu([t1], t2, (1,))
            if span_id1 in text_span_map:
                maxb, span_id2 = text_span_map[span_id1]
                if maxb < bleu_score:
                    text_span_map[span_id1] = (bleu_score, d2['span_id'])
            else:
                 text_span_map[span_id1] = (bleu_score, d2['span_id'])

    for span_id1 in text_span_map:
        bleu, _ = text_span_map[span_id1]
        if bleu < 0.8:
            text_span_map[span_id1] = (0, None)
    return text_span_map

def compute_iaa_link(match_pred, reader1, reader2, link_type):

    all_entities_pairs = []
    for element_id_a in match_pred:
        for element_id_b in match_pred:
            all_entities_pairs.append((element_id_a, element_id_b))

    links_lst = []
    links_type_lst = []
    for (element_id_a, element_id_b) in all_entities_pairs:
        # check if annotator1 annotated some relation between event_a and event_b
        relation1 = reader1.get_relation(element_id_a, element_id_b)
        element_id_a_2 = match_pred[element_id_a]
        element_id_b_2 = match_pred[element_id_b]

        relation2 = reader2.get_relation(element_id_a_2, element_id_b_2)

        if relation1 == None and relation2 == None:
            links_lst.append(("none", "none"))
            links_type_lst.append(("none", "none"))
        else:
            if relation1 == None:
                if relation2.startswith(link_type):
                    links_lst.append(("none", link_type.lower()))
                    links_type_lst.append(("none", relation2))
                else:
                    links_lst.append(("none", "other"))
                    links_type_lst.append(("none", "other"))
            else:
                if relation2 == None:
                    if relation1.startswith(link_type):
                        links_lst.append((link_type.lower(), "none"))
                        links_type_lst.append((relation1, "none"))
                    else:
                        links_lst.append(("other", "none"))
                        links_type_lst.append(("other", "none"))
                else:
                    if relation1.startswith(link_type) and relation2.startswith(link_type):
                        links_lst.append((link_type.lower(), link_type.lower()))
                        links_type_lst.append((relation1, relation2))
                    else:
                        links_lst.append(("other", "other"))
                        links_type_lst.append(("other", "other"))

    link_annotations1 = [link[0] for link in links_lst]
    link_annotations2 = [link[1] for link in links_lst]

    link_kappa = cohen_kappa_score(link_annotations1, link_annotations2)

    link_type_annotations1 = [link[0] for link in links_type_lst]
    link_type_annotations2 = [link[1] for link in links_type_lst]

    link_type_kappa = cohen_kappa_score(link_type_annotations1, link_type_annotations2)

    return link_kappa, link_type_kappa





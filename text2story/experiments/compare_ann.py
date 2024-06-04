
import text2story as t2s
from text2story.readers.read_brat import ReadBrat

from text2story.core.exceptions import InvalidIDAnn

def search_entity_span(entity, entity_lst, visited_entities):

    for entity_id in entity_lst:
        if entity.text == entity_lst[entity_id].text and\
            entity_id not in visited_entities:
            visited_entities.add(entity_id)
            return entity_id

def get_mapped_id(entity_id, entities_map):

    for entity_type in entities_map:
        if entity_id in entities_map[entity_type]:
            return entities_map[entity_type][entity_id]

def intersect_entities(doc1, doc2, entity_name):
    """

    @param doc1 Narrative:
    @param doc2 Narrative:
    @return: [(string, string)] A list of ids that doc1 and doc2 have in common.
    None is return if there is no element in common
    """

    entity_name_lst_1 = doc1.get_entity_byname(entity_name)
    entity_name_lst_2 = doc2.get_entity_byname(entity_name)

    visited_entities = set()

    intersection = {}

    for entity_id_1 in entity_name_lst_1:

        e1 = entity_name_lst_1[entity_id_1]
        entity_id_2 = search_entity_span(e1, entity_name_lst_2, visited_entities)

        if entity_id_2 is not None:
            intersection[entity_id_1] = entity_id_2

    return intersection

def is_same_relation_type(relation, relations_lst):

    for rel in relations_lst:
        if isinstance(rel, type(relation)) and rel.type == relation.type:
            return rel.id

def intersect_relations(doc1,doc2, intersections):

    intersection_rel = {}

    for link_type in doc1.links:
        for relation in doc1.links[link_type]:

            arg1_doc2 = get_mapped_id(relation.arg1, intersections)
            arg2_doc2 = get_mapped_id(relation.arg2, intersections)

            if arg1_doc2 is None:
                raise InvalidIDAnn(relation.arg1)

            if arg2_doc2 is None:
                raise InvalidIDAnn(relation.arg2)

            relations_lst = doc2.get_relations_byargs(arg1_doc2, arg2_doc2)
            relation_doc2_id = is_same_relation_type(relation, relations_lst)
            if relation_doc2_id is not None:
                intersection_rel[relation.id] = relation_doc2_id
            else:
                print("-->", relation.id)

    return intersection_rel

def compare_files(lang, file1, file2):

    reader = ReadBrat()

    token_lst_1 = reader.process_file(file1)
    token_lst_2 = reader.process_file(file2)

    doc1 = t2s.Narrative.fromTokenCorpus(lang, token_lst_1)
    doc2 = t2s.Narrative.fromTokenCorpus(lang, token_lst_2)

    intersections = {}

    intersections["participants"] = intersect_entities(doc1, doc2, "participants")
    intersections["events"] = intersect_entities(doc1, doc2, "events")
    intersections["times"] = intersect_entities(doc1, doc2, "times")
    intersections["spatial"] = intersect_entities(doc1, doc2, "spatial_relations")

    agreements = {}

    if len(doc1.participants) > 0:
        agreements["participants"] =  len(intersections["participants"]) / len(doc1.participants)
    else:
        agreements["participants"] = 0

    if len(doc1.events):
        agreements["events"] = len(intersections["events"]) / len(doc1.events)
    else:
        agreements["events"] = 0

    if len(doc1.times) > 0:
        agreements["times"] = len(intersections["times"]) / len(doc1.times)
    else:
        agreements["times"] = 0

    if len(doc1.spatial_relations) > 0:
        agreements["spatial"] = len(intersections["spatial"]) / len(doc1.spatial_relations)
    else:
        agreements["spatial"] = 0

    intersections_rel = intersect_relations(doc1, doc2, intersections)
    if len(doc1.links) > 0:
        total_links = sum([len(doc1.links[link_type]) for link_type in doc1.links])
        agreements["relations"] = len(intersections_rel) / total_links
    else:
        agreements["relations"] = 0

    return agreements
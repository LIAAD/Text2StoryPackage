from typing import List,Dict

def compose_ann_entity(ann_elements:List[str]) -> Dict[str, str]:
    """
    Given an annotation line of an entity, return a dictionary with its fields
    @return: the mapping of an annotation entity
    """
    if ';' in ann_elements[3]:
        ann = {'ann_type': 'TextBound', 'tag_id': ann_elements[0], 'attribute': ann_elements[1],
               's_pos': ann_elements[2], 'f_pos': ann_elements[4], 'value': ' '.join(ann_elements[5:]),
               'tag_ref': ann_elements[2]}
    else:
        ann = {'ann_type': 'TextBound', 'tag_id': ann_elements[0], 'attribute': ann_elements[1], \
               'tag': ann_elements[1].split(':')[0], 'value': ' '.join(ann_elements[4:]), \
               'tag_ref': ann_elements[2]}
    return ann

def file_parser(filecontent):
    ann_dict = []

    for x in filecontent:

        tmp = x.split()
        ann = {}
        if x[0] == 'E':
            ann = {'ann_type': 'TextBound', 'tag_id': tmp[0], 'tag': tmp[1].split(
                ':')[0], 'tag_ref': tmp[1].split(':')[1]}

        elif x[0] == 'A':
            ann = {'ann_type': 'Attribute',
                   'tag_id': tmp[0], 'attr_type': tmp[1], 'tag_ref': tmp[2], 'value': ' '.join(tmp[3:])}
        elif x[0] == 'T':
            ann = compose_ann_entity(tmp)
        elif x[0] == 'R':
            ann = {'ann_type': 'Relation', 'tag_id': tmp[0], 'rel_type': tmp[1], 'tag_ref1': tmp[2].split(':')[1],
                   'tag_ref2': tmp[3].split(':')[1]}
        elif x[0] == '#':
            ann = {'ann_type': 'Note',
                   'tag_id': tmp[0], 'tag_ref': tmp[1], 'note': ' '.join(tmp[2:])}

        if ann != {}:
            ann_dict.append(ann)

    return ann_dict

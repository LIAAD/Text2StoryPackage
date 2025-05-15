import json
import random
import re
from pathlib import Path

# import nltk
from nltk.tokenize import sent_tokenize
# nltk.download('punkt')
from text2story.brat2viz.brat2drs.brat2drs import get_files, read_file
from text2story.brat2viz.brat import file_parser


def get_all_entities(ann_dict):
    entities_dict = {}

    # Avoids >1 dict entries of the same entity (O Comando Metropolitano vs O Comando Metropolitano do Porto)
    tag_ref_beginning = {}

    for x in ann_dict:
        if x['ann_type'] == 'TextBound' and 'attribute' in x:
            if x.get('tag_ref') is None and x.get('s_pos') is not None:
                x['tag_ref'] = x['s_pos']
            try:
                if x['tag_ref'] not in tag_ref_beginning.keys():
                    tag_ref_beginning[x['tag_ref']] = x['tag_id']
                    entities_dict[x['tag_id']] = {'attribute': x['attribute'], 'value': x['value'],
                                              'tag_ref_beginning': x['tag_ref']}
            except TypeError as e:
                print(x, tag_ref_beginning)
                raise e
        # Completes the information in the dict
        elif x['ann_type'] == 'Attribute':
            if x['tag_ref'] in entities_dict.keys():
                entities_dict[x['tag_ref']].update({x['attr_type']: x['value']})


        # Finds Objectal links
        # Considers "OLINK_objIdentity" synonyms
        # Considers "OLINK_partOf", "OLINK_subset" and "OLINK_memberOf" as parts of a whole
        # Ignores "OLINK_referentialDisjunction"
        # Finds synonyms
        elif x['ann_type'] == 'Relation' and x['rel_type'] == 'OLINK_objIdentity':
            if entities_dict.get(x['tag_ref1']) is not None:
                if entities_dict[x['tag_ref1']].get('synonyms') is not None:
                    entities_dict[x['tag_ref1']]['synonyms'] += ';' + x['tag_ref2']
                else:
                    entities_dict[x['tag_ref1']]['synonyms'] = x['tag_ref2']

        # Finds "part of" relations
        elif x['ann_type'] == 'Relation' and x['rel_type'] == 'OLINK_partOf':
            if entities_dict.get(x['tag_ref1']) is not None:
                if entities_dict[x['tag_ref1']].get('Part_Of') is not None:
                    entities_dict[x['tag_ref1']]['Part_Of'] += ';' + x['tag_ref2']
                else:
                    entities_dict[x['tag_ref1']]['Part_Of'] = x['tag_ref2']

        # Finds "member of" relations
        elif x['ann_type'] == 'Relation' and x['rel_type'] == 'OLINK_memberOf':
            if entities_dict.get(x['tag_ref1']) is not None:
                if entities_dict[x['tag_ref1']].get('Part_Of') is not None:
                    entities_dict[x['tag_ref1']]['Part_Of'] += ';' + x['tag_ref2']
                else:
                    entities_dict[x['tag_ref1']]['Part_Of'] = x['tag_ref2']

        # Finds "subset" relations
        elif x['ann_type'] == 'Relation' and x['rel_type'] == 'OLINK_subset':
            if entities_dict.get(x['tag_ref1']) is not None:
                if entities_dict[x['tag_ref1']].get('Part_Of') is not None:
                    entities_dict[x['tag_ref1']]['Part_Of'] += ';' + x['tag_ref2']
                else:
                    entities_dict[x['tag_ref1']]['Part_Of'] = x['tag_ref2']

    return entities_dict


def get_participants(ann_dict):
    participants_dict = {}

    # Avoids >1 dict entries of the same entity (O Comando Metropolitano vs O Comando Metropolitano do Porto)
    tag_ref_beginning = {}
    status = ''

    for x in ann_dict:
        if x.get('attribute') is not None:
            status = x.get('attribute')

        # Creates a dict of participants for fast lookup
        if x['ann_type'] == 'TextBound' and 'attribute' in x \
                and (x['attribute'] == 'ACTOR' or x['attribute'] == 'Participant'):
            status = 'Participant'
            if x.get('tag_ref') is None and x.get('s_pos') is not None:
                x['tag_ref'] = x['s_pos']
            if x['tag_ref'] not in tag_ref_beginning.keys():
                tag_ref_beginning[x['tag_ref']] = x['tag_id']
                participants_dict[x['tag_id']] = {'value': x['value'], 'tag_ref_beginning': x['tag_ref']}
            #
            # else:  # assume synonym if the entity starts in the same character as another
            # or just ignore it
            #    if participants_dict[tag_ref_beginning[x['tag_ref']]].get('synonyms') is not None:
            #        participants_dict[tag_ref_beginning[x['tag_ref']]]['synonyms'] += ', ' + x['tag_id']
            #    else:
            #        participants_dict[tag_ref_beginning[x['tag_ref']]]['synonyms'] = x['tag_id']

        # Completes the information of the participants in the dict
        elif x['ann_type'] == 'Attribute' and status == "Participant":
            if x['tag_ref'] in participants_dict.keys():
                participants_dict[x['tag_ref']].update({x['attr_type']: x['value']})

        # Finds Objectal links
        # Considers "OLINK_objIdentity" synonyms
        # Considers "OLINK_partOf", "OLINK_subset" and "OLINK_memberOf" as parts of a whole
        # Ignores "OLINK_referentialDisjunction"
        # Finds synonyms
        elif x['ann_type'] == 'Relation' and x['rel_type'] == 'OLINK_objIdentity':
            if participants_dict.get(x['tag_ref1']) is not None:
                if participants_dict[x['tag_ref1']].get('synonyms') is not None:
                    participants_dict[x['tag_ref1']]['synonyms'] += ', ' + x['tag_ref2']
                else:
                    participants_dict[x['tag_ref1']]['synonyms'] = x['tag_ref2']

        # Finds "part of" relations
        elif x['ann_type'] == 'Relation' and x['rel_type'] == 'OLINK_partOf':
            if participants_dict.get(x['tag_ref1']) is not None:
                if participants_dict[x['tag_ref1']].get('Part_Of') is not None:
                    participants_dict[x['tag_ref1']]['Part_Of'] += ', ' + x['tag_ref2']
                else:
                    participants_dict[x['tag_ref1']]['Part_Of'] = x['tag_ref2']

        # Finds "member of" relations
        elif x['ann_type'] == 'Relation' and x['rel_type'] == 'OLINK_memberOf':
            if participants_dict.get(x['tag_ref1']) is not None:
                if participants_dict[x['tag_ref1']].get('Part_Of') is not None:
                    participants_dict[x['tag_ref1']]['Part_Of'] += ', ' + x['tag_ref2']
                else:
                    participants_dict[x['tag_ref1']]['Part_Of'] = x['tag_ref2']

        # Finds "subset" relations
        elif x['ann_type'] == 'Relation' and x['rel_type'] == 'OLINK_subset':
            if participants_dict.get(x['tag_ref1']) is not None:
                if participants_dict[x['tag_ref1']].get('Part_Of') is not None:
                    participants_dict[x['tag_ref1']]['Part_Of'] += ', ' + x['tag_ref2']
                else:
                    participants_dict[x['tag_ref1']]['Part_Of'] = x['tag_ref2']

    return participants_dict


def get_events(f_parser):
    events_dict = {}
    status = ''
    for x in f_parser:
        if x.get('attribute') is not None:
            status = x.get('attribute')

        # Creates a dict of events for fast lookup
        if x['ann_type'] == 'TextBound' and 'attribute' in x and x['attribute'] == 'Event':
            status = 'Event'
            events_dict[x['tag_id']] = {'value': x['value']}

        # Completes the information of the events in the dict
        elif x['ann_type'] == 'Attribute' and status == "Event":
            if x['tag_ref'] in events_dict.keys():
                events_dict[x['tag_ref']].update({x['attr_type']: x['value']})
    return events_dict


def get_semantic_relations(f_parser):
    sr_dict = {}
    for item in f_parser:
        if item['ann_type'] == 'Relation' and item['rel_type'][0:6] == 'SRLINK':
            sr_dict[item['tag_id']] = {'rel_type': item['rel_type'], 'tag_ref1': item['tag_ref1'],
                                       'tag_ref2': item['tag_ref2']}
    return sr_dict


def get_all_relations(f_parser):
    link_dict = {}
    for item in f_parser:
        if item['ann_type'] == 'Relation':
            rel_type = item['rel_type'].split('_')[0]
            if 'LINK' in rel_type:
                link_dict[item['tag_id']] = {'rel_type': item['rel_type'], 'tag_ref1': item['tag_ref1'],
                                             'tag_ref2': item['tag_ref2']}
    return link_dict


def get_all_times(f_parser):
    time_dict = {}
    status = ''
    for x in f_parser:
        if status == 'Time':
            if x.get('attribute') != None:
                status = x.get('attribute')

        # Creates a dict of times for fast lookup
        if x['ann_type'] == 'TextBound' and 'attribute' in x and x['attribute'] == 'Time':
            time_dict[x['tag_id']] = {'time': x['value'], 'attribute': x['attribute'], 'tag_ref': x['tag_ref']}
            status = 'Time'

        # Completes the information of the times in the dict
        elif x['ann_type'] == 'Attribute' and status == 'Time':
            if x['tag_ref'] in time_dict.keys():
                time_dict[x['tag_ref']].update({x['attr_type']: x['value']})

    return time_dict


def filter_dict(entities, attribute, value_list):
    return {key: item for (key, item) in entities.items() if item.get(attribute) in value_list}


def group_qslinks_relations(qslink_relations, spatial_relations):
    grouped_relations = {}
    for (k, v) in spatial_relations.items():
        if v.get('Topological') is not None:
            grouped_relations[k] = {'value': v['value'], 'tag_ref_beginning': int(v['tag_ref_beginning']),
                                    'Topological': v['Topological']}
    for (k, v) in qslink_relations.items():
        if v['tag_ref1'] in grouped_relations.keys():
            grouped_relations[v['tag_ref1']].update({v['rel_type']: v['tag_ref2']})
    return grouped_relations


def get_event_time_tlinks(tlinks_relations, events, all_times):
    results = {}
    for tlink_key, tlink_value in tlinks_relations.items():
        if tlink_value['tag_ref1'] in events.keys() and tlink_value['tag_ref2'] in all_times.keys():
            # just for viz
            tlink_value.update({'event': tlink_value['tag_ref1']})
            tlink_value.update({'time': tlink_value['tag_ref2']})
            tlink_value.update({tlink_value['tag_ref1']: events[tlink_value['tag_ref1']]})
            tlink_value.update({tlink_value['tag_ref2']: all_times[tlink_value['tag_ref2']]})
            del tlink_value['tag_ref1']
            del tlink_value['tag_ref2']

            results[tlink_key] = tlink_value

        elif tlink_value['tag_ref1'] in all_times.keys() and tlink_value['tag_ref2'] in events.keys():
            # just for viz
            tlink_value.update({'time': tlink_value['tag_ref1']})
            tlink_value.update({'event': tlink_value['tag_ref2']})
            tlink_value.update({tlink_value['tag_ref1']: all_times[tlink_value['tag_ref1']]})
            tlink_value.update({tlink_value['tag_ref2']: events[tlink_value['tag_ref2']]})
            del tlink_value['tag_ref1']
            del tlink_value['tag_ref2']

            results[tlink_key] = tlink_value
    return results


def group_entities(characters_dict, entities_dict):
    participants_to_del = set()

    for key in characters_dict.keys():
        # Groups part/whole relations
        wholes_joined = characters_dict[key].get("Part_Of")
        if wholes_joined is not None:
            wholes_list = wholes_joined.split(';')
            wholes_list = [item.strip() for item in wholes_list]

            for whole in wholes_list:
                if characters_dict.get(whole) is not None:
                    # print(whole)
                    if characters_dict[whole].get("Parts") is not None:
                        check_unique_list = characters_dict[whole]["Parts"].split(";")
                        check_unique_list = [item.strip() for item in check_unique_list]
                        if key not in check_unique_list:
                            characters_dict[whole]["Parts"] += ";" + key
                    else:
                        characters_dict[whole]["Parts"] = key

            # Falta fazer todas as partes recursivamente
            participants_to_del.add(key)

    for key in reversed(characters_dict.keys()):
        # Groups all synonyms
        synonyms = characters_dict[key].get("synonyms")
        if synonyms is not None:
            synonyms_list = synonyms.split(';')
            synonyms_list = [item.strip() for item in synonyms_list]

            # Gets synonyms of synonyms
            syn_queue = []
            for syn in synonyms_list:
                syn_queue.append(syn)
            while syn_queue:
                if syn_queue[0] != key and syn_queue[0] not in participants_to_del:
                    if characters_dict.get(syn_queue[0]):
                        new_syns = entities_dict[syn_queue[0]].get("synonyms")
                    elif entities_dict.get(syn_queue[0]):
                        new_syns = entities_dict[syn_queue[0]].get("synonyms")
                    if new_syns is not None:
                        new_syns_list = new_syns.split(';')
                        for new_syn_item in new_syns_list:
                            if new_syn_item not in synonyms_list:
                                syn_queue.append(new_syn_item)
                                synonyms_list.append(new_syn_item)
                syn_queue.pop(0)

            for syn_item in synonyms_list:
                fields_to_check = ["Individuation_Domain", "Involvement", "Lexical_Head", "Participant_Type_Domain",
                                   "Parts", "value"]
                # Add the synonym field to the original character dict, if different from original
                # If field == value, add if not present
                for field in fields_to_check:
                    if characters_dict.get(syn_item) is not None:
                        if characters_dict[syn_item].get(field) is not None:
                            check_unique_list = characters_dict[syn_item][field].split(";")
                            check_unique_list = [item.strip() for item in check_unique_list]
                            if characters_dict[key].get(field) is not None and \
                                    characters_dict[key][field] not in check_unique_list:
                                characters_dict[syn_item][field] += ";" + characters_dict[key][field]
                        elif field == "Parts":
                            if characters_dict[key].get(field) is not None:
                                characters_dict[syn_item][field] = characters_dict[key][field]

            participants_to_del.add(key)

    for key in characters_dict.copy().keys():
        if key in participants_to_del:
            del characters_dict[key]

    return characters_dict


def write_jsons(chars_dict, all_characters, file_path, txt_files_dir, output_dir, titles_list, events, locations,
                spatial_info, all_times, event_time_tlinks):
    final_json = {}
    final_json_extended = {}
    title = re.split('\\.|\\/|\\\\', file_path)[-2]

    # Organize character section
    characters = []
    list_of_locations = []
    list_of_dates = []
    for key, value in chars_dict.items():
        rgb = random.sample(range(256), 3)
        rgb_str = 'rgb(' + str(rgb[0]) + ',' + str(rgb[1]) + ',' + str(rgb[2]) + ')'

        name = value["value"].split(';')[0].strip()
        new_character = {"id": key, "name": name, "affiliation": rgb_str}

        if len(value["value"].split(';')) > 1:
            new_character["synonyms"] = [item.strip() for item in value["value"].split(';')[1:] if item != name]
            new_character["synonyms"] = list(dict.fromkeys(new_character["synonyms"]))
        if value.get("Parts") is not None:
            if len(value["Parts"].split(';')) > 0:
                new_character_parts = [item.strip() for item in value["Parts"].split(';')]
                for part in new_character_parts:
                    if new_character.get("synonyms") is not None:
                        new_character["synonyms"] += [item.strip() for item in all_characters[part]["value"].split(';')
                                                      if item != name]
                    else:
                        new_character["synonyms"] = [item.strip() for item in all_characters[part]["value"].split(';')
                                                     if item != name]
        characters.append(new_character)

    for (k, v) in locations.items():
        v['id'] = k
        v['value'] = v['value'].split(';')[0].strip()

    for (k, v) in all_times.items():
        v['id'] = k

    # Organize scene section
    txt_file = txt_files_dir + "/" + title + ".txt"
    with open(txt_file, "r", encoding="utf8") as f:
        unprocessed_txt = f.read()

    sentences_list = sent_tokenize(unprocessed_txt)

    # Clean first sentence (remove headings)
    first_sentence = sentences_list[0].split('\n')[3:]
    char_accumulator = len('\n'.join(sentences_list[0].split('\n')[0:3])) + 1
    sentences_list[0] = '\n'.join(first_sentence)

    sentences_dict = {}
    for i, sentence in enumerate(sentences_list):
        sentences_dict[i] = {'value': sentence, 'beginning_char_index': char_accumulator,
                             'end_char_index': char_accumulator + len(sentence)}
        char_accumulator += len(sentence)

    ## Normal file
    scenes = []
    for k, sentence in sentences_dict.items():
        character_list = []
        for c in characters:
            if c["name"] in sentence['value'] and c["id"] not in character_list:
                character_list.append(c["id"])
            if c.get("synonyms"):
                for synonym in c['synonyms']:
                    if synonym in sentence['value'] and c["id"] not in character_list:
                        character_list.append(c["id"])
        location_list = []
        for loc_key, loc_value in locations.items():
            if loc_value['value'] in sentence['value'] and \
                    sentence['beginning_char_index'] < int(loc_value['tag_ref_beginning']) <= sentence['end_char_index'] \
                    and loc_key not in location_list:
                location_list.append(loc_key)
        times_list = []
        for time_key, time_value in all_times.items():
            if time_value['time'] in sentence['value'] and \
                    sentence['beginning_char_index'] < int(time_value['tag_ref']) <= sentence['end_char_index'] and \
                    time_key not in times_list:
                times_list.append(time_key)

        new_scene = {"characters": character_list, "description": sentence['value'],
                     "title": sentence['value'], "date": times_list,
                     "location": location_list}
        scenes.append(new_scene)

    ## Extended file
    ordered_titles_list = {}
    scene_events = []
    for scene_key, scene_title in titles_list.items():
        events_keys_in_scene = scene_title['events'].split(';')
        event_position_index = events[events_keys_in_scene[0]]['tag_ref_beginning']
        scene_events.append((int(event_position_index), scene_key))
    scene_events_sorted = sorted(scene_events)
    for (pos, key) in scene_events_sorted:
        ordered_titles_list[key] = titles_list[key]

    extended_scenes = []
    for scene_title in ordered_titles_list.values():
        finished = False
        description = None
        title_words = scene_title['title'].split(' ')
        for k, sentence in sentences_dict.items():
            if all(word in sentence['value'] for word in title_words):
                finished = True
                description = sentence['value']
            if finished is True:
                break
        if finished is True and description is not None:
            character_list = []
            for c in characters:
                if c["name"] in scene_title['title'] and c["id"] not in character_list:
                    character_list.append(c["id"])
                if c.get("synonyms"):
                    for synonym in c['synonyms']:
                        if synonym in scene_title['title'] and c["id"] not in character_list:
                            character_list.append(c["id"])
            location_list = []
            for qs in spatial_info.values():
                if (qs.get('QSLINK_figure') in scene_title['events'].split(';') or
                    qs.get('QSLINK_figure') in scene_title['participants'].split(';') or
                    qs.get('QSLINK_figure') in location_list) and \
                        qs.get('QSLINK_ground') in locations.keys():
                    location_list.append(qs['QSLINK_ground'])
            times_list = []
            for time_key, time_value in event_time_tlinks.items():
                if time_value['event'] in scene_title['events'].split(',') and time_value['time'] not in times_list:
                    times_list.append(time_value['time'])

            new_scene = {"characters": character_list, "description": description, "title": scene_title['title'],
                         "location": location_list, "date": times_list}
            extended_scenes.append(new_scene)

    for (k, v) in locations.items():
        v.pop('tag_ref_beginning', None)
        v.pop('Lexical_Head', None)
        v.pop('Individuation_Domain', None)
        v.pop('Participant_Type_Domain', None)
        v.pop('Involvement', None)
        v.pop('attribute', None)
        list_of_locations.append(v)

    time_attributes = ['time', 'value', 'id', 'TemporalFunction']
    for (k, v) in all_times.items():
        attr_to_del = []
        for (kv, vv) in v.items():
            if kv not in time_attributes:
                attr_to_del.append(kv)
        for ka in attr_to_del:
            del v[ka]
        list_of_dates.append(v)

    # Build final dict
    final_json['title'] = title
    final_json['characters'] = characters
    final_json['locations'] = list_of_locations
    final_json['dates'] = list_of_dates
    final_json['scenes'] = scenes

    # Build final extended dict
    final_json_extended['title'] = title
    final_json_extended['characters'] = characters
    final_json_extended['locations'] = list_of_locations
    final_json_extended['dates'] = list_of_dates
    final_json_extended['scenes'] = extended_scenes

    # Dump dict into file
    with open(output_dir + title + ".json", "w") as outfile:
        outfile.write(json.dumps(final_json, indent=4))

    # Dump extended dict into file
    with open(output_dir + title + "_extended.json", "w") as outfile_extended:
        outfile_extended.write(json.dumps(final_json_extended, indent=4))


# [lusa] Considering characters all participants of:
#   Individuation_Domain = Individual, Set
#   Participant_Type_Domain = Pl_*
#   Lexical_Head = Noun
# def get_locations(participants):
#     type_domains = ['Pl_water', 'Pl_celestial', 'pl_mountain', 'Pl_civil', 'Pl_country', 'Pl_mount_range',
#                     'Pl_capital', 'Pl_region', 'Pl_state']
#     return {key: value for (key, value) in participants.items() if
#             (value.get('Individuation_Domain') == 'Individual' or value.get('Individuation_Domain') == 'Set')
#             and (value.get('Participant_Type_Domain') in type_domains) and value.get('Lexical_Head') == 'Noun'}


def filter_semantic_relations(relations_to_consider, all_relations):
    return {key: value for (key, value) in all_relations.items() if value['rel_type'][-len(value['rel_type']) + 7:]
            in relations_to_consider}


def link_sr_to_info(considered_sr_relations, participants, events):
    sr_relations_info = considered_sr_relations.copy()
    for (k, v) in sr_relations_info.items():
        if v['tag_ref1'] in participants:
            sr_relations_info[k].update({'tag_ref1_value': participants[v['tag_ref1']]['value']})
        elif v['tag_ref1'] in events:
            sr_relations_info[k].update({'tag_ref1_value': events[v['tag_ref1']]['value']})

        if v['tag_ref2'] in participants:
            sr_relations_info[k].update({'tag_ref2_value': participants[v['tag_ref2']]['value']})
        elif v['tag_ref2'] in events:
            sr_relations_info[k].update({'tag_ref2_value': events[v['tag_ref2']]['value']})
    return sr_relations_info


def assemble_extended_scenes(semantic_roles_dict_with_dups, participants, events):
    results = {}

    # Deleting duplicates
    semantic_roles_dict = {}
    for sr_key, sr_value in semantic_roles_dict_with_dups.items():
        if sr_value not in semantic_roles_dict.values():
            semantic_roles_dict[sr_key] = sr_value

    if len(semantic_roles_dict) == 0:
        return {}

    semantic_roles = iter(semantic_roles_dict.items())
    k, v = next(semantic_roles)
    prev_k = None
    prev_v = None
    current_scene = {}

    # primeiro elemento a ser feito cá fora
    if v['tag_ref1'] in events:  # verbo -> escrito em segundo
        if v['tag_ref2'] in participants or (v['tag_ref2'] in events and events[v['tag_ref2']]['Pos'] == 'Noun'):
            current_scene['relations'] = {k: v}
            current_scene['events'] = v['tag_ref1']
            current_scene['participants'] = v['tag_ref2']
    elif v['tag_ref1'] in participants or (v['tag_ref1'] in events and events[v['tag_ref1']]['Pos'] == 'Noun'):
        if v['tag_ref2'] in events:
            current_scene['relations'] = {k: v}
            current_scene['events'] = v['tag_ref2']
            current_scene['participants'] = v['tag_ref1']

    beginning_breakpoints = ['SRLINK_pivot', 'SRLINK_agent', 'SRLINK_patient']
    end_breakpoints = ['SRLINK_result']
    double_check = {}
    finished = False
    while not finished:
        (next_k, next_v) = next(semantic_roles)
        relation_processed = False

        if v['rel_type'] in beginning_breakpoints:
            if prev_v is not None and current_scene != {} and \
                    ((prev_v['rel_type'] != 'SRLINK_agent' or v['rel_type'] != 'SRLINK_patient') and
                     (prev_v['rel_type'] != 'SRLINK_patient' or v['rel_type'] != 'SRLINK_agent')):

                # Check for matches in the previous scenes
                is_event = False
                is_participant = False
                scene_id = ''
                for kres, res in results.items():
                    if v['tag_ref1'] == list(res['relations'].values())[-1]['tag_ref1']:  # normally events
                        is_event = True
                        scene_id = kres
                    elif v['tag_ref1'] == list(res['relations'].values())[-1]['tag_ref2']:  # normally nouns
                        is_participant = True
                        scene_id = kres
                if is_event and scene_id != '':
                    if v['tag_ref2'] in participants or (
                            v['tag_ref2'] in events and
                            events[v['tag_ref2']].get('Pos') == 'Noun'):
                        results[scene_id]['relations'].update({k: v})
                        results[scene_id]['participants'] += ';' + v['tag_ref2']
                        relation_processed = True
                elif is_participant and scene_id != '':
                    if v['tag_ref2'] in participants or (
                            v['tag_ref2'] in events and
                            events[v['tag_ref2']].get('Pos') == 'Noun'):
                        results[scene_id]['relations'].update({k: v})
                        results[scene_id]['participants'] += ';' + v['tag_ref2']
                        relation_processed = True

                # Finish current scene and start the next one
                current_scene['key'] = ';'.join(current_scene['relations'].keys())
                results[current_scene['key']] = current_scene
                current_scene = {}

        # events are normally in tag_ref1 and nouns in tag_ref2
        if prev_v is not None and not relation_processed:
            # ver caso o v seja a primeira relation na nova cena
            if current_scene == {}:  # first relation
                if v['rel_type'] in beginning_breakpoints:
                    if v['tag_ref1'] in events:  # verbo -> deve ser escrito em segundo
                        if v['tag_ref2'] in participants or (
                                v['tag_ref2'] in events and events[v['tag_ref2']]['Pos'] == 'Noun'):
                            current_scene['relations'] = {k: v}
                            current_scene['events'] = v['tag_ref1']
                            current_scene['participants'] = v['tag_ref2']
                    elif v['tag_ref1'] in participants or (
                            v['tag_ref1'] in events and events[v['tag_ref1']]['Pos'] == 'Noun'):
                        if v['tag_ref2'] in events:
                            current_scene['relations'] = {k: v}
                            current_scene['events'] = v['tag_ref2']
                            current_scene['participants'] = v['tag_ref1']
                else:
                    double_check.update({k: v})
            else:
                is_event = False
                is_participant = False
                scene_id = ''
                for kr, r in results.items():
                    if v['tag_ref1'] == list(r['relations'].values())[-1]['tag_ref1']:  # normally events
                        is_event = True
                        scene_id = kr
                    elif v['tag_ref1'] == list(r['relations'].values())[-1]['tag_ref2']:  # normally nouns
                        is_participant = True
                        scene_id = kr
                if v['tag_ref1'] == list(current_scene['relations'].values())[-1]['tag_ref1']:  # normally events
                    if v['tag_ref2'] in participants or (
                            v['tag_ref2'] in events and
                            events[v['tag_ref2']].get('Pos') == 'Noun'):
                        current_scene['relations'].update({k: v})
                        current_scene['participants'] += ';' + v['tag_ref2']
                elif v['tag_ref1'] == list(current_scene['relations'].values())[-1]['tag_ref2']:  # normally nouns
                    if v['tag_ref2'] in participants or (
                            v['tag_ref2'] in events and
                            events[v['tag_ref2']].get('Pos') == 'Noun'):
                        current_scene['relations'].update({k: v})
                        current_scene['participants'] += ';' + v['tag_ref2']
                    # falta locations mas isto se calhar é à parte porque há uma relaçao especifica para isto
                elif (is_event or is_participant) and scene_id != '':
                    if v['tag_ref2'] in participants or (
                            v['tag_ref2'] in events and
                            events[v['tag_ref2']].get('Pos') == 'Noun'):
                        results[scene_id]['relations'].update({k: v})
                        results[scene_id]['participants'] += ';' + v['tag_ref2']
                else:
                    double_check.update({k: v})

        if v['rel_type'] in end_breakpoints and current_scene != {}:
            if len(current_scene['relations']) == 1 and prev_v['tag_ref1'] in events and events[prev_v['tag_ref1']][
                'Class'] == 'Reporting':
                current_scene = {}
            else:
                current_scene['key'] = ';'.join(current_scene['relations'].keys())
                results[current_scene['key']] = current_scene
                current_scene = {}

        prev_k = k
        prev_v = v
        k = next_k
        v = next_v
        if list(semantic_roles_dict.keys())[-1] == k:
            finished = True
            # Last scene
            if current_scene != {}:
                if len(current_scene['relations']) == 1 and \
                        list(current_scene['relations'].values())[-1]['tag_ref1'] in events and \
                        events[list(current_scene['relations'].values())[-1]['tag_ref1']]['Class'] == 'Reporting':
                    current_scene = {}
                else:
                    current_scene['key'] = ';'.join(current_scene['relations'].keys())
                    results[current_scene['key']] = current_scene
                    current_scene = {}

    # For unordered relations, that are spread across the file
    for k, v in double_check.items():
        is_event = False
        is_participant = False
        scene_id = ''
        for kr, r in results.items():
            if v['tag_ref1'] == list(r['relations'].values())[-1]['tag_ref1']:  # normally events
                is_event = True
                scene_id = kr
            elif v['tag_ref1'] == list(r['relations'].values())[-1]['tag_ref2']:  # normally nouns
                is_participant = True
                scene_id = kr
            # falta locations mas isto se calhar é à parte porque há uma relaçao especifica para isto
        if (is_event or is_participant) and scene_id != '':
            if v['tag_ref2'] in participants or (
                    v['tag_ref2'] in events and
                    events[v['tag_ref2']].get('Pos') == 'Noun'):
                results[scene_id]['relations'].update({k: v})
                results[scene_id]['participants'] += ';' + v['tag_ref2']

    # Find the scenes to delete
    scenes_to_delete = []
    for k, r in results.items():
        if len(r['relations']) == 1 and \
                ((list(r['relations'].values())[-1]['tag_ref1'] in events and \
                  events[list(r['relations'].values())[-1]['tag_ref1']].get('Class') == 'Reporting') or
                 (list(r['relations'].values())[-1]['tag_ref2'] in events and \
                  events[list(r['relations'].values())[-1]['tag_ref2']].get('Class') == 'Reporting')):
            scenes_to_delete.append(r['key'])
        else:
            has_key_relation = False
            key_relations = ['SRLINK_agent', 'SRLINK_patient', 'SRLINK_pivot', 'SRLINK_result']
            agent_counter = 0  # if a scene has 2 or more agents then it's deleted
            for relation_key, relation_value in r['relations'].items():
                if relation_value['rel_type'] == 'SRLINK_agent':
                    has_key_relation = True
                    agent_counter += 1
                if relation_value['rel_type'] in key_relations:
                    has_key_relation = True
            if not has_key_relation or agent_counter > 1:
                scenes_to_delete.append(r['key'])
    for scene_to_del in scenes_to_delete:
        results.pop(scene_to_del)

    return results


def build_titles(extended_scenes):
    for scene_key, scene_value in extended_scenes.items():
        relations_list = list(scene_value['relations'].values())
        participants_mentioned = []
        for i, v in enumerate(relations_list):
            # First relation
            if i == 0:
                if v['tag_ref1'] in scene_value['events'].split(";"):  # verbo -> deve ser escrito em segundo
                    if v['tag_ref2'] in scene_value['participants'].split(";"):
                        scene_value['title'] = v['tag_ref2_value'].split(';')[0].strip() + ' ' + \
                                               v['tag_ref1_value'].split(';')[0].strip()
                        participants_mentioned.append(v['tag_ref2'])
                elif v['tag_ref1'] in scene_value['participants'].split(";"):
                    if v['tag_ref2'] in scene_value['events'].split(";"):
                        scene_value['title'] = v['tag_ref1_value'].split(';')[0].strip() + ' ' + \
                                               v['tag_ref2_value'].split(';')[0].strip()
                        participants_mentioned.append(v['tag_ref1'])
            # Next relation
            else:
                if v['tag_ref1'] == relations_list[i - 1]['tag_ref1']:  # normally events
                    if v['tag_ref2'] in scene_value['participants'] and v['tag_ref2'] not in participants_mentioned:
                        scene_value['title'] += ' ' + v['tag_ref2_value'].split(';')[0].strip()
                        participants_mentioned.append(v['tag_ref2'])
                elif v['tag_ref1'] == relations_list[i - 1]['tag_ref2']:  # normally nouns
                    if v['tag_ref2'] in scene_value['participants']:
                        scene_value['title'] += ' ' + v['tag_ref2_value'].split(';')[0].strip()
        if scene_value['title'][-3:] == 'que':
            scene_value['title'] = scene_value['title'][:len(scene_value['title']) - 4]

    # v['tag_ref2_value'] != 'que' -> tirar 'que's que estejam sozinhos
    return extended_scenes


def get_characters(all_entities):
    participants = filter_dict(all_entities.copy(), 'attribute', ['Participant'])
    characters = filter_dict(participants.copy(), 'Individuation_Domain', ['Individual', 'Set'])
    characters = filter_dict(characters.copy(), 'Participant_Type_Domain', ['Org', 'Per'])
    characters = filter_dict(characters.copy(), 'Lexical_Head', ['Noun'])
    characters_grouped = group_entities(characters.copy(), all_entities)

    return participants, characters, characters_grouped


def get_locations(participants, all_relations, all_entities):
    locations = filter_dict(participants.copy(), 'Individuation_Domain', ['Individual', 'Set'])
    location_type_domains = ['Fac', 'Path', 'Pl_water', 'Pl_celestial', 'Pl_mountain', 'Pl_civil', 'Pl_country',
                             'Pl_mount_range', 'Pl_capital', 'Pl_region', 'Pl_state', 'Loc']  # Loc não está documentado
    locations = filter_dict(locations.copy(), 'Participant_Type_Domain', location_type_domains)
    locations = filter_dict(locations.copy(), 'Lexical_Head', ['Noun'])

    qslink_relations = filter_dict(all_relations.copy(), 'rel_type', ['QSLINK_ground', 'QSLINK_figure'])
    spatial_relations = filter_dict(all_entities.copy(), 'attribute', ['Spatial_Relation'])

    # QSLINK_figure: entity or eventuality, QSLINK_ground: location
    spatial_information = group_qslinks_relations(qslink_relations, spatial_relations)
    #locations_grouped = group_entities(locations.copy(), all_entities)
    sr_loc_relations = filter_dict(all_relations.copy(), 'rel_type', ['SRLINK_location'])
    for kl, vl in sr_loc_relations.items():
        if vl['tag_ref1'] in locations:
            spatial_information.update({kl: {'QSLINK_figure': vl['tag_ref2'], 'QSLINK_ground': vl['tag_ref1']}})
        elif vl['tag_ref2'] in locations:
            spatial_information.update({kl: {'QSLINK_figure': vl['tag_ref1'], 'QSLINK_ground': vl['tag_ref2']}})

    return locations, spatial_information


def get_times(all_relations, events, all_times):
    tlinks_list = ['TLINK_before', 'TLINK_after', 'TLINK_includes', 'TLINK_isIncluded', 'TLINK_during',
                   'TLINK_simultaneous', 'TLINK_identity', 'TLINK_begins', 'TLINK_ends', 'TLINK_begunBy',
                   'TLINK_endedBy']
    tlinks_relations = filter_dict(all_relations.copy(), 'rel_type', tlinks_list)
    event_time_tlinks = get_event_time_tlinks(tlinks_relations, events, all_times)

    return event_time_tlinks


def get_sr_relations(f_parser, events, participants):
    sr_relations = get_semantic_relations(f_parser)
    # relations_to_consider = ['agent', 'partner', 'cause', 'patient', 'pivot', 'theme', 'patient', 'source', 'result',
    #                         'path', 'attribute']
    relations_to_consider = ['agent', 'partner', 'cause', 'patient', 'pivot', 'theme', 'beneficiary',
                             'source', 'goal', 'result', 'path', 'attribute', 'reason', 'purpose', 'amount']
    considered_sr_relations = filter_semantic_relations(relations_to_consider, sr_relations.copy())
    sr_with_info = link_sr_to_info(considered_sr_relations.copy(), participants, events)

    return sr_with_info


def get_extended_scenes(sr_with_info, participants, events):
    extended_scenes_assembled = assemble_extended_scenes(sr_with_info, participants, events)
    extended_scenes = build_titles(extended_scenes_assembled.copy())

    return extended_scenes


def brat2json(f, json_dir):
    print("-->", f)
    file_content = read_file(f)
    f_parser = file_parser(file_content)

    all_entities = get_all_entities(f_parser)
    all_relations = get_all_relations(f_parser)
    all_times = get_all_times(f_parser)

    (participants, characters, characters_grouped) = get_characters(all_entities)
    events = filter_dict(all_entities.copy(), 'attribute', ['Event'])
    (locations, spatial_information) = get_locations(participants, all_relations, all_entities)
    event_time_tlinks = get_times(all_relations, events, all_times)
    sr_with_info = get_sr_relations(f_parser, events, participants)
    extended_scenes = get_extended_scenes(sr_with_info, participants, events)

    txt_input_dir = '/'.join(f.split('/')[:len(f.split('/'))-1]) + '/'
    output_dir = '/'.join(f.split('/')[:len(f.split('/'))-2]) + '/' + json_dir
    # Create output dir if does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_jsons(characters_grouped, characters, f, txt_input_dir, output_dir, extended_scenes, events, locations,
                spatial_information, all_times, event_time_tlinks)


def bratdir2json(ann_dir, output_dir='output'):
    file_paths = get_files(ann_dir)  # get .ann files
    file_paths = [file_paths[x].replace('\\', '/') for x in range(0, len(file_paths))]
    for f in file_paths:
        brat2json(f, output_dir)



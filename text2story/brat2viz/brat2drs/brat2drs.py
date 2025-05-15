# %%
# Brat2DRT

import re
import glob
import os
import string

import platform

from pathlib import Path

# DRT
from nltk.sem import logic
from nltk.inference import TableauProver
from nltk.sem.drt import *
from text2story.brat2viz.brat import file_parser
# %%

dexpr = DrtExpression.fromstring

# ANN_DIR = 'brat_ann_files/'
#ANN_DIR = 'text_2_story_new/'
ANN_DIR = 'sample/'
#ANN_DIR = 'lusa_news/'
# ANN_DIR = 'lusa_news_old/'
#DRS_DIR = '../drs_files/'
DRS_DIR = 'drs_sample/'


def get_files(path=ANN_DIR):
    # path = r'.' # use your path
    path = r"{}".format(path)
    all_files = glob.glob(path + "/*.ann")
    print(all_files)
    return all_files


def read_file(filename):
    with open(filename, encoding="utf8") as f:
        content = f.readlines()

    content = [x.strip().replace('\t', ' ') for x in content]
    return content



# assign a variable for each event
def assign_variable(ann_dict):
    var = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    var = var + [''.join([a, b]) for a in var for b in var]

    # events_var = []
    dexpr_list = []
    tmp = []
    i = 0
    for x in ann_dict:
        if x['ann_type'] == 'Event':
            if 'tag_ref' in x:
                dexpr_list.append(
                    ({'event_var': var[i], 'event_tag_id': x['tag_id'], 'tag_ref': x['tag_ref'],'event_str': x['value']},
                    dexpr('event(%s)' % var[i])))
            else:
                dexpr_list.append(
                        ({'event_var': var[i], 'event_tag_id': x['tag_id'],'event_str': x['value'],'tag_ref':'E'},
                    dexpr('event(%s)' % var[i])))
            i += 1
            x['var'] = var[i]

        elif x['ann_type'] == 'TextBound' and 'attribute' in x\
                and x['attribute'] == 'Event':
            # new version of annotation consider Event in a textbound tagid,
            # but to remain compatible with other annotations version the bellow condition
            # is still in the source code
            dexpr_list.append(
                ({'event_var': var[i], 'event_tag_id': x['tag_id'], \
                        'event_str': x['value'],'tag_ref':'E'},
                 dexpr('event(%s)' % var[i])))
            i += 1
            x['var'] = var[i]
        elif x['ann_type'] == 'TextBound' and 'attribute' in x\
                and x['attribute'] == 'EVENT':
            tmp.append((x['tag_id'], x['value']))

    for i, x in enumerate(dexpr_list):
        # in the new version of annotation we dont need a second list like tmp

        for y in tmp:
            if x[0]['tag_ref'] == y[0]:
                x[0]['event_str'] = y[1]
                dexpr_list[i] = x

    return dexpr_list, ann_dict


def attributes_events(dexpr_list, ann_dict):
    dr_set = set()  # discourse referents set

    for x in ann_dict:

        if x['ann_type'] == 'Attribute' and \
                x['tag_ref'][0] == 'E':
            for i, y in enumerate(dexpr_list):

                if y[0].get('event_tag_id') == x['tag_ref']:
                    if x['attr_type'] == 'Class':
                        y = y + \
                            (dexpr('event%s(%s,%s)' % (
                                x['attr_type'], y[0].get('event_var'), x['value'])),)
                    else:
                        y = y + \
                            (dexpr('%s(%s,%s)' % (x['attr_type'], y[0].get(
                                'event_var'), x['value'])),)

                    dexpr_list[i] = y
                    dr_set.add(x['value'])

    dexpr_list = format_tuple(dexpr_list)
    return dr_set, dexpr_list


def format_tuple(dexpr_list):
    for n, x in enumerate(dexpr_list):
        e_attr = list(x)
        k = x + (e_attr[1],)

        for i in e_attr[2:]:
            k = k + (k[-1] + i,)
        x = x + (k[-1],)
        dexpr_list[n] = x

    return dexpr_list


# get event ref variable
def get_ref(dexpr_list, var):
    ref = ''
    for i, y in enumerate(dexpr_list):
        if y[0].get('event_tag_id') == var:
            ref = (y[1], y[0].get('event_str'), y[0].get('event_var'))
            # drs1 = y[-1]
            break
    return ref


def get_value(ann_dict, var):

    for x in ann_dict:
        # apenas TextBound e Attribute
        if (x['ann_type'] == 'TextBound' or x['ann_type'] == 'Attribute') and (x['tag_id'] == var):
            return x.get('value')

def event_event_relation(ann_dict, dexpr_list):
    for x in ann_dict:

        if x['ann_type'] == 'Relation' and 'TLINK' in x['rel_type']:
            if 'before' not in x['rel_type'] and 'after' not in x['rel_type']:
                continue
            ref1 = ref2 = ''
            # drs1 = drs2 = ''

            ref2 = get_ref(dexpr_list, x['tag_ref2'])
            ref1 = get_ref(dexpr_list, x['tag_ref1'])

            for i, y in enumerate(dexpr_list):
                # print(i, ref1[2], type(ref1[2]),y[0].get('event_var'),x['rel_type'], type(x['rel_type']))
                if not ref1 or not ref2: continue
                if (y[0].get('event_var') == ref1[2]) and ('before' not in x['rel_type']):
                    n = len(y) - 2
                    y = y[: n] + (dexpr('occursAfter(%s,%s)' % (ref1[2], ref2[2])),) + y[n:]
                    dexpr_list[i] = y
                elif (y[0].get('event_var') == ref2[2]):
                    n = len(y) - 2
                    y = y[: n] + (dexpr('occursBefore(%s,%s)' % (ref2[2], ref1[2])),) + y[n:]
                    dexpr_list[i] = y

    return dexpr_list


def write_output(dexpr_list, dr_set, actors, actors_events, out_file):
    with open(out_file, 'w') as f:
        try:
            print('» EVENTS', file=f)

            for x in dexpr_list:

                if "event_str" not in x[0].keys():
                    continue

                var = dexpr('%s' % x[0].get('event_var'))
                event_str = ''.join([x.capitalize() for x in x[0].get('event_str').split()])
                #event_str = event_str.replace('-', '')
                event_str = "".join([c for c in event_str if c not in string.punctuation])
                event_str = event_str + '=' + x[0].get('event_var')
                var_ = dexpr('%s' % event_str)
                tmp_dr = []

                for y in dr_set:
                    tmp_dr = tmp_dr + [str(y) for x in list(x)
                    [1:-1] if re.search(y, str(x))]

                tmp_dr = [k for k in list(set(tmp_dr)) if type(k) is str]
                tmp_dr.append(var)


                print('# ' + x[0].get('event_tag_id') + ' (' +
                      x[0].get('event_str') + ') -> ' + x[0].get('event_var'), file=f)
                print('# FOL: ', DRS(tmp_dr, x[1:-1]).fol(), file=f)
                print('# DRS: ', DRS([var], x[1:-1]), '\n', file=f)
                print(DRS([var_], x[1:-1]).pretty_format())
        except Exception as e:
            print('Exception while writing EVENTS\n')
            print(e)

        try:
            print('» ACTORS', file=f)
            for x in actors:
                print('# ' + x[0] + ' -> ' + x[1], file=f)
        except Exception as e:
            print('Exception while writing ACTORS\n')
            print(e)

        try:
            print('\n» RELATIONS', file=f)
            for x in actors_events:
                print('# ' + x.get('ref1') + ' - ' + x.get('relation') + ' - ' + x.get('ref2'), file=f)
        except Exception as e:
            print('Exception while writing RELATIONS\n')
            print(e)


def get_actors(ann_dict):
    actors_dict = []
    for x in ann_dict:
        # new version of lusa news requires Participant as actors
        if x['ann_type'] == 'TextBound' and 'attribute' in x\
                and (x['attribute'] == 'ACTOR' or x['attribute'] == 'Participant'):
            d = {'tag_id': x['tag_id'], 'value': x['value']}
            actors_dict.append(d)
        elif x['ann_type'] == 'Participant':
            d = {'tag_id': x['tag_id'], 'value': x['value']}
            actors_dict.append(d)
    return actors_dict


def actors_relation(ann_dict):
    relations = list()
    for x in ann_dict:
        r_set = set()
        if x['ann_type'] == 'Relation':  # and x['rel_type'] == 'OBJ_REL_objIdentity':

            for y in ann_dict:
                if y['ann_type'] == 'Relation':  # and y['rel_type'] == 'OBJ_REL_objIdentity':
                    if (x['tag_ref1'] == y['tag_ref1']):
                        r_set.add(x['tag_ref1']);
                        r_set.add(y['tag_ref1'])
                    if (x['tag_ref1'] == y['tag_ref2']):
                        r_set.add(x['tag_ref2']);
                        r_set.add(y['tag_ref2'])
                    if (x['tag_ref2'] == y['tag_ref1']):
                        r_set.add(x['tag_ref1']);
                        r_set.add(y['tag_ref1'])
                    if (x['tag_ref2'] == y['tag_ref2']):
                        r_set.add(x['tag_ref2']);
                        r_set.add(y['tag_ref2'])
                relations.append(list(r_set))

    # tuple_list = list(set(tuple(i) for i in relations))
    # tuple_list = [x for x in tuple_list if len(x) > 1]
    # graph = nx.Graph(tuple_list)
    # result = list(nx.connected_components(graph))
    # print(result)

    tuple_list = list(set(tuple(i) for i in relations))
    tuple_list = [x for x in tuple_list if len(x) > 1]

    actors = get_actors(ann_dict)
    updated_actors = aux_actors(actors, tuple_list)
    updated_actors.sort()

    return updated_actors, tuple_list


# Seleciona somente atores que aparecem em uma relação
def aux_actors(actors, actors_rel):
    tuple_list = list()
    for y in actors_rel:
        for x in actors:
            if x.get('tag_id') in y:
                tuple_list.append((x.get('tag_id'), x.get('value')))
                # break
    return list(set(tuple_list))


def update_relations(actors, actors_sr, relations):
    new_relations = list()
    a = [i[0] for i in actors]
    t_list = set()
    # print(relations)
    for n, x in enumerate(relations):
        if not x['relation'] == 'objIdentity':
            r1 = {x.get('ref1')}
            for i in actors_sr:
                if r1.intersection(i):
                    import pdb
                    pdb.set_trace()
                    v = list(i.intersection(a))[0]
                    # print(v)
                    x['ref1'] = v
        new_relations.append(x)

    return new_relations


def update_actors(relations, actors, ann_dict):

    events_set = set()
    for elem in ann_dict:
        if 'attribute' in elem and\
                elem['attribute'] == "Event":
            events_set.add(elem["tag_id"])


    a = [i[0] for i in actors]
    r_list = list()
    for x in relations:
        if x['ref1'][0] == 'T' and x['ref1'] not in events_set:
            # print(x)
            r_list.append(x.get('ref1'))
        if x['ref2'][0] == 'T' and x['ref1'] not in events_set:
            # print(x)
            r_list.append(x.get('ref2'))

    new_list = list()
    for x in r_list:
        if x not in a:
            new_list.append(x)

    new_actors = list()
    for x in new_list:
        new_actors.append((x, get_value(ann_dict, x)))

    return actors + list(set(new_actors))


# actors_events relations
def TE_relations(ann_dict, dexpr_list):
    relations = list()
    for x in ann_dict:
        if x['ann_type'] == 'Relation' and \
                ('SEMROLE' in x['rel_type'] or 'SRLINK' in x['rel_type']):
            relations.append(
                {'ref1': x.get('tag_ref1'), 'ref2': x.get('tag_ref2'), 'relation': x.get('rel_type').split('_')[1]})

            event = x['tag_ref1'] if x['tag_ref1'][0] == 'E' else x['tag_ref2']
            var = x['tag_ref1'] if x['tag_ref1'][0] != 'E' else x['tag_ref2']

            event_ref = get_ref(dexpr_list, event)

            value = get_value(ann_dict, var)
            # value = ''.join([x.capitalize() for x in value.split(' ')])
            value = var

            try:
                for i, y in enumerate(dexpr_list):
                    if y[0].get('event_var') == event_ref[2]:
                        n = len(y) - 2
                        y = y[: n] + (dexpr('relationRole(%s,%s)' % (x.get('rel_type').split('_')[1], value)),) + y[n:]
                        dexpr_list[i] = y
            except:
                continue

    return relations, dexpr_list


def TT_relations(relations, ann_dict, dexpr_list):
    # relations = list()
    for x in ann_dict:
        # if x['ann_type'] == 'Relation' and 'OBJ_REL' in x['rel_type']:
        if x['ann_type'] == 'Relation' and ('OBJ_REL' in x['rel_type'] \
                or 'REF_REL' in x['rel_type'] or x['rel_type'].startswith('OLINK')):

            rel_type = x.get('rel_type').split('_')
            relations.append(
                {'ref1': x.get('tag_ref1'), 'ref2': x.get('tag_ref2'), 'relation':rel_type[-1]})

            # print(relations)
    return relations, dexpr_list


# %%
def process(file_name):

    print(file_name)
    filecontent = read_file(file_name)
    f_parser = file_parser(filecontent)
    dexpr_list, f_parser = assign_variable(f_parser)
    dr_set, dexpr_list = attributes_events(dexpr_list, f_parser)
    dexpr_list = event_event_relation(f_parser, dexpr_list)

    # actors = get_actors(f_parser)
    actors, actors_sr = actors_relation(f_parser)
    #print(actors)


    # platform.system() returns Windows, Linux or Darwin (for OSX)
    if platform.system() == 'Windows':
        output_file_name = file_name.split('\\')[-1].replace('.ann', '_drs.txt')
    else:
        output_file_name = file_name.split('/')[-1].replace('.ann', '_drs.txt')

    relations, dexpr_list = TE_relations(f_parser, dexpr_list)
    relations, dexpr_list = TT_relations(relations, f_parser, dexpr_list)

    # relations = update_relations(actors, actors_sr, relations)
    actors = update_actors(relations, actors, f_parser)

    write_output(dexpr_list, dr_set, actors, relations, output_file_name)


def main(ann_dir):
    files = get_files(ann_dir)
    # files = [files[3]]
    for f in files:
        print(f)
        filecontent = read_file(f)
        f_parser = file_parser(filecontent)
        dexpr_list, f_parser = assign_variable(f_parser)
        dr_set, dexpr_list = attributes_events(dexpr_list, f_parser)
        dexpr_list = event_event_relation(f_parser, dexpr_list)

        # actors = get_actors(f_parser)
        actors, actors_sr = actors_relation(f_parser)
        # print(actors)

        # Create output dir if does not exist
        Path(DRS_DIR).mkdir(parents=True, exist_ok=True)

        # platform.system() returns Windows, Linux or Darwin (for OSX)
        if platform.system() == 'Windows':
            output_file_name = f.split('\\')[1].replace('.ann', '_drs.txt')
        else:
            output_file_name = f.split('/')[1].replace('.ann', '_drs.txt')

        relations, dexpr_list = TE_relations(f_parser, dexpr_list)
        relations, dexpr_list = TT_relations(relations, f_parser, dexpr_list)

        # relations = update_relations(actors, actors_sr, relations)
        actors = update_actors(relations, actors, f_parser)

        output_file = os.path.join(DRS_DIR, output_file_name)
        write_output(dexpr_list, dr_set, actors, relations, output_file)

        # break


if __name__ == "__main__":
    main(ANN_DIR)

# %%

import os
import json
working_dir = os.path.dirname(os.path.realpath(__file__))
pb_to_vb_map = json.load(open(os.path.join(working_dir,"pb-vn2.json"),"r"))
vb_to_lc_map = json.load(open(os.path.join(working_dir,"vn-lirics.json"),"r"))

straightforward_maps = {"ARGM-TMP":"time","ARGM-MNR":"manner"}

def get_index(lst, ele):
    # get the first occurrence index of ele in lst
    # if there is no element, return None
    for idx, e in enumerate(lst):
        if e == ele:
            return idx

    return None

def convert_role_propbank_to_lirics(roleset_prop, arg_label):

    # if the verb is not in the mapping, ignore

    if roleset_prop in pb_to_vb_map:

        # first it converts to verbnet, then it
        vn_mapping = pb_to_vb_map[roleset_prop].values()
        vb_label = None
        for v in vn_mapping:
            if arg_label.upper() in v:
               vb_label = v[arg_label.upper()]
               break

        if vb_label is not None and\
                vb_label in vb_to_lc_map.keys():
            return vb_to_lc_map[vb_label]

    if arg_label in straightforward_maps.keys():
        return straightforward_maps[arg_label]

    # if there is no mapping, just return None
    return None



import os.path

import plantuml

from text2story.readers.read_brat import ReadBrat
from text2story.select.bubble import BubbleMap
from text2story.select.event import sieve_bbubbles_events, get_all_events
from text2story.core.utils import join_tokens

MAP_COLORS = {"State":"#D7BDE2",\
              "Reporting":"#85C1E9",
              "Occurrence":"#EDBB99",\
              "Perception":"#EC7063 ",\
              "Aspectual":"#D5DBDB",\
              "I_Action":"#48C9B0",\
              "I_State":"#F7DC6F"}

def build_fig(doc):
    msc_str = "@startuml\nskinparam style strictuml\n"

    # for each id annotation get the related events
    all_event_lst = get_all_events(doc)

    # get only eligible big bubble events (with temporal links identity events)
    #event_lst, not_big_bubble_events = sieve_bbubbles_events(all_event_lst, "Reporting")
    #bubble_ids = {'T26','T28','T77','T31','T86','T32','T34','T33','T35'}

    excluded_events = set()
    for event_id in all_event_lst:
        #if event_id in bubble_ids:
            event_class = all_event_lst[event_id][0].get_attr_value("Class")
            event_text = join_tokens([tok.text for tok in all_event_lst[event_id]])

            msc_str += f"participant \"{event_text}\" as {event_id} {MAP_COLORS[event_class]}\n"
            excluded_events.add(event_id)

    rel_id_lst = set()
    for event_id in excluded_events:
        for rel in all_event_lst[event_id][0].relations:

            if rel.rel_id not in rel_id_lst and \
                    rel.rel_type.startswith("TLINK") and \
                    rel.toks[0].id_ann[0] in excluded_events:
                rel_name = rel.rel_type.split("_")[1]
                if rel.argn == 'arg2':
                    msc_str += f"{event_id} -> {rel.toks[0].id_ann[0]}: {rel_name}\n"
                else:
                    msc_str += f"{rel.toks[0].id_ann[0]} -> {event_id}: {rel_name}\n"
                rel_id_lst.add(rel.rel_id)

    msc_str += "@enduml"

    return msc_str
def build_fig_ann(ann_file, output_dir, lang="en"):

    url = "http://www.plantuml.com/plantuml/img/"
    pl = plantuml.PlantUML(url)

    reader = ReadBrat(lang)
    data = reader.process_file(ann_file)

    #only  get non-reporting events and the temporal relation links
    # between them
    msc_str = build_fig(data)
    ann_base_name = os.path.split(ann_file)[1]
    file_name_out = os.path.join(output_dir, f"{ann_base_name}_msc.txt")
    print("-->", file_name_out)

    with open(file_name_out, "w") as fd:
        fd.write(msc_str)

    file_name_fig = os.path.join(output_dir, f"{ann_base_name}_msc.png")
    print("-->", file_name_fig)
    success = pl.processes_file(file_name_out, outfile=file_name_fig)
import os

from pyvis.network import Network

import text2story as t2s
from brat2drs import brat2drs
from drs2viz import parser

BRAT_ANN_DIR = './files'


def write_ann_to_file(iso_notation):
    f = open(f"{BRAT_ANN_DIR}/example.ann", "w")
    for line in enumerate(iso_notation):
        f.write(f'{line[1]}\n')
    f.flush()
    f.close()


def write_txt_to_file(text):
    f = open(f"{BRAT_ANN_DIR}/example.txt", "w")
    f.write(text)
    f.flush()
    f.close()


class Text2Viz:
    def __init__(self, language, text, publication_time, list_visualizations=None):
        self.narrative = t2s.Narrative(language, text, publication_time)
        self.models_loaded = False

        if list_visualizations is None:
            self.list_visualizations = ['ann_text', 'drs', 'graph']
        else:
            self.list_visualizations = list_visualizations

    def run(self):
        write_txt_to_file(self.narrative.text)

        if not self.models_loaded:
            print("Loading Models.")

            t2s.start()

            print("Finish Loading Models")

            self.models_loaded = True

        print("Done")

        iso_notation = self.sanitize_iso(self.extract_elements())

        write_ann_to_file(iso_notation)

        self.brat_2_drs(iso_notation)

        self.drs_2_viz()

    def extract_elements(self) -> str:

        if self.narrative.lang == 'en':
            self.narrative.extract_actors('spacy')
            self.narrative.extract_times('py_heideltime')
            self.narrative.extract_events('allennlp')
            self.narrative.extract_objectal_links('allennlp')
            self.narrative.extract_semantic_role_links('allennlp')

        elif self.narrative.lang == 'pt':
            self.narrative.extract_actors('spacy')
            self.narrative.extract_times('py_heideltime')
            self.narrative.extract_events('custompt')
            # self.narrative.extract_objectal_links('allennlp')
            # self.narrative.extract_semantic_role_links('allennlp')

        return self.narrative.ISO_annotation()

    def brat_2_drs(self, list_file_content: list):
        f_parser = brat2drs.file_parser(list_file_content)

        dexpr_list, f_parser = brat2drs.assign_variable(f_parser)
        dr_set, dexpr_list = brat2drs.attributes_events(dexpr_list, f_parser)
        dexpr_list = brat2drs.event_event_relation(f_parser, dexpr_list)
        actors, actors_sr = brat2drs.actors_relation(f_parser)

        relations, dexpr_list = brat2drs.TE_relations(f_parser, dexpr_list)
        relations, dexpr_list = brat2drs.TT_relations(relations, f_parser, dexpr_list)

        actors = brat2drs.update_actors(relations, actors, f_parser)

        output_file = f'{BRAT_ANN_DIR}/example_drs.txt'

        brat2drs.write_output(dexpr_list, dr_set, actors, relations, output_file)

    def drs_2_viz_graph(self, filename, notebook_bool=False):

        actors, non_ev_rels, ev_rels = parser.get_graph_data(f'./files/{filename}_drs.txt')

        net = Network(notebook=notebook_bool)

        for key in actors.keys():
            net.add_node(key, label=actors[key])

        for elem in non_ev_rels:
            net.add_edge(elem[0], elem[2], title=elem[1])

        for elem in ev_rels:
            net.add_edge(elem[0], elem[2], title=elem[1])

        net.show('mygraph.html')

    def drs_2_viz(self, notebook_bool=False):
        drs_file = parser.get_drs_files(BRAT_ANN_DIR)[0]
        filename = drs_file.split('/')[-1].split('_drs')[0]

        if 'ann_text' in self.list_visualizations:
            self.drs_2_viz_ann_text(filename)

        if 'drs' in self.list_visualizations:
            self.drs_2_viz_drs_txt(filename)

        # self.drs_2_viz_msc(filename)

        if 'graph' in self.list_visualizations:
            self.drs_2_viz_graph(filename, notebook_bool)

    def drs_2_viz_ann_text(self, filename):
        ann_file = os.path.join(BRAT_ANN_DIR, f'{filename}.ann')

        with open(ann_file, 'r') as fp:
            ann_text = fp.readlines()

        print(f"ANN_TEXT:\n-----------------------")

        for line in ann_text:
            print(line)

    def drs_2_viz_drs_txt(self, filename):
        with open(f'./files/{filename}_drs.txt', 'r') as fp:
            drs_text = fp.readlines()

        print(f"DRS_TEXT:-----------------------\n")

        for line in drs_text:
            print(line)

    def drs_2_viz_msc(self, drs_file):

        parser.get_msc_data(f'./files/{drs_file}_drs.txt')

    def sanitize_iso(self, raw_iso: str):
        list_elem = raw_iso.split('\n')

        parsed_list = [x.strip().replace('\t', ' ') for x in list_elem]

        return list(filter(lambda elem: elem != "", parsed_list))


def require_input_from_user():
    language = input(f"Enter language 'pt' for Portuguese or 'en' for English\n")

    if language != 'pt' and language != 'en':
        print("Not a valid language. Assuming English")
        language = 'en'

    text = input("Insert Text\n")

    return [language, text]


if __name__ == '__main__':
    text2viz = Text2Viz('en', '', '2021-12-31')

    while True:

        [language, text] = require_input_from_user()

        if language == 'pt':
            text2viz.narrative.lang = 'pt'
        elif language == 'en':
            text2viz.narrative.lang = 'en'

        text2viz.narrative.text = text

        text2viz.run()

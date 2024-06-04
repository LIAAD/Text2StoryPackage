import text2story as t2s

text = 'O rei morreu na batalha. A rainha casou com seu irmão.'
my_narrative = t2s.Narrative('pt', text, '2024')

my_narrative.extract_events('srl')
my_narrative.extract_participants('spacy', 'srl')
my_narrative.extract_times('py_heideltime')

my_narrative.extract_semantic_role_links('srl')
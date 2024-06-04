import text2story as t2s

t2s.start("fr")
text_ = "Que retenir de la visite d'État d'Emmanuel Macron en Allemagne?"
my_narrative = t2s.Narrative('fr', text_, "2024")

my_narrative.extract_participants("custom_annotator")

print(my_narrative.participants)
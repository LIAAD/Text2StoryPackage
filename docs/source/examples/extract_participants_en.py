import text2story as t2s

text = 'The king died in battle. The queen married his brother.'
my_narrative = t2s.Narrative('en', text, '2024')
my_narrative.extract_participants('nltk')

print(my_narrative.participants)

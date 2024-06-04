import text2story as t2s
text = 'The traveling salesman went town to town. However, he did not sell one book.'
my_narrative = t2s.Narrative('en', text, '2024')
my_narrative.extract_times('py_heideltime')

print(my_narrative.times)
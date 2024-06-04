from text2story.readers.read_brat import ReadBrat

reader = ReadBrat()

# in the BRAT file format, you have a txt file and a corresponding ann file,
# both have the same name. For instance, in this example, data/doc1.txt is the
# raw text file, and data/doc1.ann contains the annotations. But we only provide the
# name without to specify the extension.
doc = reader.process_file("data/doc1")
for tok in doc:
    print(tok.text)

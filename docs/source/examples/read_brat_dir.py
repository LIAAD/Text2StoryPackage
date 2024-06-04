from text2story.readers.read_brat import ReadBrat

reader = ReadBrat()

doc_lst = reader.process("data/")
for doc in doc_lst:
    # each doc is a TokenCorpus list
    for tok in doc:
        print(tok.text)

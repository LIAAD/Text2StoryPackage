Usage
=====


Getting Start
-----

After installation, you will be able to import and create an object of type Narrative. In this object,
the text2story package will perform all automatic annotations. In the next section, we cover all the
functionalities about the annotators.

The Narrative Object
-----

The narrative object contains all the properties and functions for a narrative extraction
process. So, to use text2story, the first step is to import the library and create an object
of this kind. To create such an object, the required arguments are the language code (i.e., 'en'
for English, 'pt' for Portuguese, and so on.), the text of the narrative (the length of the input is restricted
to the models applied in the pipeline), and the document creation date. The last one is especially important for news
stories which usually present a publication date.

The code bellow presents an example with a raw text English that is used to create a Narrative object.

.. code-block:: python
    import text2story as t2s

    text = 'The king died in battle. The queen married his brother.'
    my_narrative = t2s.Narrative('en', text, '2024')

The narrative object is used then to process all the pipeline of annotators that will extract the narrative components.
The Section Annotators Module details how to build such pipeline.

The Readers Module
-----

If the user want to read an already human annotated dataset, it is possible to do such a thing using some object of the
type reader. text2story readers module supports the following formats: ACE, BRAT, CSV, ECB, Framenet, Propbank.
Each one of this module inherits the methods from the abstract class `Read`, which obliges all inherited classes to
implement the method `process` and `process_file`. The first method reads all files (text and annotations) from a
given directory, and the second reads only one file (text and its annotations).

It is assumed that both methods returns a list of `TokenCorpus`, which is type that contains a token and its
annotations, if they exists. This is also a class defined in readers module.

Next, a code example to read a directory with annotations in BRAT format.

.. code-block:: python
    from text2story.readers.read_brat import ReadBrat
    reader = ReadBrat()

    doc_lst = reader.process("data/")
    for doc in doc_lst:
        # each doc is a TokenCorpus list
        for tok in doc:
            print(tok.text)


The next code illustrate how to use `ReadBrat` to read only one file.

.. code-block:: python
    from text2story.readers.read_brat import ReadBrat
    reader = ReadBrat()

    # in the BRAT file format, you have a txt file and a corresponding ann file,
    # both have the same name. For instance, in this example, data/doc1.txt is the
    # raw text file, and data/doc1.ann contains the annotations. But we only provide the
    # name without to specify the extension.
    doc = reader.process_file("data/doc1")
    for tok in doc:
        print(tok.text)



The Annotators Module
-----

There are two type of annotators in the text2tstory: the native ones and the customized ones.
The first is composed by a set of pre-trained models that are part of the library, and
are all naturally integrated in our pipeline. The second type is composed by annotators that
anyone can built and integrate in our pipeline. For both, it is required to load the models for the
language of the used examples. The code bellow is used to load the models for the English language.

.. code-block:: python
    import text2story as t2s
    t2s.start('en')

.. note::

   Before load models, it is required to install the model for tei2go. For instance, if you are going to use english models. You should execute `pip install https://huggingface.co/hugosousa/en_tei2go/resolve/main/en_tei2go-any-py3-none-any.whl`.

Next, we describe how the native and custom annotators work.

Native Annotators
^^^^^^^^^^^^^^^^^

The native annotators are the following modules: NLTK, PY_HEIDELTIME, BERTNERPT, TEI2GO, SPACY and ALLENNLP. Next, we detail
the usage of each one of these annotators.

Participants
''''''''
For  participants, we have the following annotators SPACY ('pt','en'), NLTK ('en'), ALLENNLP ('en'),
BERTNERPT ('pt'), and SRL ('pt').

The NLTK module uses a Named Entity Recognition (NER) model trained in the ACE dataset to identify participants in
the English language. So, after loading the english models, you can use the code bellow to extract
participants using the NLTK module. Others modules that employs NER to identify participants are SPACY
(en_core_web_lg/'en', pt_core_news_lg/'pt') and BERTNERPT (https://huggingface.co/arubenruben/NER-PT-BERT-CRF-Conll2003).
Bellow, an example of using only NLTK to extract participants from a narrative.

.. code-block:: python
    import text2story as t2s
    text = 'The king died in battle. The queen married his brother.'
    my_narrative = t2s.Narrative('en', text, '2024')
    my_narrative.extract_participants('nltk')

    print(my_narrative.participants)

The ALLENNLP ('en') and SRL('pt') modules employ Semantic Role Labeling modules to identify participants and the
code for them is the same as above, only changing the name of the module.

It is also possible to use pipeline models to obtain better or different results. The code below extracts
participants from a narrative text in Portuguese using SPACY and SRL modules.

.. code-block:: python
    import text2story as t2s
    text = 'O rei morreu na batalha. A rainha casou com seu irmão.'
    my_narrative = t2s.Narrative('pt', text, '2024')
    my_narrative.extract_participants('srl','spacy')

    print(my_narrative.participants)


Time
''''''''

For time expression, text2story has py_heideltime and tei2go to identify time expressions both in Portuguese and
English languages. The code is similar to the extraction of participants. See the example bellow:


.. code-block:: python
    import text2story as t2s
    text = 'The traveling salesman went town to town. However, he did not sell one book.'
    my_narrative = t2s.Narrative('en', text, '2024')
    my_narrative.extract_times('py_heideltime')

    print(my_narrative.times)



Events
''''''''

There are only two modules devoted to the extraction of events, ALLENNLP ('en') and SRL ('pt'). The extraction of
events is done in the same way as the extraction of time and participants. See the code below.

.. code-block:: python
    import text2story as t2s
    text = 'O rei morreu na batalha. A rainha casou com seu irmão.'
    my_narrative = t2s.Narrative('pt', text, '2024')
    my_narrative.extract_events('srl')

    print(my_narrative.events)

Semantic Links
''''''''

Semantic links can only be extracted after the extraction of events, participants, and time. So, the code below
updates the example code from the extraction of events.

.. code-block:: python
    import text2story as t2s
    text = 'O rei morreu na batalha. A rainha casou com seu irmão.'
    my_narrative = t2s.Narrative('pt', text, '2024')

    my_narrative.extract_events('srl')
    my_narrative.extract_participants('spacy','srl')
    my_narrative.extract_times('py_heideltime')

    my_narrative.extract_semantic_role_links('srl')


Custom Annotators
^^^^^^^^^^^^^^^^^

A custom annotator should follow the structure of a standard annotator, i.e., it should contain at least the
load function. The main goal of this method is to load the models used in its pipeline. For instance, consider the
following implementation of a custom annotator that uses tei2go French model to extract time expressions, and
the spacy NER French model to extract participants.

.. literalinclude:: custom_annotator.py
  :language: python

To use your new annotator, first, you need to add it to the text2story pipeline using the following code.

.. code-block:: python
    import text2story as t2s

    t2s.add_annotator("custom_annotator", ['fr'], ['participant', 'time'])

Then, you can use the annotator like the native ones. See the code below.

.. code-block:: python
    import text2story as t2s

    t2s.start("fr")
    text_ = "Que retenir de la visite d'État d'Emmanuel Macron en Allemagne?"
    my_narrative = t2s.Narrative('fr',text_,"2024")

    my_narrative.extract_participants("custom_annotator")

    print(my_narrative.participants)

.. The Visualization Module
.. -----


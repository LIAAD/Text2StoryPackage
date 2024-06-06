Usage
=====


Getting Started
-----

After installation, you will be able to import and create an object of the Narrative type. In this object,
the text2story package will perform all automatic annotations. In the next section, we cover all the
functionalities of the annotators.

The Narrative Object
-----

The narrative object contains all the properties and functions for a narrative extraction
process. So, to use text2story, the first step is to import the library and create an object
of this kind. To create such an object, the required arguments are the language code (i.e., 'en'
for English, 'pt' for Portuguese, and so on.), the text of the narrative (the length of the input is restricted
to the models applied in the pipeline), and the document creation date. The last one is especially important for news
stories which usually present a publication date.

The code below presents an example with a raw text English that is used to create a Narrative object.

.. literalinclude:: examples/narrative_object.py
  :language: python


The narrative object is used then to process all the pipeline of annotators that will extract the narrative components.
The Section Annotators Module details how to build such a pipeline.

The Readers Module
-----

If the user wants to read an already human-annotated dataset, it is possible to do such a thing using some object of the
type reader. text2story readers module supports the following formats: ACE, BRAT, CSV, ECB, Framenet, Propbank.
Each one of these modules inherits the methods from the abstract class `Read`, which obliges all inherited classes to
implement the method `process` and `process_file`. The first method reads all files (text and annotations) from a
given directory, and the second reads only one file (text and its annotations).

It is assumed that both methods return a list of `TokenCorpus`, which is the type that contains a token and its
annotations if they exist. This is also a class defined in the reader's module.

Next, a code example to read a directory with annotations in BRAT format.

.. literalinclude:: examples/read_brat_dir.py
  :language: python


The next code illustrates how to use `ReadBrat` to read only one file.

.. literalinclude:: examples/read_brat_file.py
  :language: python


The Annotators Module
-----

There are two types of annotators in the text2tstory: the native ones and the customized ones.
The first is composed of a set of pre-trained models that are part of the library, and
are all naturally integrated in our pipeline. The second type is composed of annotators that
anyone can build and integrate into our pipeline. For both, the models must be loaded in the
language of the examples used. The code below is used to load the models for the English language.

.. literalinclude:: examples/load_models.py
  :language: python


.. note::

   Before loading models, the model for tei2go must be installed. For instance, if you are going to use English models. You should execute `pip install https://huggingface.co/hugosousa/en_tei2go/resolve/main/en_tei2go-any-py3-none-any.whl`.

Next, we describe how the native and custom annotators work.

Native Annotators
^^^^^^^^^^^^^^^^^

The native annotators are the following modules: NLTK, PY_HEIDELTIME, BERTNERPT, TEI2GO, SPACY, and ALLENNLP. Next, we detail
the usage of each one of these annotators.

Participants
''''''''
For  participants, we have the following annotators SPACY ('pt','en'), NLTK ('en'), ALLENNLP ('en'),
BERTNERPT ('pt'), and SRL ('pt').

The NLTK module uses a Named Entity Recognition (NER) model trained in the ACE dataset to identify participants in
the English language. So, after loading the English models, you can use the code below to extract
participants using the NLTK module. Other modules that employ NER to identify participants are SPACY
(en_core_web_lg/'en', pt_core_news_lg/'pt') and BERTNERPT (https://huggingface.co/arubenruben/NER-PT-BERT-CRF-Conll2003).
Bellow, an example of using only NLTK to extract participants from a narrative.

.. literalinclude:: examples/extract_participants_en.py
  :language: python

The ALLENNLP ('en') and SRL('pt') modules employ Semantic Role Labeling modules to identify participants and the
code for them is the same as above, only changing the name of the module.

It is also possible to use pipeline models to obtain better or different results. The code below extracts
participants from a narrative text in Portuguese using SPACY and SRL modules.

.. literalinclude:: examples/extract_participants_pt.py
  :language: python



Time
''''''''

For time expression, text2story has py_heideltime and tei2go to identify time expressions both in Portuguese and
English languages. The code is similar to the extraction of participants. See the example below:


.. literalinclude:: examples/extract_time.py
  :language: python



Events
''''''''

There are only two modules devoted to the extraction of events, ALLENNLP ('en') and SRL ('pt'). The extraction of
events is done in the same way as the extraction of time and participants. See the code below.

.. literalinclude:: examples/extract_events.py
  :language: python

Semantic Links
''''''''

Semantic links can only be extracted after the extraction of events, participants, and time. So, the code below
updates the example code from the extraction of events.

.. literalinclude:: examples/extract_semantic_links.py
  :language: python


Custom Annotators
^^^^^^^^^^^^^^^^^

A custom annotator should follow the structure of a standard annotator, i.e., it should contain at least the
load function. The main goal of this method is to load the models used in its pipeline. For instance, consider the
following implementation of a custom annotator that uses tei2go French model to extract time expressions, and
the spacy NER French model to extract participants.

.. literalinclude:: examples/custom_annotator.py
  :language: python

To use your new annotator, first, you need to add it to the text2story pipeline using the following code.

.. literalinclude:: examples/add_custom_annotator.py
  :language: python

Then, you can use the annotator like the native ones. See the code below.

.. literalinclude:: examples/test_custom_annotator.py
  :language: python

.. The Visualization Module
.. -----




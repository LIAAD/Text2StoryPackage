# Text2Story main package
The Text2Story main package contains the main classes and methods for the T2S pipeline: from text to formal representation to visualization or other representation.

- **Relation to Brat2Viz**
The Text2Story package is a generalization of Brat2Viz and should in fact contain all the funcionalities and variants of the T2S project output.

## Installation


### Language and OS Requirements

Text2Story package is written entirely in Python 3.8 modules ensuring compatibility with UNIX type Operating systems.

### Swap Size

T2S is an NLP project, which means that is intended to operate over large amounts of data using complex models, some of the third-party libraries that demand great computing resources.

To ensure enough computation power, you should use a computer where the sum of physical and virtual RAM should be at least 16GB.

* ####  [How to increase swap/virtual memory size in Linux systems](https://askubuntu.com/questions/178712/how-to-increase-swap-space)

### Steps for installation

1. Create a virtual enviroment with the following command
   ```bash
   python3.8 -m venv venv    
   ```
2. Activate the virtual enviroment with the following command
   ```bash
   source venv/bin/activate 
   ```
3. Installation of py_heideltime package (more detailed instructions in https://github.com/JMendes1995/py_heideltime)
   ```bash
    pip install git+https://github.com/JMendes1995/py_heideltime.git
   ```
4. Give tree parser of py_heideltime package permission to execute
   ```bash
    chmod +x $(VENV_HOME)/lib/python3.8/site-packages/py_heideltime/Heideltime/TreeTaggerLinux/bin/tree-tagger
   ```
5. Installation of plantuml package, which is used in the visualization.
   ```
   pip install git+https://github.com/SamuelMarks/python-plantuml#egg=plantuml
   ```
6. Installation of the text2story package.
   ```bash
     python -m pip install text2story
   ```

The following steps are optional to use the text2story package, but essential to run the our TLDR Python notebook locally (https://bit.ly/3s36Bxf).

7. Adding virtual enviroment to Jupyter Notebook.
   ```bash
      python3.8 -m pip install --user ipykernel
   ```

8. Adding your virtual enviroment to Jupyter.
   ```bash
      python -m ipykernel install --user --name=venv
   ```

9. Changing the kernel in the Jupyter, by cliking in Kernel -> Change Kernel -> (kernel name).



### Usage


```python
import text2story as t2s # Import the package

t2s.start('en') # Load the pipelines in en language

text = 'On Friday morning, Max Healthcare, which runs 10 private hospitals around Delhi, put out an "SOS" message, saying it had less than an hour\'s supply remaining at two of its sites. The shortage was later resolved.'

doc = t2s.Narrative('en', text, '2020-05-30')

doc.extract_actors('spacy') # Extraction done with just the SPACY tool.

doc.extract_times() # Extraction done with all tools (same as specifying 'py_heideltime', since we have just one tool to extract timexs)


doc.extract_events('allennlp') # Extraction of events with allennlp tool
doc.extract_semantic_role_link('allennlp') # Extraction of semantic role links with all tools (should be done after extracting events since most semantic relations are between an actor and an event)

doc.ISO_annotation('annotations.ann') # Outputs ISO annotation in .ann format (txt) in a file called 'annotations.ann', which is a standard of BRAT annotation tool


```

## Examples: Python Notebooks

A  basic notebook that teaches how to use our reader of annotations, which format is assumed is to be in the BRAT format is in the following link: [How to read a BRAT file](https://colab.research.google.com/drive/1_jc6SJNAdWMYBMVlGPldFDmGNg4gFUCs?usp=sharing).

There is the TLDR Python notebook, which extracts the main narrative elements and draw an MSC visulization:  [Too Long Didn't Read Tutorial](https://bit.ly/3s36Bxf).

Finally, there is a notebook that shows how to produce a bubble visualization: [How To: Bubble Visualization](https://colab.research.google.com/drive/1V2DCuP1qAlwUXThTKNUnZ98WxARZXC_v?usp=sharing).

## Structure
.
│   README.md
|   env.yml
│   requirements.txt
|   pyproject.toml
|   MANIFEST.in
|   LICENSE
|
└── src
    └─ text2story
        └─ core
         │   annotator.py (META-annotator)
         │   entity_structures.py (ActorEntity, TimexEntity and EventEntity classes)
         |   exceptions.py (Exceptions raised by the package)
         |   link_structures.py (TemporalLink, AspectualLink, SubordinationLink, SemanticRoleLink and ObjectalLink classes)
         |   narrative.py (Narrative class)
         |   utils.py (Utility functions)
         
        └─ annotators (tools supported by the package to do the extractions)
         |   NLTK
         │   PY_HEIDELTIME
         |   SPACY
	 |   ALLENNLP
	 |   CUSTOMPT (A CRF customized model to detect events in the Portuguese language)
         
        └─ brat2viz (tool devoted to create visual representations of ann files)
         |   brat2drs (scripts that do a conversion from a brat stand off format (.ann) to DRS format)
         │   drs2viz (scripts that do a conversion from drs format to a visual representation)

        └─ readers (module dedicated to read different kind of corpora)
         |   fn-lirics.json (conversion map from framenet to lirics: semlink project -> https://github.com/cu-clear/semlink)
         |   pb-vn2.json   (conversion map from propbank to verbnet: semlink project -> https://github.com/cu-clear/semlink)
         |   vn-lirics.json (conversion map from verbnet to lirics: semlink project -> https://github.com/cu-clear/semlink)
         |   read_brat.py  (read brat stand off format)
         |   read_ecb.py  (read ecb+ format)
         |   read_framenet.py  (read nltk data of framenet dataset)
         |   read_propbank.py  (read nltk data of propbank dataset)
         |   read.py  (META-reader)
         |   token_corpus.py  (Token representation of data)
         |   utils.py  (Utility functions for readers)

        └─ experiments (module dedicated to perform batch experiments with narrative datasets)
         |   evaluation.py  (It performs experiments in only one dataset)
         |   metrics.py   (It implements some metrics for classification recall, precision, and f1. Strict and relaxed versions (ref. Semeval-2013 task 1: Tempeval-3))
         |   run_experiments.py  (It implements batch experiments for narrative datasets)
         |   stats.py (It implements methods to evaluate some statistics of narrative datasets)



### Annotators
All annotators have the same interface: they implement a function called 'extract_' followed by the name of the particular extraction.
E.g., if they are extracting actors, then they implement a function named 'extract_actors', with two arguments: the language of text and the text itself.

|  Extractions |           Interface                                      |     Supporting tools  |
|      ---     |             ---                                          |           ---         |
|     Actor    | extract_actors(lang, text)                               | SPACY, NLTK 	  |
|    Timexs    | extract_timexs(lang, text, publication_time)             |      PY_HEIDELTIME    |
| ObjectalLink | extract_objectal_links(lang, text, publication_time)     |        ALLENNLP       |
|     Event    | extract_events(lang, text, publication_time)             | ALLENNLP, CUSTOMPT    |
| SemanticLink | extract_semantic_role_link(lang, text, publication_time) |        ALLENNLP       |

To **change some model used in the supported tools**, just go to text2story/annotators/ANNOTATOR_TO_BE_CHANGED and change the model in the file: \_\_init\_\_.py.

To **add a new tool**, add a folder to text2story/annotators with the name of the annotator all capitalized (just a convention; useful to avoid name colisions).
In that folder, create a file called '\_\_init\_\_.py' and there implement a function load() and the desired extraction functions.
The function load() should load the pipeline to some variable defined by you, so that, every time we do an extraction, we don't need to load the pipeline all over again. (Implement it, even if your annotator doesn't load anything. Leave it with an empty body.)

In the text2story.annotators.\_\_init\_\_.py file, add a call to the load() function, and to the extract functions.
(See the already implemented tools for guidance.)

And it should be done.

PS: Don't forget to normalize the labels to our semantic framework!


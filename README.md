# Text2Story main package
The Text2Story main package contains the main classes and methods for the T2S pipeline: from text to formal representation to visualization or other representation.

- **Relation to Brat2Viz**
The Text2Story package is a generalization of Brat2Viz and should in fact contain all the funcionalities and variants of the T2S project output.

## Table of Contents

1. [ Getting Start. ](#start)
2. [ The Framework Structure. ](#structure)
3. [ The Annotators. ](#annotators)
4. [ Installation. ](#installation)
   - 4.1\. [ Linux Ubuntu ](#installationlinux)
   - 4.2\. [ Windows ](#installationwindows)
5. [ The Web App. ](#webapp)

<a name="start"></a>
## 1. Getting Started

The main goal of the text2story is to extract narrative from raw text. The narrative 
components comprise events, the participants (or actors) in the events, and the time expressions. 



*   **Event**: Eventuality that happens or occurs or state or circumstance that is temporally relevant
*   **Time**: Temporal expressions that represent units of time.
*   **Participants**: Named entities, or participants, that play an important role in the event or state.


These elements relate to each other by some relations, like Semantic Role Links 
and Objectal Links. 

*  **Objectal Links**: It states how two discourse entities are referentially related to one another. For instance, there is the "identity" objectal link, which links entities that refer to the same referents, and there is the "part of" objectal link, which links a referent that is part of another.
*  **Semantic Role Links**: The identification of the way an entity is involved/participates in an
eventuality. For instance, there is the "agent" semantic role link, in which an event is linked to
a participant that intentionally caused it.

For more details, please read the following paper:

> Silvano, P., Leal, A., Silva, F., Cantante, I., Oliveira, F., & Jorge, A. M. (2021, June). Developing a multilayer 
semantic annotation scheme based on ISO standards for the visualization of a newswire corpus. 
In Proceedings of the 17th Joint ACL-ISO Workshop on Interoperable Semantic Annotation (pp. 1-13)..

A simple code to perform the extraction of the narrative elements, and the two type of relations
described above is like the following.

```python
import text2story as t2s # Import the package

t2s.start() # Load the pipelines

text = 'On Friday morning, Max Healthcare, which runs 10 private hospitals around Delhi, put out an "SOS" message, saying it had less than an hour\'s supply remaining at two of its sites. The shortage was later resolved.'

doc = t2s.Narrative('en', text, '2020-05-30')

doc.extract_actors() # Extraction done with all tools.
doc.extract_actors('spacy', 'nltk') # Extraction done with the SPACY and NLTK tools.
doc.extract_actors('allennlp') # Extraction done with just the ALLENNLP tool.

doc.extract_times() # Extraction done with all tools (same as specifying 'py_heideltime', since we have just one tool to extract timexs)

doc.extract_objectal_links() # Extraction of objectal links from the text with all tools (needs to be done after extracting actors, since it requires actors to make the co-reference resolution)

doc.extract_events() # Extraction of events with all tools
doc.extract_semantic_role_link() # Extraction of semantic role links with all tools (should be done after extracting events since most semantic relations are between an actor and an event)

ann_str = doc.ISO_annotation() # Outputs ISO annotation in .ann format (txt) in a file called 'annotations.ann'
with open('annotations.ann', "w") as fd:
    fd.write(ann_str)
```
<a name="structure"></a>
## 2. Framework Structure
```
.
│   README.md
|   env.yml
│   requirements.txt
|
└──Text2Story
      └──core
      │   │   annotator.py (META-annotator)
      │   │   entity_structures.py (ActorEntity, TimexEntity and EventEntity classes)
      │   |   exceptions.py (Exceptions raised by the package)
      │   |   link_structures.py (TemporalLink, AspectualLink, SubordinationLink, SemanticRoleLink and ObjectalLink classes)
      │   |   narrative.py (Narrative class)
      │   |   utils.py (Utility functions)
      │
      └───readers (tools to support the reading of some specific kind of annotated corpus)
      |   | read.py (Abstract class: defines the structure of a reader)
      |   | TokenCorpus (Internal representation of a token, its annotations and relations)
      |   | read_brat.py (it reads annotated file of type supported the BRAT annotation tool)
      |   | read_ecb.py (it processes ecb+ corpus format)
      |   | read_framenet.py (it processes Framenet corpus format)
      |   | read_propbank.py (it processes Propbank corpus format)  
      └───annotators (tools supported by the package to do the extractions)
      |   |   NLTK
      |   │   PY_HEIDELTIME
      |   |   SPACY
      |   |   ALLENNLP
      └───experiments
      |   |   evaluation.py (it performs batch evaluation of narrative corpora)
      |   |   metrics.py (it implements some specific metrics, like relaxed recall and relaxed precision)
      |   |   stats.py (it counts some narrative elements, and produce some stats of the narrative corpora)
      └───visualization
      |   |   brat2viz: a module that converts a BRAT annotation file to visual representations, like  Message Sequence Chart (MSC) and (Knowledge Graph) KG
      |   |   viz: a module that contain bubble_tikz.py, a class dedicate to build Bubble diagrams
      
└── Webapp
      |  backend.py
      |  main.py
      |  session_state.py
      |  input_phase.py
      |  output_phase.py

```


<a name="annotators"></a>
## 3. The Annotators
All annotators have the same interface: they implement a function called 'extract_' followed by the name of the particular extraction.
E.g., if they are extracting actors, then they implement a function named 'extract_actors', with two arguments: the language of text and the text itself.

|  Extractions |           Interface                                      | Supporting tools        |
|      ---     |             ---                                          |-------------------------|
|     Actor    | extract_actors(lang, text)                               | SPACY,  NLTK , ALLENNLP |
|    Timexs    | extract_timexs(lang, text, publication_time)             | PY_HEIDELTIME           |
| ObjectalLink | extract_objectal_links(lang, text, publication_time)     | ALLENNLP                |
|     Event    | extract_events(lang, text, publication_time)             | ALLENNLP                |
| SemanticLink | extract_semantic_role_link(lang, text, publication_time) | ALLENNLP                |

To **change some model used in the supported tools**, just go to text2story/annotators/ANNOTATOR_TO_BE_CHANGED and change the model in the file: \_\_init\_\_.py.

To **add a new tool**, add a folder to text2story/annotators with the name of the annotator all capitalized (just a convention; useful to avoid name colisions).
In that folder, create a file called '\_\_init\_\_.py' and there implement a function load() and the desired extraction functions.
The function load() should load the pipeline to some variable defined by you, so that, every time we do an extraction, we don't need to load the pipeline all over again. (Implement it, even if your annotator doesn't load anything. Leave it with an empty body.)

In the text2story.annotators.\_\_init\_\_.py file, add a call to the load() function, and to the extract functions.
(See the already implemented tools for guidance.)

And it should be done.

PS: Don't forget to normalize the labels to our semantic framework!

<a name="installation"></a>
## 4. Installation

<a name="installationlinux"></a>
### 4.1 Linux / Ubuntu

The installation requires graphviz software, the latex suite and the software poppler to convert pdf to png. 
In Linux, to install these software open a terminal and type the following commands:

```
sudo apt-get install graphviz libgraphviz-dev texlive-latex-base  texlive-latex-extra poppler-utils
```

After that, create a virtual enviroment using venv or other tool of your preference. For instance, 
using the following command in the prompt line:

```
$ python3 -m venv venv
```

Then, activate the virtual enviroment in the prompt line. Like, the following command:

```
$ source venv/bin/activate
```

After that, you are ready to install 

<a name="windows"></a>
### 4.2 Windows

First, make sure you have Microsoft C++ Build Tools. Then install graphviz software by download one suitable version 
in this [link](https://graphviz.org/download/#windows). Next, install the latex-suite like these 
[tutorial](https://www.tug.org/texlive/windows.html#install) explains. Then, install Popple packed for windows, 
which you download [here](https://github.com/oschwartz10612/poppler-windows).

Finnally, you can install text2story using pip. If it did not recognize the graphviz installation, then you can 
use the following command.

```
pip install text2story  --global-option=build_ext --global-option="-IC:\Program Files\Graphviz\include" --global-option="-LC:\Program Files\Graphviz\lib\"
```

<a name="webapp"></a>
## Web App
```
#### Web app
```ssh
python backend.py
streamlit run main.py
```
and a page on your browser will open!



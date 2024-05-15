import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text2story.core.exceptions import InvalidTool


from collections import ChainMap

sys.path.insert(0, Path(__file__).parent)

PARTICIPANT_EXTRACTION_TOOLS = {'spacy':['pt','en'], 'nltk':['en'], 'allennlp':['en'],'bertnerpt':['pt'], 'dbpedia':['pt','en'],'srl':['pt']}
TIME_EXTRACTION_TOOLS = {'py_heideltime':['pt','en'], 'tei2go':['pt','en','it','de','es','fr']}
EVENT_EXTRACTION_TOOLS = {'allennlp':['en'],"srl":["pt"]}
OBJECTAL_LINKS_RESOLUTION_TOOLS = {'allennlp':['en']}
SEMANTIC_ROLE_LABELLING_TOOLS = {'allennlp':['en'],"srl":["pt"]}

def get_tools():
    return dict(ChainMap(PARTICIPANT_EXTRACTION_TOOLS, TIME_EXTRACTION_TOOLS, \
             EVENT_EXTRACTION_TOOLS, OBJECTAL_LINKS_RESOLUTION_TOOLS, \
                         SEMANTIC_ROLE_LABELLING_TOOLS))

def load(lang, tools=None):
    """

    It loads models associated with the tools aggregated in the text2story pipeline

    @param str lang: The language (pt, en, fr, de, it, es)
    @param [str] tools(optional): a list of the tools (options are: spacy, nltk, py_heideltime, allennlp, bertnerpt, tei2go)
    to be employed, if None is given, all availables tools for that language are loaded
    @return: None
    """

    available_tools = get_tools()
    if tools is None:
        tools = available_tools

    # the importing of libraries is made upon use
    # the user, thus, has to check the installation
    if lang in tools["spacy"] and "spacy" in tools:
        from text2story.annotators import SPACY
        SPACY.load(lang)
    if lang in tools["nltk"] and "nltk" in tools:
        from text2story.annotators import NLTK
        NLTK.load(lang)
    if lang in tools["py_heideltime"] and "py_heideltime" in tools:
        from text2story.annotators import PY_HEIDELTIME
        PY_HEIDELTIME.load(lang)
    if lang in tools["allennlp"] and "allennlp" in tools:
        from text2story.annotators import ALLENNLP
        ALLENNLP.load(lang)
    if lang in tools["bertnerpt"] and "bertnerpt" in tools:
        from text2story.annotators import BERTNERPT
        BERTNERPT.load(lang)
    if lang in tools["tei2go"]  and "tei2go" in tools:
        from text2story.annotators import TEI2GO
        TEI2GO.load(lang)
    if lang in tools["dbpedia"]  and "dbpedia" in tools:
        from text2story.annotators import DBPEDIA
        DBPEDIA.load(lang)
    if lang in tools["srl"]  and "srl" in tools:
        from text2story.annotators import SRL
        SRL.load(lang)

def get_available_tools():
    return PARTICIPANT_EXTRACTION_TOOLS.keys()+\
            TIME_EXTRACTION_TOOLS.keys()+\
            EVENT_EXTRACTION_TOOLS.keys()+\
            OBJECTAL_LINKS_RESOLUTION_TOOLS.keys()+\
            SEMANTIC_ROLE_LABELLING_TOOLS.keys()

def get_participant_tools():
    return PARTICIPANT_EXTRACTION_TOOLS

def get_time_tools():
    return TIME_EXTRACTION_TOOLS

def get_event_tools():
    return EVENT_EXTRACTION_TOOLS

def get_srlink_tools():
    return SEMANTIC_ROLE_LABELLING_TOOLS

def extract_participants(tool, lang, text, url=None):

    if tool == 'spacy':
        from text2story.annotators import SPACY
        return SPACY.extract_participants(lang, text)
    elif tool == 'nltk' and lang == "en":
        from text2story.annotators import NLTK
        return NLTK.extract_participants(lang, text)
    elif tool == 'allennlp':
        from text2story.annotators import ALLENNLP
        return ALLENNLP.extract_participants(lang, text)
    elif tool == 'bertnerpt':
        from text2story.annotators import BERTNERPT
        return BERTNERPT.extract_participants(lang, text)
    elif tool == 'dbpedia':
        from text2story.annotators import DBPEDIA
        return DBPEDIA.extract_participants(text, lang, url)
    if tool == 'srl':
        from text2story.annotators import SRL
        return SRL.extract_participants(lang, text)
    else:
        raise InvalidTool


def extract_times(tool, lang, text, publication_time):
    if tool == 'py_heideltime':
        from text2story.annotators import PY_HEIDELTIME
        return PY_HEIDELTIME.extract_times(lang, text, publication_time)
    elif tool == 'tei2go':
        from text2story.annotators import TEI2GO
        return TEI2GO.extract_times(lang, text)

    raise InvalidTool


def extract_objectal_links(tool, lang, text):
    if tool == 'allennlp':
        from text2story.annotators import ALLENNLP
        olink_lst =  ALLENNLP.extract_objectal_links(lang, text)
        return olink_lst

    raise InvalidTool


def extract_events(tool, lang, text):
    if tool == 'allennlp':
        from text2story.annotators import ALLENNLP
        return ALLENNLP.extract_events(lang, text)
    if tool == 'custompt':
        from text2story.annotators import CUSTOMPT
        return CUSTOMPT.extract_events(lang, text)
    if tool == 'srl':
        from text2story.annotators import SRL
        return SRL.extract_events(lang, text)

    raise InvalidTool


def extract_semantic_role_links(tool, lang, text):
    if tool == 'allennlp':
        from text2story.annotators import ALLENNLP
        return ALLENNLP.extract_semantic_role_links(lang, text)
    if tool == 'srl':
        from text2story.annotators import SRL
        return SRL.extract_semantic_role_links(lang, text)

    raise InvalidTool

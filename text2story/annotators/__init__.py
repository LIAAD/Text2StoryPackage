import sys
from pathlib import Path

from text2story.core.exceptions import InvalidTool
from text2story.annotators import SPACY, NLTK, SPARKNLP
from text2story.annotators import PY_HEIDELTIME, ALLENNLP, CUSTOMPT
from text2story.annotators import SRLWeakLabeling

sys.path.insert(0, Path(__file__).parent)

ACTOR_EXTRACTION_TOOLS = ['spacy', 'nltk', 'sparknlp','allennlp','srlweaklabeling']
TIME_EXTRACTION_TOOLS = ['py_heideltime']
EVENT_EXTRACTION_TOOLS = ['allennlp','custompt']
OBJECTAL_LINKS_RESOLUTION_TOOLS = ['allennlp']
SEMANTIC_ROLE_LABELLING_TOOLS = ['allennlp']


def load(lang):
    SPACY.load(lang)
    NLTK.load(lang)
    #SPARKNLP.load(lang)
    PY_HEIDELTIME.load(lang)
    ALLENNLP.load(lang)
    CUSTOMPT.load(lang)

def get_available_tools():
    return ACTOR_EXTRACTION_TOOLS+\
            TIME_EXTRACTION_TOOLS+\
            EVENT_EXTRACTION_TOOLS+\
            OBJECTAL_LINKS_RESOLUTION_TOOLS+\
            SEMANTIC_ROLE_LABELLING_TOOLS

def get_participant_tools():
    return ACTOR_EXTRACTION_TOOLS

def get_time_tools():
    return TIME_EXTRACTION_TOOLS

def get_event_tools():
    return EVENT_EXTRACTION_TOOLS

def get_srlink_tools():
    return SEMANTIC_ROLE_LABELLING_TOOLS

def extract_actors(tool, lang, text):

    if tool == 'spacy':
        return SPACY.extract_actors(lang, text)
    elif tool == 'nltk':
        return NLTK.extract_actors(lang, text)
    elif tool == 'allennlp':
        return ALLENNLP.extract_actors(lang, text)
    elif tool == 'srlweaklabeling':
        return SRLWeakLabeling.extract_actors(lang, text)
    #elif tool == 'sparknlp':
    #    return SPARKNLP.extract_actors(lang, text)

    raise InvalidTool


def extract_times(tool, lang, text, publication_time):
    if tool == 'py_heideltime':
        return PY_HEIDELTIME.extract_times(lang, text, publication_time)

    raise InvalidTool


def extract_objectal_links(tool, lang, text):
    if tool == 'allennlp':
        return ALLENNLP.extract_objectal_links(lang, text)

    raise InvalidTool


def extract_events(tool, lang, text):
    if tool == 'allennlp':
        return ALLENNLP.extract_events(lang, text)
    if tool == 'custompt':
        return CUSTOMPT.extract_events(lang, text)

    raise InvalidTool


def extract_semantic_role_links(tool, lang, text):
    if tool == 'allennlp':
        return ALLENNLP.extract_semantic_role_links(lang, text)

    raise InvalidTool

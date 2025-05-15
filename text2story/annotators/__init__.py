import importlib
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text2story.core.exceptions import InvalidNarrativeComponent, DuplicateNarrativeComponent, InvalidTool
from typing import List
from types import ModuleType

from collections import ChainMap

sys.path.insert(0, str(Path(__file__).parent))

PARTICIPANT_EXTRACTION_TOOLS = {'spacy':['pt','en'], 'nltk':['en'], 'bertnerpt':['pt'], 'dbpedia':['pt','en'],'srl':['pt','en']}
#TIME_EXTRACTION_TOOLS = {'py_heideltime':['pt','en'], 'tei2go':['pt','en','it','de','es','fr']}
TIME_EXTRACTION_TOOLS = {'py_heideltime':['pt','en']}
EVENT_EXTRACTION_TOOLS = {"srl":["pt","en"]}
#OBJECTAL_LINKS_RESOLUTION_TOOLS = {'allennlp':['en']}
OBJECTAL_LINKS_RESOLUTION_TOOLS = {}
SEMANTIC_ROLE_LABELLING_TOOLS = {"srl":["pt","en"]}

LOCAL_ANNOTATORS = set()

def get_tools():
    return dict(ChainMap(PARTICIPANT_EXTRACTION_TOOLS, TIME_EXTRACTION_TOOLS, \
             EVENT_EXTRACTION_TOOLS, OBJECTAL_LINKS_RESOLUTION_TOOLS, \
                         SEMANTIC_ROLE_LABELLING_TOOLS))

def get_tool_list(tool_type:str):
    """

    @param tool_type: given a type of a tool , i.e, one of the following items ['participant','time', 'event','objectal_links',
                     'semantic_links']
    @return: the dictionary of tools names or None if it is not found
    """

    if tool_type == 'participant':
        return PARTICIPANT_EXTRACTION_TOOLS
    elif tool_type == 'event':
        return EVENT_EXTRACTION_TOOLS
    elif tool_type == 'time':
        return TIME_EXTRACTION_TOOLS
    elif tool_type == 'objectal_links':
        return OBJECTAL_LINKS_RESOLUTION_TOOLS
    elif tool_type == 'semantic_links':
        return  SEMANTIC_ROLE_LABELLING_TOOLS
    else:
        raise InvalidNarrativeComponent(tool_type)
def add_tool(tool_name:str, lang_lst:List[str], tool_types:List[str]):
    """
    It adds a tool to the gallery of annotators.

    @param tool: the name of the module where is your annotator
    @param lang_lst: a list of languages supported by your custom annotator,
                        for instance, ['en','fr']
    @param tool_types: a list of types that your annotator labels. The
                     supported types are ['participant','time', 'event','objectal_links',
                     'semantic_links']
    @return: None
    """

    LOCAL_ANNOTATORS.add(tool_name)

    for ttype in tool_types:
        tool_list = get_tool_list(ttype)
        if tool_name not in tool_list:
            tool_list[tool_name] = lang_lst
        else:
            raise DuplicateNarrativeComponent(tool_name)

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

    # loading the local annotators
    for tl in tools:
        if tl in LOCAL_ANNOTATORS and lang in tools[tl]:
            local_annotator = importlib.import_module(tl)
            local_annotator.load(lang)

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
    if lang in tools["bertnerpt"] and "bertnerpt" in tools:
        from text2story.annotators import BERTNERPT
        BERTNERPT.load(lang)
    #if lang in tools["tei2go"]  and "tei2go" in tools:
    #    from text2story.annotators import TEI2GO
    #    TEI2GO.load(lang)
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

    if tool in LOCAL_ANNOTATORS:
        local_annotator = importlib.import_module(tool)
        return local_annotator.extract_participants(lang, text)

    if tool == 'spacy':
        from text2story.annotators import SPACY
        return SPACY.extract_participants(lang, text)
    elif tool == 'nltk' and lang == "en":
        from text2story.annotators import NLTK
        return NLTK.extract_participants(lang, text)
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

    if tool in LOCAL_ANNOTATORS:
        local_annotator = importlib.import_module(tool)
        return local_annotator.extract_times(lang, text)

    if tool == 'py_heideltime':
        from text2story.annotators import PY_HEIDELTIME
        return PY_HEIDELTIME.extract_times(lang, text, publication_time)
    elif tool == 'tei2go':
        from text2story.annotators import TEI2GO
        return TEI2GO.extract_times(lang, text)

    raise InvalidTool


def extract_objectal_links(tool, lang, text):

    if tool in LOCAL_ANNOTATORS:
        local_annotator = importlib.import_module(tool)
        return local_annotator.extract_objectal_links(lang, text)


    raise InvalidTool

def extract_events(tool, lang, text):

    if tool in LOCAL_ANNOTATORS:
        local_annotator = importlib.import_module(tool)
        return local_annotator.extract_events(lang, text)

    if tool == 'custompt':
        from text2story.annotators import CUSTOMPT
        return CUSTOMPT.extract_events(lang, text)
    if tool == 'srl':
        from text2story.annotators import SRL
        return SRL.extract_events(lang, text)

    raise InvalidTool


def extract_semantic_role_links(tool, lang, text):

    if tool in LOCAL_ANNOTATORS:
        local_annotator = importlib.import_module(tool)
        return local_annotator.extract_semantic_role_links(lang, text)

    if tool == 'srl':
        from text2story.annotators import SRL
        return SRL.extract_semantic_role_links(lang, text)

    raise InvalidTool

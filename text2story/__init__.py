# Version of the text2story package
__version__ = "1.0.0"

from typing import List

from text2story.annotators import load
from text2story.annotators import get_available_tools
from text2story.annotators import get_participant_tools
from text2story.annotators import get_event_tools
from text2story.annotators import get_time_tools
from text2story.annotators import get_srlink_tools
from text2story.annotators import add_tool, get_tool_list

def get_tools(narrative_component:str):
    """
    get the tools for the given narrative component
    @param narrative_component:
    @return: a dicionary of tools and the languages it process
    """
    get_tool_list(narrative_component)

def add_annotator(tool_name:str, lang_lst:List[str], tool_types:List[str]):
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

    add_tool(tool_name,lang_lst,tool_types)

def start(lang):
    load(lang)

def get_tools_name():
    return get_available_tools()

def participant_tool_names():
    return get_participant_tools()

def event_tool_names():
    return get_event_tools()

def time_tool_names():
    return get_time_tools()

def srlink_tool_names():
    return get_srlink_tools()


# Export to out of the package
from text2story.core.narrative import Narrative


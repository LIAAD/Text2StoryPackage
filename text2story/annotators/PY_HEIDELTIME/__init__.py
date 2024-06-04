'''
    PY_HEIDELTIME annotator (https://github.com/JMendes1995/py_heideltime)

    Used for:
        - Timexs extraction
            'en' : default
            'pt' : default
'''
import traceback

#from text2story.core.exceptions import InvalidLanguage

from py_heideltime import py_heideltime
import re
import sys

def load(lang):
    """
    Used, at start, to load the pipeline for the supported languages.
    """
    pass # Nothing to load


def extract_times(lang, text, publication_time):
    """
    Parameters
    ----------
    lang : str
        the language of text to be annotated

    text : str
        the text to be annotated

    Returns
    -------
    list[tuple[tuple[int, int], str, str]]
        a list consisting of the times identified, where each time is represented by a tuple
        with the start and end character offset, it's value and type, respectively

    Raises
    ------
    InvalidLanguage if the language given is invalid/unsupported
    """

    if lang not in ['en', 'pt']:
        #raise InvalidLanguage
        print("Invalid Language")
        raise sys.exit(0)

    lang_mapping = {'pt' : 'Portuguese', 'en' : 'English'}
    lang = lang_mapping[lang]

    # annotations = py_heideltime(re.escape(text), language=lang, document_creation_time=publication_time)
    try:
        annotations = py_heideltime.heideltime(text, lang, "news")
    except IndexError as e:
        #traceback.print_exc()
        annotations = []
        print()
        print("WARNING: no time expression processed in this document for some bug in py_heideltime.")

    timexs = []
    for ann in annotations:
        start, end = ann["span"][0], ann["span"][1]
        timex = ((start,end), ann["type"], ann["text"])
        timexs.append(timex)

    return timexs
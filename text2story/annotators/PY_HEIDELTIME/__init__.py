'''
    PY_HEIDELTIME annotator (https://github.com/JMendes1995/py_heideltime)

    Used for:
        - Timexs extraction
            'en' : default
            'pt' : default
'''

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
    annotations = py_heideltime(text, language=lang, document_creation_time=publication_time)

    pattern = r'<TIMEX3 tid=.*? type=.*? value=.*?>.*?</TIMEX3>'

    timexs_list = re.finditer(pattern, annotations[2])

    timexs = []

    char_offset = 0

    for timex in timexs_list:
        match = timex.group()

        i = 13 # Discard all till the starting of the 'tid' value
        while match[i] != '"': # Consume the 'tid' value
            i += 1

        i += 8 # Now, we are at the field 'type'
        timex_type = ''
        while match[i] != '"':
            timex_type += match[i]
            i += 1

        i += 9 # Now, we are the field 'value'
        timex_value = ''
        while match[i] != '"':
            timex_value += match[i]
            i += 1

        # Now, we are the text of the timex
        while match[i] != '>': #ignore everything until it reaches the end of the tag
            i += 1
        i += 1

        timex_text = ''
        while match[i] != '<':
            timex_text += match[i]
            i += 1

        char_offset = text.find(timex_text, char_offset)

        timex_start_offset = char_offset
        timex_end_offset = char_offset + len(timex_text)
        timex_character_span =  (timex_start_offset, timex_end_offset)
        char_offset = timex_end_offset

        timex = (timex_character_span, timex_type, timex_value)
        timexs.append(timex)

    return timexs

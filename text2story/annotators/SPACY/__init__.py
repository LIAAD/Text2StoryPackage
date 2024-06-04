"""
    spaCy annotator

    Used for:
        - Actor extraction
            'pt' : https://spacy.io/models/en#en_core_web_lg)
            'en' : https://spacy.io/models/en#en_core_web_lg)
"""

from text2story.core.utils import chunknize_actors, normalize_tag
from text2story.core.exceptions import InvalidLanguage

import spacy

pipeline = {}

def load(lang):
    """
    Used, at start, to load the pipeline for the supported languages.
    """
    if lang == "pt":
        if not(spacy.util.is_package('pt_core_news_lg')):
            spacy.cli.download('pt_core_news_lg')
        pipeline['pt'] = spacy.load('pt_core_news_lg')

    elif lang == "en":
        if not(spacy.util.is_package('en_core_web_lg')):
            spacy.cli.download('en_core_web_lg')

        pipeline['en'] = spacy.load('en_core_web_lg')
    else:
        raise InvalidLanguage(lang)

    
def extract_participants(lang, text):
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
        the list of actors identified where each actor is represented by a tuple

    Raises
    ------
        InvalidLanguage if the language given is invalid/unsupported
    """

    if lang not in ['pt', 'en']:
        raise InvalidLanguage(lang)

    doc = pipeline[lang](text)

    iob_token_list = []
    for token in doc:
        start_character_offset = token.idx
        end_character_offset = token.idx + len(token)
        character_span = (start_character_offset, end_character_offset)
        pos = normalize_tag(token.pos_)
        ne = token.ent_iob_ + "-" + normalize_tag(token.ent_type_) if token.ent_iob_ != 'O' else 'O'

        iob_token_list.append((character_span, pos, ne))

    actor_list = chunknize_actors(iob_token_list)

    return actor_list  



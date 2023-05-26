"""
    spaCy annotator

    Used for:
        - Actor extraction
            'pt' : https://spacy.io/models/en#en_core_web_lg)
            'en' : https://spacy.io/models/en#en_core_web_lg)
"""

from text2story.core.utils import chunknize_actors
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
        spacy.cli.download('en_core_web_sm')
        if not(spacy.util.is_package('en_core_web_lg')):
            spacy.cli.download('en_core_web_lg')

        pipeline['en'] = spacy.load('en_core_web_lg')
    else:
        raise InvalidLanguage(lang)

    
def extract_actors(lang, text):
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
        pos = normalize(token.pos_)
        ne = token.ent_iob_ + "-" + normalize(token.ent_type_) if token.ent_iob_ != 'O' else 'O'

        iob_token_list.append((character_span, pos, ne))

    actor_list = chunknize_actors(iob_token_list)

    return actor_list  


def normalize(label):
    """
    Parameters
    ----------
    label : str
    
    Returns
    -------
    str
        the label normalized
    """

    mapping = {
        # POS tags
        # Universal POS Tags
        # http://universaldependencies.org/u/pos/
        
        #"ADJ": "adjective",
        #"ADP": "adposition",
        #"ADV": "adverb",
        #"AUX": "auxiliary",
        #"CONJ": "conjunction",
        #"CCONJ": "coordinating conjunction",
        #"DET": "determiner",
        #"INTJ": "interjection",
        "NOUN": "Noun",
        #"NUM": "numeral",
        #"PART": "particle",
        "PRON": "Pronoun",
        "PROPN": "Noun",
        #"PUNCT": "punctuation",
        #"SCONJ": "subordinating conjunction",
        #"SYM": "symbol",
        #"VERB": "verb",
        #"X": "other",
        #"EOL": "end of line",
        #"SPACE": "space",

        # NE
        # en
        'CARDINAL'    : 'Other', # 'Numerals that do not fall under another type'
        'DATE'        : 'Date',  # 'Absolute or relative dates or periods'
        'EVENT'       : 'Other', # 'Named hurricanes, battles, wars, sports events, etc.'
        'FAC'         : 'Loc',   # 'Buildings, airports, highways, bridges, etc.'
        'GPE'         : 'Loc',   # 'Countries, cities, states'
        'LANGUAGE'    : 'Other', # 'Any named language'
        'LAW'         : 'Other', # 'Named documents made into laws.'
        'LOC'         : 'Loc',   # 'Non-GPE locations, mountain ranges, bodies of water'
        'MONEY'       : 'Other', # 'Monetary values, including unit'
        'NORP'        : 'Other', # 'Nationalities or religious or political groups'
        'ORDINAL'     : 'Other', # '"first", "second", etc.'
        'ORG'         : 'Org',   # 'Companies, agencies, institutions, etc.'
        'PERCENT'     : 'Other', # 'Percentage, including "%"'
        'PERSON'      : 'Per',   # 'People, including fictional'
        'PRODUCT'     : 'Obj',   # 'Objects, vehicles, foods, etc. (not services)'
        'QUANTITY'    : 'Other', # 'Measurements, as of weight or distance'
        'TIME'        : 'Time',  # 'Times smaller than a day'
        'WORK_OF_ART' : 'Other', # 'Titles of books, songs, etc.'

        # pt
        # 'LOC'
        'MISC'        : 'Other', # 'Miscellaneous entities, e.g. events, nationalities, products or works of art'
        # 'ORG'
        'PER'         : 'Per' # 'People, including fictional'
    }

    return mapping.get(label, 'UNDEF')
        

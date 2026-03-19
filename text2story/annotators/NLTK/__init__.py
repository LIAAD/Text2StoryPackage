"""
    NLTK annotator

    Used for:
        - Actor extraction 
            'en' : default                
"""

from text2story.core.utils import chunknize_actors
from text2story.core.exceptions import InvalidLanguage

import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk, tree2conlltags
from nltk.data import find


def load(lang, allow_download=False):
    """
    Load NLTK resources for supported languages.

    Parameters
    ----------
    lang : str
        Language code (currently supports "en").
    allow_download : bool, optional
        If True, download missing resources (requires internet).
        Default is False (safe for offline / HPC usage).

    Raises
    ------
    RuntimeError
        If required NLTK resources are missing and downloads are disabled.
    """

    if lang != "en":
        return

    resources = {
        "chunkers/maxent_ne_chunker": "maxent_ne_chunker",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
        # Optional / version-dependent
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
        "corpora/words": "words",
    }

    missing = []

    for path, name in resources.items():
        try:
            find(path)
        except LookupError:
            if allow_download:
                nltk.download(name)
            else:
                missing.append(path)

    if missing:
        raise RuntimeError(
            "Missing required NLTK resources:\n"
            + "\n".join(missing)
            + "\n\nPlease install them using:\n"
            ">>> import nltk\n"
            ">>> nltk.download(<resource>)\n"
            "or run load(lang='en', allow_download=True) on a machine with internet."
        )


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

    if lang not in ['en']:
        raise InvalidLanguage(lang)

    language_mapping = {'en' : 'english'}

    iob_token_list = []

    char_offset = 0 
    sents = sent_tokenize(text, language=language_mapping[lang])
    
    for sent in sents:
        tree = ne_chunk(pos_tag(word_tokenize(sent, language=language_mapping[lang])))
        
        doc = tree2conlltags(tree) # doc :: [(Token, POS_TAG, IOB-NE)]
        
        for token in doc:
            token_text = token[0] # Don't call token[0] 'text', it will ofuscate the 'text' parameter!
            char_offset = text.find(token_text, char_offset) # always actualize char_offset to where we are, so next search we are doing, we are following the part of the text we didn't search yet
            char_span = (char_offset, char_offset + len(token_text))
            char_offset += len(token_text)
            pos = normalize(token[1])
            ne = token[2][:2] + normalize(token[2][2:]) if token[2] != 'O' else 'O'
            
            iob_token_list.append((char_span, pos, ne))
    
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
        # POS tags (Penn Treebank Project: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
        'NN'    : 'Noun',
        'NNS'   : 'Noun',
        'NNP'   : 'Noun',
        'NNPS'  : 'Noun',
        'PRP'   : 'Pronoun',
        'PRP$'  : 'Pronoun',
        'WP'    : 'Pronoun',
        'WP$'   : 'Pronoun',

        # NE labels
        'DATE'        : 'Date',
        'FACILITY'    : 'Loc',
        'GPE'         : 'Other',
        'GSP'         : 'Other', 
        'LOCATION'    : 'Loc',
        'MONEY'       : 'Other',
        'ORGANIZATION': 'Org',
        'PERCENT'     : 'Other',
        'PERSON'      : 'Per'
    }

    return mapping.get(label, 'UNDEF')

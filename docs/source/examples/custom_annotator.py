import spacy

from text2story.core.exceptions import UninstalledModel, InvalidLanguage
from text2story.core.utils import normalize_tag, chunknize_actors

# this stores the pipeline of models used to extract narrative components
# for a given language (whose code is the key of this dictionary)
pipeline = {}

def load(lang:str):
    """
    Definition of load method is mandatory, otherwise the package will raise errors.
    If you do not want to define it, just define an empty method with the command pass

    @param lang: The language code to load models. For instance (pt, en, fr, etc)
    @return:
    """
    if not (spacy.util.is_package('fr_core_news_lg')):
        spacy.cli.download('fr_core_news_lg')
    pipeline['fr'] = spacy.load('fr_core_news_lg')

    try:
        pipeline['fr_time'] = spacy.load(lang + "_tei2go")
    except OSError:
        model_name = lang + "_tei2go"
        command = f"pip install https://huggingface.co/hugosousa/{lang}_tei2go/resolve/main/{lang}_tei2go-any-py3-none-any.whl"
        raise UninstalledModel(model_name, command)


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

    if lang not in ['fr']:
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

def extract_times(lang, text, publication_time=None):
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
    if lang not in ["fr"]:
        raise InvalidLanguage(lang)

    timex_lst = pipeline["fr"](text).ents

    ans = []
    for timex in timex_lst:

        start = timex.start_char
        end = timex.end_char
        label = timex.label_
        text = timex.text

        ans.append(((start, end), label, text))
    return ans

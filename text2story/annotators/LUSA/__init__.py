"""
    LUSA annotator

    Used for:
        - Event extraction
            'en' : https://huggingface.co/evelinamorim/bert-lusa-eventype-classifier)
"""

from text2story.core.exceptions import InvalidLanguage

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch
import torch.nn.functional as F

pipelines = {}


def load(lang):
    """
    Used, at start, to load the pipeline for the supported languages.
    """
    if lang == "en":
        extractor = pipeline(
            "token-classification", 
            model="lfcc/lusa_events_en", 
            aggregation_strategy="simple"
        )
        
        # type of event classifier
        classifier = pipeline(
            "token-classification", 
            model="evelinamorim/bert-lusa-eventype-classifier", 
            aggregation_strategy="simple"
        )

        pipelines['en'] = {
            'extractor': extractor,
            'classifier': classifier
        }
    else:
        raise InvalidLanguage(lang)


def extract_events(lang, text):
    """
    Parameters
    ----------
    lang : str
        the language of text to be annotated
    text : str
        the text to be annotated

    Returns
    -------
    list[tuple[tuple[int, int], Dict[str,str]]]
        the list of events identified where each event is represented by its offset (tuple), and a dictionary of attributes

    Raises
    ------
        InvalidLanguage if the language given is invalid/unsupported
    """

    if lang not in ['en']:
        raise InvalidLanguage(lang)

    extractions = pipelines[lang]['extractor'](text)
    classifications = pipelines[lang]['classifier'](text)

    event_list = []
    
    for ext in extractions:
        event_type = None
        if ext['entity_group'] != "Event":
            continue
        
        
        for cls in classifications:
            if (cls['start'] <= ext['start'] < cls['end']) or \
               (ext['start'] <= cls['start'] < ext['end']) :
                if cls['score'] >=0.3:
                    event_type = cls['entity_group']
                    break
        
        if event_type is not None:
            event_list.append(((ext['start'], ext['end']), {"Type": event_type}))
        else:
            event_list.append(((ext['start'], ext['end']),{}))

    return event_list


def extract_participants(lang, text):
    """
    Main function that applies the SRL pipeline to extract participant entities from each sentence.

    @param lang: The language of the text
    @param text: The full text to be annotated
    Returns
    -------
    list[tuple[tuple[int, int], str, str]]
        the list of participants identified where each participant is represented by a tuple
    """
    if lang not in ['en']:
        raise InvalidLanguage(lang)
    
    extractions = pipelines[lang]['extractor'](text)
    participant_lst = []
    for ext in extractions:
        if ext['entity_group'] != "Participant":
            continue
        # default pos tag is Noun and the Ner type is Participant
        participant_lst.append(((ext['start'],ext['end']),"Noun","Participant"))
    return participant_lst

def extract_times(lang, text, publication_time):
    """
    Parameters
    ----------
    lang : str
        the language of text to be annotated

    text : str
        the text to be annotated
    publication_time:
        legacy parameter from py_heideltime

    Returns
    -------
    list[tuple[tuple[int, int], str, str]]
        a list consisting of the times identified, where each time is represented by a tuple
        with the start and end character offset, it's value and type, respectively

    Raises
    ------
    InvalidLanguage if the language given is invalid/unsupported
    """
    if lang not in ['en']:
        raise InvalidLanguage(lang)
    
    extractions = pipelines[lang]['extractor'](text)
    time_lst = []
    for ext in extractions:
        if ext['entity_group'] != "Time":
            continue
        # default pos tag is Noun and the Ner type is Participant
        time_lst.append(((ext['start'],ext['end']),None,ext['word']))
    return time_lst


import platform

from text2story.core.exceptions import InvalidLanguage
import nltk
import joblib
from pathlib import Path

import os
import sys


from text2story.core.utils import normalize_tag, install

try:
    import torch
    from torchcrf import CRF
except ModuleNotFoundError:
    print("Warning: If you are going to use BERT NER PT module, you should install transformers, torch  and pytorch-crf libraries.")


pipeline = {}




def load(lang):
    if lang == "pt":
        try:
            import transformers
            import torch
            from torchcrf import CRF
        except ModuleNotFoundError:
            raise Exception("It is not possible to load BERT NER PT module, since some of \\"
                   "the required libraries are not installed: transformers,  torch or pytorch-crf")


        pipeline['pt']  = transformers.pipeline(
            model="arubenruben/NER-PT-BERT-CRF-Conll2003",
            device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
            trust_remote_code=True
        )
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
    if lang == "pt":
        return extract_actors_pt(text)
    else:
        raise InvalidLanguage(lang)

def extract_actors_pt(text):
    tokens = nltk.wordpunct_tokenize(text)
    result = pipeline['pt']({"tokens":tokens})

    # pos tagger: https://github.com/inoueMashuu/POS-tagger-portuguese-nltk
    current_path = Path(__file__).parent
    tagger = joblib.load(os.path.join(current_path, 'POS_tagger_brill.pkl'))

    pos_tags = tagger.tag(tokens)

    # find the tokens positions
    current_pos = 0
    pos_lst = []
    for tok in tokens:
        current_pos = text.find(tok, current_pos)
        pos_lst.append(current_pos)

    actor_lst = []
    i = 0
    last_label = 'O'
    current_actor = []
    while i < len(result):
        if result[i] != 'O':
            if last_label == 'O' and current_actor != []:
                actor_lst.append(current_actor)
                current_actor = []

            current_actor.append(i)
        last_label = result[i]
        i += 1

    if current_actor != []:
        actor_lst.append(current_actor)
    # the answer of the method
    ans = []
    for actor in actor_lst:
        idx_fst = actor[0]
        idx_last = actor[-1]

        tok_last = tokens[idx_last]

        position_fst = pos_lst[idx_fst]
        position_last = pos_lst[idx_last] + len(tok_last)

        start = position_fst
        end = len(text[position_fst:position_last])

        category = result[idx_fst].split('-')[1]

        normalized_postag = normalize_tag(pos_tags[idx_fst][1])
        ans.append(((start, end), normalized_postag , normalize_tag(category)))


    return ans

from maverick import Maverick
from text2story.core.exceptions import InvalidLanguage
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
import torch

# Patch torch.load to force weights_only=False for all loads
_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=None, mmap=None, **pickle_load_args):
    # Force weights_only=False
    return _original_torch_load(
        f,
        map_location=map_location,
        pickle_module=pickle_module,
        weights_only=False,  # Force this
        mmap=mmap,
        **pickle_load_args
    )

torch.load = _patched_torch_load

pipeline = {}
def load(lang):
    if lang == "en":
        pipeline['coref_en'] = Maverick(hf_name_or_path="sapienzanlp/maverick-mes-ontonotes", device="cpu")
    else:
        raise InvalidLanguage(lang)

def extract_objectal_links(lang, text):
    prediction = pipeline['coref_en'].predict(text)

    cluster_indexes_list = prediction["clusters_char_offsets"]
    # for some reason, maverick model is cutting the last character
    #, so in the following line we update the end

    cluster_indexes_list = [[ (start, end + 1) for (start, end) in cluster]   for cluster in cluster_indexes_list]

    return cluster_indexes_list


import spacy
import transformers
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModel, AutoTokenizer

nltk.download('punkt')

from itertools import zip_longest

from text2story.core.utils import bsearch_tuplelist
from transformers.pipelines import PIPELINE_REGISTRY

from text2story.annotators.SRL.srl_pipeline import SrlPipeline

from text2story.annotators.SRL.srl_model.model import SRLModel
from text2story.annotators.SRL.srl_model.config import SRLModelConfig

### Global Variables ###

SRL_TYPE_MAPPING = {
    "TMP": "time",
    "LOC": "location",
    "ADV": "theme",  # Adverbial
    "MNR": "manner",
    "CAU": "cause",
    "PRP":"cause",
    "EXT": "theme",  # should be attribute -> changed for compatibility
    "DIS": "theme",  # connection of two expressions -> should be theme -> changed for compatibility
    "PNC": "purpose",
    "PR": "purpose",
    "NEG": "theme",  # should be attribute -> changed for compatibility | NEG = negation
    "DIR": "path",  # should be setting -> changed for compatibility
    "MOD": "instrument",  # MOD = Modal
    "PRD": "theme",  # Secondary predicate -> should be attribute -> changed for compatibility
    "ADJ": "theme",
    "COM": "agent",  # COM = Comitative -> used to express accompaniment
    "GOL": "goal",  # GOL = goal
    "REC": "instrument",  # REC = reciprocal
    "PAS": "passive", # passive verb
    "NSE": "agent",
    "TML": "time",
    "ASP": "theme"
}
pipeline = {}

def load(lang):
    """
    Used, at start, to load the pipeline for the supported languages.
    """
    nltk.download('punkt_tab')

    if lang == "pt":
        pipeline['srl_pt'] = transformers.pipeline(model="liaad/propbank_br_srl_albertina_100m_portuguese_ptpt_encoder",
                                               trust_remote_code=True)
        # regular expression to capture SRL tags
        pipeline["srl_re_tags"] = r".*[A][0-9]|AM"
        pipeline["ARGM_STR"] = "B-AM"
        # since allennlp does not have NER, download the spacy model
        if not (spacy.util.is_package('pt_core_news_lg')):
            spacy.cli.download('pt_core_news_lg')
        pipeline["pt"] = spacy.load('pt_core_news_lg')

    elif lang == "en":

        PIPELINE_REGISTRY.register_pipeline(
            "srl",
            pipeline_class=SrlPipeline,
            pt_model=SRLModel,
            default={"pt": ("liaad/srl-pt_bertimbau-base_hf", "main")},
            type="text",
        )
        model = AutoModel.from_pretrained(
            "liaad/srl-en_roberta-large_hf", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "liaad/srl-en_roberta-large_hf", trust_remote_code=True
        )
        #  Load the token classification pipeline with a pre-trained model
        pipeline['srl_en'] =  transformers.pipeline("srl", model=model, tokenizer=tokenizer, pipeline_class=SrlPipeline, lang="en")

        pipeline["srl_re_tags"] = r"[BI]-(A\d|ARG\d|AM-[A-Z]+|ARGM-[A-Z]+)"
        pipeline["ARGM_STR"] = "ARGM"

        # this coreference model is too heavy. Analyze another model to fit here
        #pipeline['coref_en'] =transformers.pipeline(model="VincentNLP/seq2seq-coref-t0-3b-partial-linear",
        #                                       trust_remote_code=True)
        pipeline['en'] = spacy.load('en_core_web_lg')


def process_srl_output(srl_output_list):
    """
    Aux function to process SRL english output

    @param srl_output:
    @return:
    """
    processed_output_list = []

    for srl_output in srl_output_list:
        for verb, (labels, tokens_with_labels) in srl_output.items():
            tokens = []
            predictions = []

            for token, label in tokens_with_labels:
                # Ignore <s> and </s> tokens
                if token not in ['<s>', '</s>']:
                    # Remove the leading extra token and add to the list
                    cleaned_token = token[1:] if token.startswith('Ġ') else token
                    tokens.append(cleaned_token)
                    predictions.append(label)

            output_dict = {
                "tokens": tokens,
                "predictions": predictions
            }

            processed_output_list.append(output_dict)

    return processed_output_list


def _make_srl_df(lang, text):
    """
    Make a pandas DataFrame with the results from the SRL for each sentence in the text.
    Each row of the DataFrame is a frame from the SRL.

    @param text: The full text to annotate
    @return: List of pandas DataFrames with the contents of the SRL (values) for each frame (row) by word (column)
    """
    sentences = sent_tokenize(text)

    srl = []
    # se for um texto muito grande, da problema. O Allennlp tem um limite aqui
    for sent in sentences:

        toks = nltk.word_tokenize(sent)

        # allennlp has a limitation in the srl pipeline, they only accept sentences
        # with at most 512 tokens. So in here there is a test, and if necessary it splits
        # the sentence in batches of tokens
        if len(toks) > 256:

            result = []
            batch = 1
            len_batch = 256
            start_batch = 0
            end_batch = len_batch * batch

            while end_batch < len(toks):

                batch_sent = " ".join(toks[start_batch:end_batch])
                if abs(end_batch - start_batch) < 5:
                    batch = batch + 1
                    start_batch = end_batch
                    end_batch = len_batch * batch
                    continue


                if lang == "en":
                    result = pipeline["srl_en"].predict(sentence=batch_sent)
                elif lang == "pt":
                    try:
                         result = pipeline["srl_pt"].predict_json(batch_sent)
                    except RuntimeError:
                        print("Warning: Runtime Error -- %s" % batch_sent)
                        break
                    # there is a bug for some sentences, that there is no result from the srl
                    if isinstance(result, list):
                        continue

                batch = batch + 1
                start_batch = end_batch
                end_batch = len_batch * batch
                srl.append(result)
                for r in result:
                    print(len(r))
        else:
            if lang == "en":
                result = process_srl_output(pipeline["srl_en"](sent))
            elif lang == "pt":
                result = pipeline["srl_pt"](sent)
                #  there is a bug for some sentences, that there is no result from the srl?
            else:
                result = []
            srl.append(result)

    dfs_by_sent = []
    for sentence in srl:
        try:
            tags_by_frame = pd.DataFrame(columns=sentence[0]["tokens"])
        except IndexError as e:
            if len(sentence) == 0:
                continue
            #print("-->",sentence)
            #raise

        for i, frame in enumerate(sentence):
            #print("-->", i, len(sentence[i]["tokens"]),len(sentence[i]["predictions"]), sentence[i])
            #for i, frame in zip(np.arange(len(sentence["verb"])), sentence["verb"]):
            tags_by_frame.loc[i] = frame["predictions"]

        if tags_by_frame.shape[0] != 0:
            dfs_by_sent.append(tags_by_frame)

    return dfs_by_sent

def _normalize_sent_tags(sentence_df):
    """
    Normalize the frames retrieved from the SRL from one sentence.
    Each word must have only one label.

    @param sentence_df: DataFrame of the SRL each column is a word from the sentence and each row is the results of SRL
    for one frame.
    @return: List of the normalized tags
    @return: List of booleans of whether the tag is the beginning of the argument.
    """
    normalized_tags, begin_tags = [], []
    if sentence_df.size  == 0:
        return normalized_tags, begin_tags

    for col in np.arange(len(sentence_df.columns)):
        word_vals = sentence_df.iloc[:, col]

        word_vals = word_vals[word_vals != "O"]
        if word_vals.shape[0] == 1:
            normalized_tags.append(word_vals.iloc[0])
            begin_tags.append(word_vals.iloc[0].startswith("B"))
            continue
        verb_words = word_vals[word_vals.isin(["I-V", "B-V"])]
        if verb_words.shape[0] != 0:  # a) - verbo
            normalized_tags.append(verb_words.iloc[0])
            begin_tags.append(False)  # Event
            continue
        # b) - ARGM e ARG (o último e mais especifico tem prio)
        # TODO: em português as tags estao ligeiramente diferentes:
        # B-AM-TMP, B-A1, I-A2, etc
        arg_words = word_vals[word_vals.str.contains(pipeline["srl_re_tags"])]
        if arg_words.shape[0] != 0:
            normalized_tags.append(arg_words.iloc[-1])  # desempate entre dois ARGM-X diferentes
            begin_tags.append(arg_words.iloc[-1].startswith("B"))
            continue
        else:
            print("\nNORMALIZATION ERROR - MULTIPLE TAG VALUES FOUND FOR WORD.")
            print(word_vals.values)
            print(sentence_df)
    #print(">>>",begin_tags, sentence_df)
    #print("-->",normalized_tags)
    return normalized_tags, begin_tags

def _find_events(normalized_tags, verb_tags, event_threshold=2):
    """
    Find words that belong to the same event.
    Each event can have a number of arguments between verbs and still be considered the same event.

    @param normalized_tags: result of normalized_sent_tags - a list of SRL tags for each word
    @param verb_tags: list of SRL tags for the algorithm to classify as event tags
    @param event_threshold: Threshold of non-verb arguments that can be found between verbs
    and still be considered part of the same event. n_args = event_threshold - 1
    @return: Boolean list of whether or not the word is part of an event
    """
    event_tags = []
    event_continue, event_begin = False, False
    for i, tag in zip_longest(np.arange(len(normalized_tags) - event_threshold), normalized_tags):
        if i is not None:
            if ("ARGM" in tag) & (normalized_tags[i + 1] in verb_tags):
                event_tags.append(True)
                event_begin = True
                continue

        if event_continue:
            event_tags.append(True)
        elif tag in verb_tags:
            event_tags.append(True)
            event_begin = True
        else:
            event_tags.append(False)

        if i is not None:
            conds = []
            for j in np.arange(1, event_threshold + 1):
                conds.append(normalized_tags[i + j] in verb_tags)

            if event_begin & any(conds):
                event_continue = True
            else:
                event_continue = False
                event_begin = False
        else:
            event_continue = False
            event_begin = False

    return event_tags

def _find_participants(begin_tags, event_tags):
    """
    Finds words that belong to the same participant or event and categorize them into participants.
    Each different argument represents a different participant. Arguments start at the begin tag (B).

    @param begin_tags: result of normalized_sent_tags - boolean list of tags that begin SRL arguments
    @param event_tags: result of find_events - boolean list of words that represent events
    @return: List of the participants that are represented by each word in the sentence
    """
    participant, i, participant_tags, event, is_event = False, 0, [], 1, False
    for btag, etag in zip(begin_tags, event_tags):
        if etag:
            is_event = True
        elif (not etag) & is_event:
            is_event = False
            event += 1

        if btag & (not etag):
            participant = True
            i += 1
            participant_tags.append("T" + str(i))
            continue

        if etag:
            participant_tags.append("EVENT" + str(event))
            participant = False
            continue

        if (not btag) & participant:
            participant_tags.append("T" + str(i))
        else:
            if len(participant_tags) > 0:
                participant_tags.append(participant_tags[-1])  # In case everything else fails, just join with previous participant
            else:
                # TODO: when participant_tags is empty what to do?
                i += 1
                participant_tags.append("T" + str(i))

    return participant_tags

def _srl_by_participant(srl_by_token, text, char_offset):
    """
    Organizes participants as words or expressions in the full text with their respective semantic role and character span.

    The semantic role is "EVENT" for event arguments and the SRL result for other participants.
    If the SRL result is a modifier, the most common in the participant is taken into account.

    The character span is a tuple - (start_char, end_char) -
    where the "start_char" is where the participant starts in the text and the "end_char" is where the participant ends.

    @param srl_by_token: DataFrame containing the results for the rest of the pipeline - namely, the participant references
    @param text: The full text to be annotated
    @param char_offset: The current char position in the text. Each word in the text increments it
    @return: Dict list with the participant, its semantic role and the its character position span in the text

    @note: CAN MALFUNCTION IF SRL DOES NOT FIND FRAMES IN EVERY SENTENCE.
    The char positions are checked in the entire text, if there is a word before in a sentence not recognized by the SRL
    it will malfunction slightly (give wrong position values).
    """
    result_list = []
    ARGM_STR = pipeline["ARGM_STR"]
    for participant in srl_by_token["participant"].unique():
        rows = srl_by_token[srl_by_token["participant"] == participant]
        tags = rows["tag"]
        if participant.startswith("EVENT"):
            sem_role_type = "EVENT"
        elif any(ARGM_STR in tag for tag in tags):
            argm_tags = [tag for tag in tags if ARGM_STR in tag]
            #print("-->",argm_tags[0].split("-")[-1])
            sem_role_type = SRL_TYPE_MAPPING[argm_tags[0].split("-")[-1]]
        else:
            sem_role_type = "PARTICIPANT"

        char_spans = []
        for token in rows.index:
            char_offset = text.find(token, char_offset)
            char_spans.append(char_offset)
            char_offset += len(token)

        result_list.append({
            "participant": ' '.join(rows.index), "sem_role_type": sem_role_type,
            "char_span": (char_spans[0], char_spans[-1] + len(token))
        })

    return result_list, char_offset

def _srl_pipeline(sentence_df, text, char_offset, verb_tags, event_threshold=3):
    """
    Pipeline to retrieve participants and events by their order in the text, with their semantic roles and character spans.
    Each iteration of the pipeline takes a dataframe of the SRL results for a sentence -> as returned by _make_srl_df.

    @param sentence_df: The DataFrame with the SRL results with the structure shown in _make_srl_df
    @param text: The full text to be annotated
    @param char_offset: The current character position offset to start looking for character spans (recursive)
    @param verb_tags: SRL tags that represent verbs -> defaults to ["B-V", "I-V"]
    @param event_threshold: Threshold of non-verb arguments that can be found between verbs
    and still be considered part of the same event. number_of_args = event_threshold - 1

    @return: df_by_participant -> Pandas DataFrame with each participant, their semantic roles and character spans
    character_offset -> The current character position offset in the full text
    """
    # 2. Remove words out of vocabulary in every frame #
    oov_mask = []
    for name, values in sentence_df.items():
        oov_mask.append(any(~(values == "O")))
    sentence_df = sentence_df.loc[:, oov_mask]
    sentence_df = sentence_df.apply(lambda x: x.sort_values(ascending=False).values)  # Begin tags in the end always

    # 3. Normalize SRL tags for each word in the sentence dataframe #
    normalized_tags, begin_tags = _normalize_sent_tags(sentence_df)

    # 4. Define events #
    event_tags = _find_events(normalized_tags, verb_tags=verb_tags, event_threshold=event_threshold)

    #print("---", event_tags)
    #print()
    # 5. Find and categorize expressions into participants #
    participant_tags = _find_participants(begin_tags, event_tags)

    # Putting together a dataframe for the characteristics of each word in this sentence
    if sentence_df.size == 0:
        return pd.DataFrame(), 0
    df = pd.DataFrame({
        "tag": normalized_tags, "is_begin_tag": begin_tags, "is_event": event_tags, "participant": participant_tags
    }, index=sentence_df.columns)

    # 6. Find semantic roles and character spans for each participant in the sentence #
    df_by_participant, character_offset = _srl_by_participant(df, text, char_offset)

    return df_by_participant, character_offset
def get_participant_tags(text, participant_lst, lang):
    """
    it adds POS tags and NER tags to the participant list

    @param text: the original text to be processed as narrative
    @param list[tuple[int,int]]: A list of character span of the participant
    candidates
    Returns
    ------
    list[tuple[tuple[int, int], str, str]]
       a list of participants of narratives, where the first is the character
       span of that participant, the second elements is the POS tag, anf third
       element is the NER tag, if it is a entity
    """
    participant_lst = sorted(participant_lst, key=lambda x:x[0])
    tagged_participants = {}

    doc = pipeline[lang](text)
    tagged_participants_lst = []
    for tok in doc:
        pos = bsearch_tuplelist(tok.idx, participant_lst)

        if pos != -1:

            # only update the tag value with the lexical head of the
            # participant

            if pos not in tagged_participants:
                offset_participant = participant_lst[pos]
                if tok.ent_type_ == '':
                    tagged_participants[pos] = (offset_participant, tok.pos_, 'Arg')
                else:
                    tagged_participants[pos] = (offset_participant, tok.pos_,tok.ent_type_.lower().capitalize())
            else:
                # if there is a token inside the text span that  has a NER tag, then replace the ner tag with more specific than UNDEF
                offset_participant, pos_tag, ner_tag = tagged_participants[pos]
                if ner_tag == 'Arg' and tok.ent_type_ != '':
                    tagged_participants[pos] = (offset_participant, tok.pos_, tok.ent_type_.lower().capitalize())
        else:

            tagged_participants_lst.append(((tok.idx, tok.idx + len(tok.text)),tok.pos_,"UNDEF"))

    # it is possible that some participant was not found by the bsearch?
    for idx in range(len(participant_lst)):
        tagged_participants_lst.append(tagged_participants[idx])

    tagged_participants_lst.sort(key=lambda x:x[0][0])

    return tagged_participants_lst

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
    # 1. DATAFRAME WITH THE SRL RESULTS OF EVERY FRAME FOR EACH SENTENCE IN THE TEXT #
    dfs_by_sent = _make_srl_df(lang, text)

    # FIND EVENTS - PIPELINE #
    character_offset, srl_participants_list = 0, []
    for sent_df in dfs_by_sent:
        df_by_participant, character_offset = _srl_pipeline(sent_df, text, character_offset, verb_tags=["B-V", "I-V"],
                                                      event_threshold=2)

        srl_participants_list.append(df_by_participant)


    srl_participants_list = [item for sublist in srl_participants_list for item in sublist]  # flatten list
    srl_df = pd.DataFrame(srl_participants_list)
    if len(srl_df) == 0:
        return []

    participants_df = srl_df[srl_df["sem_role_type"] == "PARTICIPANT"]
    participants_lst = participants_df["char_span"].values.tolist()

    # TODO: these participants still dont have POS and NER tags
    participants_tagged_lst = get_participant_tags(text, participants_lst, lang)
    #participants_lst = [ (p, "NOUN", "Per") for p in participants_lst]

    return participants_tagged_lst

def extract_events(lang, text):
    """
    Main function that applies the SRL pipeline to extract event entities from each sentence.
    Joins every event participant from each sentence in the text.

    @param lang: The language of the text
    @param text: The full text to be annotated

    @return: Pandas DataFrame with every event entity and their character span
    """
    # 1. DATAFRAME WITH THE SRL RESULTS OF EVERY FRAME FOR EACH SENTENCE IN THE TEXT #
    dfs_by_sent = _make_srl_df(lang, text)

    # FIND EVENTS - PIPELINE #
    character_offset, srl_participants_list = 0, []
    for sent_df in dfs_by_sent:
        df_by_participant, character_offset = _srl_pipeline(sent_df, text, character_offset, verb_tags=["B-V", "I-V"],
                                                      event_threshold=3)
        srl_participants_list.append(df_by_participant)

    srl_participants_list = [item for sublist in srl_participants_list for item in sublist]  # flatten list
    srl_df = pd.DataFrame(srl_participants_list)
    if len(srl_df) == 0:
        return []
    return srl_df[srl_df["sem_role_type"] == "EVENT"]

def extract_semantic_role_links(lang, text):
    dfs_by_sent = _make_srl_df(lang, text)

    character_offset, srl_by_sentence = 0, []
    for sent_df in dfs_by_sent:
        df_by_participant, character_offset = _srl_pipeline(sent_df, text, character_offset, verb_tags=["B-V", "I-V"],
                                                      event_threshold=3)
        srl_by_sentence.append(pd.DataFrame(df_by_participant))
    return srl_by_sentence

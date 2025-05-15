from transformers import Pipeline, AutoModel, AutoTokenizer
import spacy
import spacy_transformers
from torch import Tensor
import torch
from text2story.annotators.SRL.evaluator import Evaluator
import logging
from string import punctuation, whitespace

logger = logging.getLogger(__name__)


# Study wether subclassing is worth it due to the needed changes in postprocessing and sanitize parameters
class SrlPipeline(Pipeline):
    def __init__(self, model: str, tokenizer: str, framework, task, **kwargs):
        super().__init__(model, tokenizer=tokenizer)
        if "lang" in kwargs and kwargs["lang"] == "en":
            logger.info("Loading English verb predictor model...")
            if not (spacy.util.is_package('en_core_web_lg')):
                spacy.cli.download('en_core_web_lg')
            self.verb_predictor = spacy.load("en_core_web_lg")
        else:
            logger.info("Loading Portuguese verb predictor model...")
            if not (spacy.util.is_package('pt_core_news_lg')):
                spacy.cli.download('pt_core_news_lg')
            self.verb_predictor = spacy.load("pt_core_news_lg")
        logger.info("Got verb prediction model\n")

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, sentence: str):
        # Extract sentence verbs
        doc = self.verb_predictor(sentence)
        #print([token.pos_ for token in doc])
        verbs = {token.text for token in doc if token.pos_ == "VERB"}
        # If the sentence only contains auxiliary verbs, consider those as the main verbs
        if not verbs:
            verbs = {token.text for token in doc if token.pos_ == "AUX"}
        #print(verbs)

        # Tokenize sentence
        tokens = self.tokenizer.encode_plus(
            sentence,
            truncation=True,
            return_token_type_ids=False,
            return_offsets_mapping=True,
        )
        tokens_lst = tokens.tokens()
        offsets = tokens["offset_mapping"]

        input_ids = torch.tensor([tokens["input_ids"]], dtype=torch.long)
        attention_mask = torch.tensor([tokens["attention_mask"]], dtype=torch.long)
        # token_type_ids = torch.tensor([token_type_id], dtype=torch.long)

        model_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": [],
            "tokens": tokens_lst,
            "verb": "",
        }

        model_inputs = [
            {**model_input} for _ in verbs
        ]  # Create a new dictionary for each verb

        for i, verb in enumerate(verbs):
            model_inputs[i]["verb"] = verb
            token_type_ids = model_inputs[i]["token_type_ids"]
            token_type_ids.append([])
            curr_word_offsets: tuple[int, int] = None

            for j in range(len(tokens_lst)):
                curr_offsets = offsets[j]
                curr_slice = sentence[curr_offsets[0] : curr_offsets[1]]
                if not curr_slice:
                    token_type_ids[-1].append(0)
                # Check if new token still belongs to same word
                elif (
                    curr_word_offsets
                    and curr_offsets[0] >= curr_word_offsets[0]
                    and curr_offsets[1] <= curr_word_offsets[1]
                ):
                    # Extend previous token type
                    token_type_ids[-1].append(token_type_ids[-1][-1])
                else:
                    curr_word_offsets = self._find_word(sentence, start=curr_offsets[0])
                    curr_word = sentence[curr_word_offsets[0] : curr_word_offsets[1]]

                    token_type_ids[-1].append(
                        int(curr_word != "" and curr_word == verb)
                    )

            model_inputs[i]["token_type_ids"] = torch.tensor(
                token_type_ids, dtype=torch.long
            )

        #print(sentence, model_inputs)
        return model_inputs

    def _forward(self, model_inputs):
        outputs = []
        for model_input in model_inputs:
            output = self.model(
                input_ids=model_input["input_ids"],
                attention_mask=model_input["attention_mask"],
                token_type_ids=model_input["token_type_ids"],
            )
            output["verb"] = model_input["verb"]
            output["tokens"] = model_input["tokens"]
            outputs.append(output)
        return outputs

    def postprocess(self, model_outputs):
        """
        Every list entry in the output is of the type {verb: (labels, List[(token, label)])}
        """
        result = []
        id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        evaluator = Evaluator(id2label)

        for model_output in model_outputs:
            class_probabilities = model_output["class_probabilities"]
            attention_mask = model_output["attention_mask"]
            output_dict = evaluator.make_output_human_readable(
                class_probabilities, attention_mask
            )
            # Here we always fetch the first list because in a pipeline every sentence is processed one at a time
            wordpiece_label_ids = output_dict["wordpiece_label_ids"][0]
            labels = list(map(lambda idx: id2label[idx], wordpiece_label_ids))
            result.append({model_output["verb"]: (labels, list(zip(model_output["tokens"], labels)))})
        return result

    def _find_word(self, s, start=0):
        for i, char in enumerate(s[start:], start):
            if not char.isalpha():
                return start, i
        return start, len(s)

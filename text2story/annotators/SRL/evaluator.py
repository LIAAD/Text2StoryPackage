from typing import Optional, List, Dict
import math
import torch
import evaluate
import logging
from transformers import EvalPrediction


class Evaluator:
    def __init__(self, index2label: Dict[int, str]):
        self.index2label = index2label

    def get_lengths_from_binary_sequence_mask(
        self, mask: torch.BoolTensor
    ) -> torch.LongTensor:
        """
        Compute sequence lengths for each batch element in a tensor using a
        binary mask.

        # Parameters

        mask : `torch.BoolTensor`, required.
            A 2D binary mask of shape (batch_size, sequence_length) to
            calculate the per-batch sequence lengths from.

        # Returns

        `torch.LongTensor`
            A torch.LongTensor of shape (batch_size,) representing the lengths
            of the sequences in the batch.
        """
        # Removes -100 entries which correspond to HF added padding
        return (mask >= 0).sum(-1)

    def viterbi_decode(
        self,
        tag_sequence: torch.Tensor,
        transition_matrix: torch.Tensor,
        tag_observations: Optional[List[int]] = None,
        allowed_start_transitions: torch.Tensor = None,
        allowed_end_transitions: torch.Tensor = None,
        top_k: int = None,
    ):
        """
        Perform Viterbi decoding in log space over a sequence given a transition matrix
        specifying pairwise (transition) potentials between tags and a matrix of shape
        (sequence_length, num_tags) specifying unary potentials for possible tags per
        timestep.

        # Parameters

        tag_sequence : torch.Tensor, required.
            A tensor of shape (sequence_length, num_tags) representing scores for
            a set of tags over a given sequence.
        transition_matrix : torch.Tensor, required.
            A tensor of shape (num_tags, num_tags) representing the binary potentials
            for transitioning between a given pair of tags.
        tag_observations : Optional[List[int]], optional, (default = None)
            A list of length `sequence_length` containing the class ids of observed
            elements in the sequence, with unobserved elements being set to -1. Note that
            it is possible to provide evidence which results in degenerate labelings if
            the sequences of tags you provide as evidence cannot transition between each
            other, or those transitions are extremely unlikely. In this situation we log a
            warning, but the responsibility for providing self-consistent evidence ultimately
            lies with the user.
        allowed_start_transitions : torch.Tensor, optional, (default = None)
            An optional tensor of shape (num_tags,) describing which tags the START token
            may transition *to*. If provided, additional transition constraints will be used for
            determining the start element of the sequence.
        allowed_end_transitions : torch.Tensor, optional, (default = None)
            An optional tensor of shape (num_tags,) describing which tags may transition *to* the
            end tag. If provided, additional transition constraints will be used for determining
            the end element of the sequence.
        top_k : int, optional, (default = None)
            Optional integer specifying how many of the top paths to return. For top_k>=1, returns
            a tuple of two lists: top_k_paths, top_k_scores, For top_k==None, returns a flattened
            tuple with just the top path and its score (not in lists, for backwards compatibility).

        # Returns

        viterbi_path : `List[int]`
            The tag indices of the maximum likelihood tag sequence.
        viterbi_score : `torch.Tensor`
            The score of the viterbi path.
        """
        if top_k is None:
            top_k = 1
            flatten_output = True
        elif top_k >= 1:
            flatten_output = False
        else:
            raise ValueError(
                f"top_k must be either None or an integer >=1. Instead received {top_k}"
            )

        sequence_length, num_tags = list(tag_sequence.size())

        has_start_end_restrictions = (
            allowed_end_transitions is not None or allowed_start_transitions is not None
        )

        if has_start_end_restrictions:

            if allowed_end_transitions is None:
                allowed_end_transitions = torch.zeros(num_tags)
            if allowed_start_transitions is None:
                allowed_start_transitions = torch.zeros(num_tags)

            num_tags = num_tags + 2
            new_transition_matrix = torch.zeros(num_tags, num_tags)
            new_transition_matrix[:-2, :-2] = transition_matrix

            # Start and end transitions are fully defined, but cannot transition between each other.

            allowed_start_transitions = torch.cat(
                [allowed_start_transitions, torch.tensor([-math.inf, -math.inf])]
            )
            allowed_end_transitions = torch.cat(
                [allowed_end_transitions, torch.tensor([-math.inf, -math.inf])]
            )

            # First define how we may transition FROM the start and end tags.
            new_transition_matrix[-2, :] = allowed_start_transitions
            # We cannot transition from the end tag to any tag.
            new_transition_matrix[-1, :] = -math.inf

            new_transition_matrix[:, -1] = allowed_end_transitions
            # We cannot transition to the start tag from any tag.
            new_transition_matrix[:, -2] = -math.inf

            transition_matrix = new_transition_matrix

        if tag_observations:
            if len(tag_observations) != sequence_length:
                raise Exception(
                    f"Observations were provided, but they were not the same length as the sequence. Found sequence of length: {sequence_length} and evidence: {tag_observations}"
                )
        else:
            tag_observations = [-1 for _ in range(sequence_length)]

        if has_start_end_restrictions:
            tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
            zero_sentinel = torch.zeros(1, num_tags)
            extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
            tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
            tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
            sequence_length = tag_sequence.size(0)

        path_scores = []
        path_indices = []

        if tag_observations[0] != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[tag_observations[0]] = 100000.0
            path_scores.append(one_hot.unsqueeze(0))
        else:
            path_scores.append(tag_sequence[0, :].unsqueeze(0))

        # Evaluate the scores for all possible paths.
        for timestep in range(1, sequence_length):
            # Add pairwise potentials to current scores.
            summed_potentials = (
                path_scores[timestep - 1].unsqueeze(2) + transition_matrix
            )
            summed_potentials = summed_potentials.view(-1, num_tags)

            # Best pairwise potential path score from the previous timestep.
            max_k = min(summed_potentials.size()[0], top_k)
            scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)

            # If we have an observation for this timestep, use it
            # instead of the distribution over tags.
            observation = tag_observations[timestep]
            # Warn the user if they have passed
            # invalid/extremely unlikely evidence.
            if tag_observations[timestep - 1] != -1 and observation != -1:
                if (
                    transition_matrix[tag_observations[timestep - 1], observation]
                    < -10000
                ):
                    logging.info(
                        "The pairwise potential between tags you have passed as "
                        "observations is extremely unlikely. Double check your evidence "
                        "or transition potentials!"
                    )
            if observation != -1:
                one_hot = torch.zeros(num_tags)
                one_hot[observation] = 100000.0
                path_scores.append(one_hot.unsqueeze(0))
            else:
                path_scores.append(tag_sequence[timestep, :] + scores)
            path_indices.append(paths.squeeze())

        # Construct the most likely sequence backwards.
        path_scores_v = path_scores[-1].view(-1)
        max_k = min(path_scores_v.size()[0], top_k)
        viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k, dim=0)
        viterbi_paths = []
        for i in range(max_k):
            viterbi_path = [best_paths[i]]
            for backward_timestep in reversed(path_indices):
                viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
            # Reverse the backward path.
            viterbi_path.reverse()

            if has_start_end_restrictions:
                viterbi_path = viterbi_path[1:-1]

            # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
            viterbi_path = [j % num_tags for j in viterbi_path]
            viterbi_paths.append(viterbi_path)

        if flatten_output:
            return viterbi_paths[0], viterbi_scores[0]

        return viterbi_paths, viterbi_scores

    def make_output_human_readable(
        self, class_probabilities, attention_mask
    ) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        `"tags"` key to the dictionary with the result.

        NOTE: First, we decode a BIO sequence on top of the wordpieces. This is important; viterbi
        decoding produces low quality output if you decode on top of word representations directly,
        because the model gets confused by the 'missing' positions (which is sensible as it is trained
        to perform tagging on wordpieces, not words).

        Secondly, it's important that the indices we use to recover words from the wordpieces are the
        start_offsets (i.e offsets which correspond to using the first wordpiece of words which are
        tokenized into multiple wordpieces) as otherwise, we might get an ill-formed BIO sequence
        when we select out the word tags from the wordpiece tags. This happens in the case that a word
        is split into multiple word pieces, and then we take the last tag of the word, which might
        correspond to, e.g, I-V, which would not be allowed as it is not preceeded by a B tag.
        """

        sequence_lengths = self.get_lengths_from_binary_sequence_mask(
            attention_mask
        ).data.tolist()

        # Multiple sequences
        if class_probabilities.dim() == 3:
            predictions_list = [
                class_probabilities[i].detach().cpu()
                for i in range(class_probabilities.size(0))
            ]
        else:
            predictions_list = [class_probabilities]

        wordpiece_tags = []
        wordpiece_label_ids = []
        # word_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        start_transitions = self.get_start_transitions()
        # **************** Different ********************
        # We add in the offsets here so we can compute the un-wordpieced tags.

        output_dict = {}

        for predictions, length in zip(
            predictions_list, sequence_lengths  # , output_dict["start_offsets"]
        ):

            max_likelihood_sequence, _ = self.viterbi_decode(
                predictions[:length],
                transition_matrix,
                allowed_start_transitions=start_transitions,
            )

            tags = [self.index2label[x] for x in max_likelihood_sequence]

            wordpiece_tags.append(tags)
            wordpiece_label_ids.append(max_likelihood_sequence)
            # word_tags.append([tags[i] for i in offsets])

        output_dict["wordpiece_tags"] = wordpiece_tags
        output_dict["wordpiece_label_ids"] = wordpiece_label_ids
        # output_dict["tags"] = word_tags
        return output_dict

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        # Returns

        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = (self.index2label).copy()

        # Workaround: During loss calculation and hf's padding process, -100 is used as the ignore index
        # However, here fo the concept of the transition matrix that would not be applicable, as it would create entries
        # in the -100th row and column. So I temporarily map it to 0 and remmap it back after the decoding
        # ignore_label = all_labels.pop(-100)
        # all_labels[0] = ignore_label

        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == "I" and not previous_label == "B" + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    def get_start_transitions(self):
        """
        In the BIO sequence, we cannot start the sequence with an I-XXX tag.
        This transition sequence is passed to viterbi_decode to specify this constraint.

        # Returns

        start_transitions : torch.Tensor
            The pairwise potentials between a START token and
            the first token of the sequence.
        """
        all_labels = self.index2label
        num_labels = len(all_labels)

        start_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I":
                start_transitions[i] = float("-inf")

        return start_transitions


def prepare_compute_metrics(evaluator: Evaluator):
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    id2label = evaluator.index2label

    labels = [
        k for k, v in evaluator.index2label.items() if v.split("-")[-1] != "V"
    ]  # Ignore Bverb labels as that will be provided to the model

    def compute_metrics(eval_prediction: EvalPrediction):
        class_probabilities = torch.tensor(
            eval_prediction.predictions[1]
        )  # Get class probabilities
        attention_mask = torch.tensor(eval_prediction.predictions[2])  # Get mask
        # input_ids = torch.tensor(eval_prediction.predictions[3])

        output_dict = evaluator.make_output_human_readable(
            class_probabilities, attention_mask
        )
        references = eval_prediction.label_ids

        overall_precision = 0
        overall_recall = 0
        overall_f1 = 0

        for i, curr_ref in enumerate(references):
            curr_ref = curr_ref[curr_ref != -100]  # Remove HF padding
            curr_ref = curr_ref.tolist()

            predictions = output_dict["wordpiece_label_ids"][i]

            overall_precision += precision.compute(
                references=curr_ref,
                predictions=predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )["precision"]

            overall_recall += recall.compute(
                references=curr_ref,
                predictions=predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )["recall"]

            overall_f1 += f1.compute(
                references=curr_ref,
                predictions=predictions,
                labels=labels,
                average="weighted",
            )["f1"]

            print(set(id2label[ref] for ref in curr_ref) - set(id2label[pred] for pred in predictions))

        return {
            "precision": overall_precision / len(references),
            "recall": overall_recall / len(references),
            "f1": overall_f1 / len(references),
        }

    return compute_metrics

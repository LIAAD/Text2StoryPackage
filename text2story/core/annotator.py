"""
	text2story.core.annotator

	META-annotator
"""

from text2story.annotators import PARTICIPANT_EXTRACTION_TOOLS, TIME_EXTRACTION_TOOLS, OBJECTAL_LINKS_RESOLUTION_TOOLS
from text2story.annotators import EVENT_EXTRACTION_TOOLS, SEMANTIC_ROLE_LABELLING_TOOLS
from text2story.annotators import extract_participants, extract_times, extract_objectal_links, extract_events
from text2story.annotators import extract_semantic_role_links


class Annotator:
    """
    Representation of a META-Annotator (a combination of one or more annotators).
    Useful to give a uniform interface to the rest of the package.

    Attributes
    ----------
    tools : list[str]
        the list of annotators to be used

    Methods
    -------
    extract_participants(lang, text)
        Returns a list with the participants identified in the text.
        Each actor is represented by a tuple, consisting of the start character offset, end character offset, the POS tag and the NE IOB tag, resp.
            Example: (0, 4, 'Noun', 'Per')
        Possible POS tags: 'Noun', 'Pronoun'.
        Possible NE IOB tags: 'Per', 'Org', 'Loc', 'Obj', 'Nat' and 'Other'.


    extract_times(lang, text)
        Returns a list with the times identified in the text.
        Each time is represented by a tuple, consisting of the start character offset, end character offset, it's type and it's value, resp.
            Example: (6, 17, 'DATE', '2021-08-31')

    extract_corefs(lang, text)
        Returns a list with the clusters of entities identified in the text
        Each cluster is a list with tuples, where every tuple is a 2D tuple with the start and end character offset of the span corresponding to the same entity.
            Example: [(0, 6), (20, 22)]
    """

    def __init__(self, tools):
        """
        Parameters
        ----------
        tools : list[str]
            the list of the annotators to be used; can be used any combination of them
            possible annotators are: 'spacy', 'nltk' and 'sparknlp'
        """
        self.tools = tools


    def extract_participants(self, lang, text, url=None):
        """
        Parameters
        ----------
        lang : str
            the language of the text
            current supported languages are: portuguese ('pt'); english ('en')
        text : str
            the text to be made the extraction

        Returns
        -------
        list[tuple[tuple[int, int], str, str]]
            the list of actors identified where each actor is represented by a tuple
        """

        # The extraction can be done with more than one tool.
        # Since different tools can make a different tokenization of the text, efforts were made to identify the same entity, even when character span doesn't match.
        # For instance, some tool identified the entity with character span (2, 7) and other (5, 10). We are assuming that the entirely entity has the char. span of (2, 10).
        # To do that, we are taking the first (in the sense of the char. span) identification made, and keep extending the character end offset as much as possible, with every entity that has a span that intersects with our current span.
        # Also, note that we are obtaning a bunch of POS tags and NE IOB tags and we just want one.
        # For the POS tag, we are taking the most common label.
        # For the NE IOB tag, we do the same, but we favor all labels versus the generic 'Other' label. That is, even if the label 'Other' is the most common, if we have a more specific one, we use that, instead.

        nr_tools = len(self.tools)

        # If no tool specified, use all
        if nr_tools == 0:
            self.tools = PARTICIPANT_EXTRACTION_TOOLS.keys()

            nr_tools = len(self.tools)

        # Gather the annotations made by the tools specified and combine the results
        annotations = []
        idxs = []
        for tool in self.tools:
            if url is not None:
                annotations.append(extract_participants(tool, lang, text, url))
            else:
                annotations.append(extract_participants(tool, lang, text))

        final_annotation = []

        idxs = [0] * nr_tools # Current actor position from each tool


        while not(all([len(annotations[i]) == idxs[i] for i in range(nr_tools)])): # We finish when we consumed every actor identified by every tool
            tool_id = -1
            tool_id_start_char_offset = len(text) # len(self.text) acting as infinite

            # Get the next entity chunk to be gather (the one with the lowest start character span)
            for i in range(nr_tools):
                if idxs[i] < len(annotations[i]):

                    current_actor = annotations[i][idxs[i]]
                    current_actor_start_character_span = current_actor[0][0]

                    if current_actor_start_character_span < tool_id_start_char_offset:
                        tool_id = i
                        tool_id_start_char_offset = current_actor_start_character_span


            # For now, our actor consists of a unique annotation made by some tool
            actor_start_character_offset = annotations[tool_id][idxs[tool_id]][0][0]
            actor_end_character_offset   = annotations[tool_id][idxs[tool_id]][0][1]
            # For the lexical head and type we will accumulate the results and latter choose the best following a criterion
            actor_lexical_heads          = [annotations[tool_id][idxs[tool_id]][1]]
            actor_types                  = [annotations[tool_id][idxs[tool_id]][2]]

            idxs[tool_id] += 1 # Consume the annotation

            # Other tools may have identified the same actor.
            # We need to search if theres some intersection in the span of the identified actor, with the other tools.
            # That is, they identified the same actor, but maybe missed some initial part of it.
            # We identify this situation by looking to the character start char offset of the current actor identified by each tool,
            # and if it happens to be less than our end char offset of our current identified actor, then we can extend the current information we have.
            # Note that we may extend the end char offset, each force us to repete this process, till the offsets stabilize.

            # In the first interation, the tool that first identified the actor, will be matched and add double information and we don't want that
            # If we get to a second iteration, then that means the actor end char offset was extended which, in that case, the tool that first identified the actor, may now extend also...
            first_iteration = True
            while True:
                flag = False
                for i in range(nr_tools):
                    if first_iteration and i == tool_id:
                        continue

                    if idxs[i] < len(annotations[i]):
                        if annotations[i][idxs[i]][0][0] <= actor_end_character_offset:
                            if actor_end_character_offset < annotations[i][idxs[i]][0][1]:
                                actor_end_character_offset = annotations[i][idxs[i]][0][1]
                                flag = True

                            actor_lexical_heads.append(annotations[i][idxs[i]][1])
                            actor_types.append(annotations[i][idxs[i]][2])
                            idxs[i] = idxs[i] + 1

                first_iteration = False
                if not(flag):
                    break

            # Now that we identified the larger span possible for the actor, we need to fix the lexical head and type.

            # For the POS tag, we favor specifics the 'Noun' and 'Pronoun' tag and then we take the most common.
            # Since we only defined this labels, the others will appear as 'UNDEF'
            rmv_undef_pos_tags = [ne for ne in actor_lexical_heads if ne != 'UNDEF']
            if rmv_undef_pos_tags:
                actor_lexical_head = max(rmv_undef_pos_tags, key=rmv_undef_pos_tags.count)
            else:
                continue # Discard the actor if it's lexical head isn't a 'Noun' or 'Pronoun'

            # For the NE, we also favor specifics NEs, in this case all labels versus the NE 'OTHER' and we take the most common.
            rmv_other_ne = [ne for ne in actor_types if ne != 'Other' or ne != 'Arg']
            if rmv_other_ne:
                actor_type = max(rmv_other_ne, key=rmv_other_ne.count)
            else:
                actor_type = 'Other'

            # Discard entities with types other than 'Per', 'Org', 'Loc', 'Obj', 'Nat' & 'Other'.
            # if it is Agr, it was an argument of an semantic role labeling event
            # Used, typically, to eliminate dates and durations incorrectly identified as an actor.
            if actor_type.lower() in ['arg','per', 'org', 'loc', 'obj', 'nat', 'other','person','gpe']:
                final_annotation.append(((actor_start_character_offset, actor_end_character_offset), actor_lexical_head, actor_type))

        return final_annotation

    def extract_times(self, lang, text, publication_time):
        """
        Parameters
        ----------
        lang : str
            the language of the text
            current supported languages are: portuguese ('pt'); english ('en')
        text : str
            the text to be made the extraction
        publication_time: str
            the publication time

        Returns
        -------
        list[tuple[tuple[int, int], str, str]]
            a list consisting of the times identified, where each time is represented by a tuple
            with the start and end character offset, it's value and type, respectively
        """

        nr_tools = len(self.tools)

        # If no tool specified, use all
        if nr_tools == 0:
            self.tools = list(TIME_EXTRACTION_TOOLS.keys())
            nr_tools = len(self.tools)

        # NOTE: The extraction is done with only one tool, so the result in just the extraction done by the tool
        times = extract_times(self.tools[0], lang, text, publication_time) # :: [(time_start_offset, time_end_offset, time_type, time_value)]
        return times

    def extract_events(self, lang, text):
        """
        Event extraction. Only has one tool so it only returns what the annotator finds.

        @param lang: The language of the text
        @param text: The text to be annotated

        @return: Pandas DataFrame with each event found in the text and their character spans
        """
        nr_tools = len(self.tools)

        if nr_tools == 0:
            self.tools = list(EVENT_EXTRACTION_TOOLS.keys())
            nr_tools = len(self.tools)

        events = extract_events(self.tools[0], lang, text)
        return events

    def extract_objectal_links(self, lang, text):
        """
        Parameters
        ----------
        lang : str
            The language of the text.
            Current supported languages are: english ('en')
        text : str
            The text to be made the extraction.

        Returns
        -------
        list[list[tuple[int, int]]
            A list with the clusters identified.
            Each cluster is a list with tuples, where every tuple is a 2D tuple with the start and end character offset of the span corresponding to the same entity.
        """

        nr_tools = len(self.tools)

        # If no tool specified, use all
        if nr_tools == 0:
            self.tools = list(OBJECTAL_LINKS_RESOLUTION_TOOLS.keys())
            nr_tools = len(self.tools)

        # NOTE: The extraction is done with only one tool, so the result in just the extraction done by the tool
        return extract_objectal_links(self.tools[0], lang, text)

    def extract_semantic_role_links(self, lang, text):
        """
        Semantic Role Link extraction. Only has one tool, so no tool merging is needed.

        @param lang: The language of the text
        @param text: The text to be annotated

        @return: List of pandas DataFrames with the actors, events and their semantic roles to be linked
        """
        nr_tools = len(self.tools)

        # If no tool specified, use all
        if nr_tools == 0:
            self.tools = list(SEMANTIC_ROLE_LABELLING_TOOLS.keys())
            nr_tools = len(self.tools)

        srl_by_sentence = extract_semantic_role_links(self.tools[0], lang, text)

        return srl_by_sentence

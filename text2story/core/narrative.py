"""
	text2story.core.narrative

	Narrative class
"""
import warnings

from text2story.core.annotator import Annotator
from text2story.core.entity_structures import *
from text2story.core.link_structures import *
from text2story.core.utils import pairwise, capfirst, join_tokens, find_first_non_space

import re

from text2story.core.utils import map_pos2head
class Narrative:
    """
    Representation of a narrative.

    Attributes
    ----------
    lang: str
            the language of the text; supported languages are portuguese ('pt') and english ('en')
    text: str
            the text itself
    publication_time : str
                    the publication time ('XXXX-XX-XX')
    participants: dict{str -> participant}
            the participants identified in the text.
            each key in the dict, of the form 'T' concatenated with some int, has an participant as a value.
    times: dict{str -> Time}
            the temporal expressions identified in the text.
            each key in the dict, of the form 'T' concatenated with some int, has an time as a value.
    obj_links: dict{str -> ObjectalLink}
            the corefs identified in the text
            each key in the dict, of the form 'R' concatenated with some int, has an coref as a value.

    Methods
    -------
    extract_participants(*tools)
            extracts all the participants in the text using the annotators defined in 'tools', updating self.participants
    extract_timexs(*tools)
            extracts all the timexs in the text using the annotators defined in 'tools', updating self.timexs
    extract_corefs(*tools)
            coreference resolution in the text using the tools 'tools', updating self.obj_rels
            typically, this call increases self.participants since news entities can be identified
    _get_participant_key(char_offset)
            returns the key of the participant with the corresponding character offset or None if such participant wasn't identified before
    _add_participant(char_offset)
            update self.participants by adding the new participant with character offset 'char_offset' and returns the key given to the new participant
    ISO_annotation(file_name)
            outputs ISO annotation in .ann format (txt)
    """

    def __init__(self, lang, text, publication_time):
        """
        Parameters
        ----------
        lang : str
                the language of the text
        text : str
                the text ifself
        publication_time : str
                the publication time ('XXXX-XX-XX')
        """

        self.lang = lang
        self.text = text
        self.publication_time = publication_time

        # Counter to generate a unique ID for every participant
        # TODO: Fix the counter, when repeting some extraction: The counter just keep going up.
        self._id = 1
        self._event_id = 1
        self._rel_id = 1

        self.participants = {}
        self.times = {}
        self.events = {}
        self.spatial_relations = {}

        self.links = {}

        self.sem_role_map = {"participant":"agent","event":"cause",\
                             "location":"location", "theme":"theme",\
                             "path":"path","manner":"manner","cause":"cause",\
                             "agent":"agent","goal":"goal"}
        self.sem_role_ignore = ["time"] # semantic links to ignore, there is no mapping
        #self.obj_links = {}
        #self.sem_links = {}

        #self.temp_links = {}
        #self.subordination_links = {}
        #self.qualitative_spatial_links = {}
        #self.movement_links = {}
        # self.measure_links = {}

    def get_relations_byargs(self, arg1, arg2):
        """
        Search for a relation entity whose arguments are arg1 and arg2
        @param arg1:
        @param arg2:
        @return: [link_structures], a set of relations (ids) between the two entities
        """
        relation_lst = []
        for relation_type in self.links:
            for relation in self.links[relation_type]:
                if relation.arg1 == arg1 and relation.arg2 == arg2:
                    relation_lst.append(relation)

                if relation.arg1 == arg2 and relation.arg2 == arg1:
                    relation_lst.append(relation)

        return relation_lst


    def get_entity_byname(self, name):
        return self.__getattribute__(name)

    @classmethod
    def fromTokenCorpus(cls, lang,  doc, raw_text=None):
        """
        build a narrative object from a list of TokenCorpus objects

        @return: a narrative object

        """

        cls.lang = lang
        cls.publication_time = None

        cls.participants = {}
        cls.times = {}
        cls.events = {}
        cls.spatial_relations = {}
        cls.links = {}

        # first collect only the annotations
        n = len(doc)
        i = 0
        map_id_tok = {}
        rel_set = set()
        doc_txt = ""
        while i < n:

            for idx, id_ann in enumerate(doc[i].id_ann):
                if id_ann in map_id_tok:
                    map_id_tok[id_ann] = (map_id_tok[id_ann][0] + [i],\
                                          map_id_tok[id_ann][1],\
                                          map_id_tok[id_ann][2])
                else:
                    ann_type, ann_map = doc[i].attr[idx]
                    map_id_tok[id_ann] = ([i], ann_type, ann_map)

            # collect relations
            for rel in doc[i].relations:
                if rel.rel_id not in rel_set:

                    rel_type = rel.rel_type.split("_")

                    main_type = rel_type[0]
                    subtype = rel_type[1]
                    if rel.argn == "arg2":
                        # this could produce an error, if the token has more than one id
                        # TODO: make a function that takes the id of arg1 for the given relation
                        # maybe I should chek in rel.toks[0].relations with the same id , and get the argument
                        arg1 = doc[i].id_ann[0]
                        arg2 = rel.toks[0].id_ann[0]
                    else:
                        # this could produce an error, if the token has more than one id
                        arg2 = doc[i].id_ann[0]
                        arg1 = rel.toks[0].id_ann[0]

                    linkObj = createLinkObject(arg1,arg2,main_type, subtype, rel.rel_id)
                    if main_type in cls.links:
                        cls.links[main_type].append(linkObj)
                    else:
                        cls.links[main_type] = [linkObj]

                    rel_set.add(rel.rel_id)

            doc_txt = doc_txt + doc[i].text + " "
            i += 1

        for id_ann in map_id_tok:
            # the tokens of this annotation
            idx_lst, ann_type, ann_map = map_id_tok[id_ann]
            text = [doc[i].text for i in idx_lst]
            text = join_tokens(text)

            span_start = doc[idx_lst[0]].offset
            span_end = doc[idx_lst[-1]].offset + len(doc[idx_lst[-1]].text)

            if ann_type == "Participant":
                # idx_lst[0].pos is the first pos tag option to label the lexical head of the participant
                lexical_head = ann_map.get("Lexical_Head", map_pos2head(doc[idx_lst[0]].pos))

                participant_type = ann_map.get("Participant_Type_Domain","")
                cls.participants[id_ann] = ParticipantEntity(text, (span_start, span_end), lexical_head, participant_type=participant_type)
            elif ann_type == "Event":
                event_class = ann_map.get("Class", "")
                event_polarity = ann_map.get("Polarity","Pos")
                event_type = ann_map.get("Event_Type","Occurrence")
                event_pos = ann_map.get("Pos","Verb")
                event_tense = ann_map.get("Tense","Pres")
                event_aspect = ann_map.get("Aspect","Perfective")

                cls.events[id_ann] = EventEntity(text, (span_start, span_end),event_class=event_class,\
                                                 polarity=event_polarity,event_type=event_type,pos=event_pos,\
                                                 tense=event_tense,aspect=event_aspect)
            elif ann_type == "Time":
                # annotations :: [(TimeStartOffset, TimeEndOffset, TimeType, TimeValue)]
                time_type = ann_map.get("Time_Type","Time")
                temporal_function = ann_map.get("TemporalFunction","Publication_Time")
                cls.times[id_ann] = TimeEntity(text,(span_start,span_end),
                                                                 time_type, temporal_function=temporal_function)
            elif ann_type == "Spatial_Relation":
                spatial_relation_type = ann_map.get("Type","Topological")
                topological = ann_map.get("Topological","Equal")
                pathDefining = ann_map.get("PathDefining","Starts")
                cls.spatial_relations[id_ann] = SpatialRelationEntity(text, (span_start, span_end),\
                                                                      type=spatial_relation_type,\
                                                                      topological=topological,
                                                                      pathDefining=pathDefining)

            else:
                warnings.warn("Annotation type not recognized: %s -> %s" % (id_ann, ann_type))



        if raw_text is None:
            narrativeSelf = cls(lang,doc_txt,"2023")
        else:
            narrativeSelf = cls(lang, raw_text, "2023")
        narrativeSelf.participants = cls.participants
        narrativeSelf.events = cls.events
        narrativeSelf.times = cls.times
        narrativeSelf.spatial_relations = cls.spatial_relations
        narrativeSelf.links = cls.links
        return narrativeSelf

    def extract_participants(self, *tools, url=None):
        """
        Parameters
        ----------
        tools : str, ...
                the tools to be used in the annotation

        Returns
        -------
                self.participants updated
        """

        participants = Annotator(tools).extract_participants(self.lang,
                                                 self.text,url)  # annotations :: [(EntityStartOffset, EntityEndOffset, EntityPOSTag, EntityType)]

        for participant in participants:
            self.participants['T' + str(self._id)] = ParticipantEntity(self.text[participant[0][0]:participant[0][1]], participant[0], participant[1],
                                                           participant[2])
            self._id += 1

        return self.participants

    def extract_times(self, *tools):
        """
        Parameters
        ----------
        tools : str, ...
                the tools to be used in the annotation

        Returns
        -------
                self.times updated
        """
        times = Annotator(tools).extract_times(
            self.lang, self.text, self.publication_time)  # annotations :: [(TimeStartOffset, TimeEndOffset, TimeType, TimeValue)]

        for time in times:
            self.times['T' + str(self._id)] = TimeEntity(self.text[time[0][0]:time[0][1]], time[0], time[1],
                                                         time[2])
            self._id += 1

        return self.times

    def extract_events(self, *tools):
        """
        Event extraction function to combine different tools of event extraction.
        Currently there is only one tool (AllenNLP) so it just uses that one.

        @param tools: Iterable with the tools to use

        @return: Returns a list of events extracted from the text in the form of EventEntity objects
        """
        events = Annotator(tools).extract_events(self.lang, self.text)
        if len(events) == 0:
            self.events = []
            return []

        for event in events:
            start_event = event[0][0]
            end_event = event[0][1]
            event_text = self.text[start_event:end_event]
            self.events["T" + str(self._id)
                        ] = EventEntity(event_text, (start_event, end_event), **event[1])
            self._id += 1

        return self.events

    def extract_objectal_links(self, *tools):
        """
        Parameters
        ----------
        tools : str, ...
                the tools to be used in the annotation

        Returns
        -------
                self.obj_rels updated
        """

        # annotations ::
        clusters = Annotator(tools).extract_objectal_links(
            self.lang, self.text)
        obj_links = []

        for cluster in clusters:
            for i in range(0, len(cluster) - 1):
                e1 = cluster[i]
                e2 = cluster[i + 1]

                # Get the participants
                arg1, arg2 = self._get_participant_key(e1), self._get_participant_key(e2)

                # If one of them wasn't identified in the participant extration, add it as a new participant
                if arg1 == None:
                    arg1 = self._add_participant(e1)
                if arg2 == None:
                    arg2 = self._add_participant(e2)

                # (Type (sameHead, partOf, ...), Arg1, Arg2)
                subtype = type='objIdentity'
                obj_links.append(ObjectalLink(arg1, arg2, subtype, "R" + str(self._rel_id)))
                self._rel_id += 1

        self.links["OLINK"] = obj_links
        return obj_links

    def extract_semantic_role_links(self, *tools):
        """
        Find semantic role links between extracted participants and events.
        Since the SRL model is different from the NER model, this function maps participants found by the SRL model into
        participants that were already extracted. If the participant was not yet extracted, it adds a new one.

        Links participants to events by text order. If we have participant1 -> EVENT1 -> participant2 in the text,
        then we make two semantic role links - EVENT1 -> ROLE -> participant1 | EVENT1 -> ROLE -> participant2

        @param tools: Iterable of tools to be used

        @return: A dict with the SRL entities by key -> R10: SemanticRoleLink<10>
        """
        srl_by_sentence = Annotator(
            tools).extract_semantic_role_links(self.lang, self.text)

        # FIND OUT IF ARGUMENT OF SRL HAS AN participant RETRIEVED BY THE NER COMPONENT
        # IF NOT, ADD A NEW participant CORRESPONDING TO THE ARGUMENT
        for sentence_df in srl_by_sentence:

            key_list = []
            for row in sentence_df.itertuples():
                event_key = None
                if row.sem_role_type == "EVENT":
                    event_key = self._get_event_key(row.char_span)
                    if event_key is not None:
                        key_list.append(event_key)
                    else:
                        event_key = self._get_event_key(
                            row.char_span, match_type="partial")
                        if event_key is not None:
                            key_list.append(event_key)
                        else:
                            event_key = self._add_event(row.char_span)
                            key_list.append(event_key)
                    continue
                participant_key = self._get_participant_key(
                    row.char_span, match_type="partial")

                if participant_key is not None:
                    key_list.append(participant_key)
                else:
                    if participant_key is None and event_key is None:
                        key_list.append("None")
                #else:
                #    participant_key = self._add_participant(
                #        row.char_span, lexical_head="Noun", participant_type="Other")
                #    key_list.append(participant_key)
            
            sentence_df["key"] = key_list

        # MAKE SEMANTIC ROLE LINK ENTITIES
        sem_links = []
        for sentence_df in srl_by_sentence:
            for row1, row2 in pairwise(sentence_df.itertuples()):
                sem1 = row1.sem_role_type
                sem2 = row2.sem_role_type

                if (sem1 == "EVENT") | (sem2 == "EVENT"):
                    if sem1 != "EVENT":
                        sem_role, participant, event = sem1, row1.key, row2.key
                    else:
                        sem_role, participant, event = sem2, row2.key, row1.key

                    if participant == "None" or event == "None":
                        continue
                    if sem_role not in self.sem_role_ignore:
                        sem_links.append(SemanticRoleLink(
                            participant, event, self.sem_role_map[sem_role.lower()], "R" + str(self._rel_id)))
                        self._rel_id += 1

        self.links["SRLINK"] =sem_links
        return sem_links

    def _add_srlink(self, participant, event, sem_role):
        self.links["SRLINKS"]["R" + str(self._rel_id)
                       ] = SemanticRoleLink(participant, event, self.sem_role_map[sem_role.lower()], "R" + str(self._rel_id))
        self._rel_id += 1

    def _get_participant_key(self, char_span, match_type="exact"):
        """
        Parameters
        ----------
        char_span : (int, int)
                the participant character offset of the participant to find the key
        match_type: str
                type of match for the character span.
                If exact, the span must be exactly the same to match an participant.
                If partial, the span must be partially contained in the participant span to match it.

        Returns
        -------
                the key of the participant with the corresponding character offset
                or None if it doesn't exist
        """
        if match_type == "exact":
            for key in self.participants.keys():
                if self.participants[key].character_span == char_span:
                    return str(key)
        elif match_type == "partial":
            for key in self.participants.keys():
                aSpan = self.participants[key].character_span
                if aSpan[0] <= char_span[0] <= aSpan[1]:
                    return str(key)
                elif aSpan[0] <= char_span[1] <= aSpan[1]:
                    return str(key)
        else:
            raise ValueError(
                f"Parameter match_type must be one of [exact, partial].\nInstead it was {match_type}")

        return None

    def _add_participant(self, char_span, lexical_head="Pronoun", participant_type="Other"):
        """
        Parameters
        ----------
        char_span : (int, int)
                the participant character offset
        lexical_head: str
                The lexical head of the participant: "Noun" or "Pronoun" -> Defaults to "Pronoun"
        participant_type: str
                The type of the participant as expressed by the NER models -> Defaults to "Other"

        Returns
        -------
                the key of the new added participant
        """
        key = 'T' + str(self._id)
        self.participants[key] = ParticipantEntity(self.text[char_span[0]:char_span[1]], char_span, lexical_head,
                                       participant_type)  # Hard-coded lexical head and type as 'Pronoun' and 'Other', resp., for now

        self._id += 1

        return key

    def _get_event_key(self, char_span, match_type="exact"):
        """
        Get the key of an event entity based on its character span on the full document text
        nota: Esta função está redundante com a _get_participant_key. Só muda a lista em que se procura

        @param char_span: The character span of the event in the text
        @return: The key of the event if it is found. None otherwise
        """

        if match_type == "exact":
            for key in self.events.keys():
                if self.events[key].character_span == char_span:
                    return str(key)
        elif match_type == "partial":
            for key in self.events.keys():
                aSpan = self.events[key].character_span
                if aSpan[0] <= char_span[0] <= aSpan[1]:
                    return str(key)
                elif aSpan[0] <= char_span[1] <= aSpan[1]:
                    return str(key)
        else:
            raise ValueError(
                f"Parameter match_type must be one of [exact, partial].\nInstead it was {match_type}")

        return None

    def _add_event(self, char_span):
        """
        Adds a new event to the narrative.

        @param char_span: tuple of characters (first_char, last_char) that delimit the event

        @return: The key of the new added event
        """
        key = 'T' + str(self._id)
        self.events[key] = EventEntity(
            self.text[char_span[0]:char_span[1]], char_span,  event_class="Occurrence",\
                                        polarity="Pos", factuality="Factual",tense="Pres")

        self._id += 1

        return key

    def _add_time(self, char_span):
        """
        Adds a new time expression to the narrative.

        @param char_span: tuple of characters (first_char, last_char) that delimit the time expression

        @return: The key of the new added time expression
        """
        key = 'T' + str(self._id)

        # TODO: change default values for value and times_type parameters
        self.times[key] = TimeEntity(
            self.text[char_span[0]:char_span[1]], char_span, "","Time")

        self._id += 1

        return key

    def _get_span_segments(self, el):
        """
        If a element of the narrative is composed of various segments (multiline annotations), then we have
        to collect each segment. Then it returns the character offsets of the segments and the segment text
        @param el:
        @return:
        """

        el_text = self.text[el.character_span[0]:el.character_span[1]]
        result = [_.start() for _ in re.finditer("\n", el_text)]

        span0 = el.character_span[0]
        span1 = el.character_span[0] + result[0]
        char_span_text = f"{span0} {span1};"
        span_text = f"{self.text[span0:span1]}"
        idx_result = 0

        while idx_result < len(result):

            span0 = str(el.character_span[0] + result[idx_result] + 1)
            if idx_result == len(result) - 1:
                span1 = str(el.character_span[1])
                char_span_text += span0 + " " + span1
            else:
                span1 = str(el.character_span[0] + result[idx_result + 1] - 1)
                char_span_text += span0 + " " + span1 + ";"

            # find the first non-space char to start the next text segment
            start_text_span = find_first_non_space(self.text, el.character_span[0] + result[idx_result])

            span_text += " " + f"{self.text[start_text_span:int(span1)]}"
            idx_result += 1

        return char_span_text, span_text

    def ISO_annotation(self):
        """
        Parameters
        ----------
                None

        Returns
        -------
                the ISO annotation in the .ann format
        """

        attribute_id = 1

        r = ""

        for participant_id in self.participants:
            participant = self.participants[participant_id]

            # brat does not deal well with multiline annotation, so
            # we need to split into several segments a multiline annotation
            if '\n' in participant.text:
                char_span_text, span_text = self._get_span_segments(participant)


                r += (participant_id + '\t' + 'Participant' + ' ' + char_span_text + '\t' + span_text + '\n')
            else:
                # T1 participant 0 22 O presidente de França
                span_text = self.text[participant.character_span[0]:participant.character_span[1]]
                r += (participant_id + '\t' + 'Participant' + ' ' + str(participant.character_span[0]) + ' ' + str(
                    participant.character_span[1]) + '\t' + span_text + '\n')

            # A1 Lexical_Head T1 Noun
            try:
                r += ('A' + str(attribute_id) + '\t' + 'Lexical_Head' +
                       ' ' + participant_id + ' ' + participant.lexical_head + '\n')
            except TypeError:
                print(attribute_id, type(attribute_id), participant_id,type(participant_id), \
                      participant.lexical_head, type(participant.lexical_head))

            attribute_id += 1

            # A2 Individuation T1 Individual
            r += ('A' + str(attribute_id) + '\t' + 'Individuation_Domain' +
                  ' ' + participant_id + ' ' + participant.individuation + '\n')
            attribute_id += 1

            # A3 participant_Type T1 Per
            r += ('A' + str(attribute_id) + '\t' + 'Participant_Type_Domain' +
                  ' ' + participant_id + ' ' + participant.type + '\n')
            attribute_id += 1

            # A4 Involvement T1 1
            r += ('A' + str(attribute_id) + '\t' + 'Involvement' +
                  ' ' + participant_id + ' ' + participant.involvement + '\n')
            attribute_id += 1

        for time_id in self.times:
            time = self.times[time_id]

            if '\n' in time.text:
                char_span_text, span_text = self._get_span_segments(time)

                r += (time_id + '\t' + 'Time' + ' ' + char_span_text + '\t' + span_text + '\n')
            else:
                # T26 TIME_X3 413 429 novembro de 2015
                span_text = self.text[time.character_span[0]:time.character_span[1]]
                r += (time_id + '\t' + 'Time' + ' ' + str(time.character_span[0]) + ' ' + str(
                    time.character_span[1]) + '\t' + span_text + '\n')

            # A55 Time_Type T26 Date
            # r += ('A' + str(attribute_id) + '\t' + 'Time_Type' + ' ' + time_id + ' ' + time.type + '\n')
            # attribute_id += 1

            # 6 AnnotatorNotes T26 value=2015-11-XX  ?????
            # r += ('A' + str(attribute_id) + '\t' + 'Value' + ' ' + time_id + ' ' + time.value + '\n')
            # attribute_id += 1

            # A107 FunctionInDocument T4 Publication_Time
            # r += ('A' + str(attribute_id) + '\t' + 'FunctionInDocument' + ' ' + time_id + ' ' + time.temporal_function + '\n')
            r += ('A' + str(attribute_id) + '\t' + 'TemporalFunction' +
                  ' ' + time_id + ' ' + time.temporal_function + '\n')
            attribute_id += 1

        for spatialRelation_id in self.spatial_relations:
            spatialRelation = self.spatial_relations[spatialRelation_id]

            r += (spatialRelation_id + '\t' + 'Spatial_Relation' + ' ' + str(spatialRelation.character_span[0]) + ' ' + str(
                spatialRelation.character_span[1]) + '\t' + spatialRelation.text + '\n')

            attr_dct = vars(spatialRelation)
            for att in attr_dct:
                if att != "character_span" and att != "text":
                    r +=(f"A{attribute_id}\t{capfirst(att)} {spatialRelation_id} {attr_dct[att]}\n")
                    attribute_id += 1

        for event_id in self.events:

            event = self.events[event_id]
            if len(event.text.strip()) == 0:
                continue

            if '\n' in event.text:
                char_span_text, span_text = self._get_span_segments(event)

                r += (event_id + '\t' + 'Event' + ' ' + char_span_text + '\t' + span_text + '\n')
            else:
                span_text = self.text[event.character_span[0]:event.character_span[1]]

                r += (event_id + '\t' + 'Event' + ' ' + str(event.character_span[0]) + ' ' + str(
                    event.character_span[1]) + '\t' + span_text + '\n')

            attr_dct = vars(event)
            for att in attr_dct:
                if att != "character_span" and att != "text" and attr_dct[att] is not None:
                    r += (f"A{attribute_id}\t{capfirst(att)} {event_id} {attr_dct[att]}\n")
                    attribute_id += 1

            #r += (f"A{attribute_id}\tFactuality {event_id} {event.factuality}\n")
            #import pdb
            # pdb.set_trace()
            attribute_id += 1

        for link_type in self.links:
            for link in self.links[link_type]:
                r += str(link) + "\n"

        return r

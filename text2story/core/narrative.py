"""
	text2story.core.narrative

	Narrative class
"""

from text2story.core.annotator import Annotator
from text2story.core.entity_structures import *
from text2story.core.link_structures import *
from text2story.core.utils import pairwise


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
    actors: dict{str -> Actor}
            the actors identified in the text.
            each key in the dict, of the form 'T' concatenated with some int, has an actor as a value.
    times: dict{str -> Time}
            the temporal expressions identified in the text.
            each key in the dict, of the form 'T' concatenated with some int, has an time as a value.
    obj_links: dict{str -> ObjectalLink}
            the corefs identified in the text
            each key in the dict, of the form 'R' concatenated with some int, has an coref as a value.

    Methods
    -------
    extract_actors(*tools)
            extracts all the actors in the text using the annotators defined in 'tools', updating self.actors
    extract_timexs(*tools)
            extracts all the timexs in the text using the annotators defined in 'tools', updating self.timexs
    extract_corefs(*tools)
            coreference resolution in the text using the tools 'tools', updating self.obj_rels
            typically, this call increases self.actors since news entities can be identified
    _get_actor_key(char_offset)
            returns the key of the actor with the corresponding character offset or None if such actor wasn't identified before
    _add_actor(char_offset)
            update self.actors by adding the new actor with character offset 'char_offset' and returns the key given to the new actor
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

        self.actors = {}
        self.times = {}
        self.events = {}
        self.obj_links = {}
        self.sem_links = {}

    def extract_actors(self, *tools):
        """
        Parameters
        ----------
        tools : str, ...
                the tools to be used in the annotation

        Returns
        -------
                self.actors updated
        """

        actors = Annotator(tools).extract_actors(self.lang,
                                                 self.text)  # annotations :: [(EntityStartOffset, EntityEndOffset, EntityPOSTag, EntityType)]

        for actor in actors:
            self.actors['T' + str(self._id)] = ActorEntity(self.text[actor[0][0]:actor[0][1]], actor[0], actor[1],
                                                           actor[2])
            self._id += 1

        return self.actors

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

        for event in events.itertuples():
            self.events["T" + str(self._id)
                        ] = EventEntity(event.actor, event.char_span)
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

        for cluster in clusters:
            for i in range(0, len(cluster) - 1):
                e1 = cluster[i]
                e2 = cluster[i + 1]

                # Get the actors
                arg1, arg2 = self._get_actor_key(e1), self._get_actor_key(e2)

                # If one of them wasn't identified in the actor extration, add it as a new actor
                if arg1 == None:
                    arg1 = self._add_actor(e1)
                if arg2 == None:
                    arg2 = self._add_actor(e2)

                # (Type (sameHead, partOf, ...), Arg1, Arg2)
                self.obj_links['R' + str(self._rel_id)
                               ] = ObjectalLink(arg1, arg2)
                self._rel_id += 1

        return self.obj_links

    def extract_semantic_role_links(self, *tools):
        """
        Find semantic role links between extracted actors and events.
        Since the SRL model is different from the NER model, this function maps actors found by the SRL model into
        actors that were already extracted. If the actor was not yet extracted, it adds a new one.

        Links actors to events by text order. If we have ACTOR1 -> EVENT1 -> ACTOR2 in the text,
        then we make two semantic role links - EVENT1 -> ROLE -> ACTOR1 | EVENT1 -> ROLE -> ACTOR2

        @param tools: Iterable of tools to be used

        @return: A dict with the SRL entities by key -> R10: SemanticRoleLink<10>
        """
        srl_by_sentence = Annotator(
            tools).extract_semantic_role_links(self.lang, self.text)

        # FIND OUT IF ARGUMENT OF SRL HAS AN ACTOR RETRIEVED BY THE NER COMPONENT
        # IF NOT, ADD A NEW ACTOR CORRESPONDING TO THE ARGUMENT
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
                actor_key = self._get_actor_key(
                    row.char_span, match_type="partial")

                if actor_key is not None:
                    key_list.append(actor_key)
                else:
                    if actor_key is None and event_key is None:
                        key_list.append("None")
                #else:
                #    actor_key = self._add_actor(
                #        row.char_span, lexical_head="Noun", actor_type="Other")
                #    key_list.append(actor_key)
            
            sentence_df["key"] = key_list

        # MAKE SEMANTIC ROLE LINK ENTITIES
        for sentence_df in srl_by_sentence:
            for row1, row2 in pairwise(sentence_df.itertuples()):
                sem1 = row1.sem_role_type
                sem2 = row2.sem_role_type

                if (sem1 == "EVENT") | (sem2 == "EVENT"):
                    if sem1 != "EVENT":
                        sem_role, actor, event = sem1, row1.key, row2.key
                    else:
                        sem_role, actor, event = sem2, row2.key, row1.key

                    if actor == "None" or event == "None":
                        continue

                    self.sem_links["R" + str(self._rel_id)] = SemanticRoleLink(
                        actor, event, sem_role.lower())
                    self._rel_id += 1

        return self.sem_links

    def _add_srlink(self, actor, event, sem_role):
        self.sem_links["R" + str(self._rel_id)
                       ] = SemanticRoleLink(actor, event, sem_role.lower())
        self._rel_id += 1

    def _get_actor_key(self, char_span, match_type="exact"):
        """
        Parameters
        ----------
        char_span : (int, int)
                the actor character offset of the actor to find the key
        match_type: str
                type of match for the character span.
                If exact, the span must be exactly the same to match an actor.
                If partial, the span must be partially contained in the actor span to match it.

        Returns
        -------
                the key of the actor with the corresponding character offset
                or None if it doesn't exist
        """
        if match_type == "exact":
            for key in self.actors.keys():
                if self.actors[key].character_span == char_span:
                    return str(key)
        elif match_type == "partial":
            for key in self.actors.keys():
                aSpan = self.actors[key].character_span
                if aSpan[0] <= char_span[0] <= aSpan[1]:
                    return str(key)
                elif aSpan[0] <= char_span[1] <= aSpan[1]:
                    return str(key)
        else:
            raise ValueError(
                f"Parameter match_type must be one of [exact, partial].\nInstead it was {match_type}")

        return None

    def _add_actor(self, char_span, lexical_head="Pronoun", actor_type="Other"):
        """
        Parameters
        ----------
        char_span : (int, int)
                the actor character offset
        lexical_head: str
                The lexical head of the actor: "Noun" or "Pronoun" -> Defaults to "Pronoun"
        actor_type: str
                The type of the actor as expressed by the NER models -> Defaults to "Other"

        Returns
        -------
                the key of the new added actor
        """
        key = 'T' + str(self._id)
        self.actors[key] = ActorEntity(self.text[char_span[0]:char_span[1]], char_span, lexical_head,
                                       actor_type)  # Hard-coded lexical head and type as 'Pronoun' and 'Other', resp., for now

        self._id += 1

        return key

    def _get_event_key(self, char_span, match_type="exact"):
        """
        Get the key of an event entity based on its character span on the full document text
        todo: Esta função está redundante com a _get_actor_key. Só muda a lista em que se procura

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
            self.text[char_span[0]:char_span[1]], char_span)

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

        for actor_id in self.actors:
            actor = self.actors[actor_id]
            if '\n' in actor.text:
                continue
            # T1 ACTOR 0 22 O presidente de França
            r += (actor_id + '\t' + 'Participant' + ' ' + str(actor.character_span[0]) + ' ' + str(
                actor.character_span[1]) + '\t' + actor.text + '\n')

            # A1 Lexical_Head T1 Noun
            r += ('A' + str(attribute_id) + '\t' + 'Lexical_Head' +
                  ' ' + actor_id + ' ' + actor.lexical_head + '\n')
            attribute_id += 1

            # A2 Individuation T1 Individual
            r += ('A' + str(attribute_id) + '\t' + 'Individuation_Domain' +
                  ' ' + actor_id + ' ' + actor.individuation + '\n')
            attribute_id += 1

            # A3 Actor_Type T1 Per
            r += ('A' + str(attribute_id) + '\t' + 'Participant_Type_Domain' +
                  ' ' + actor_id + ' ' + actor.type + '\n')
            attribute_id += 1

            # A4 Involvement T1 1
            r += ('A' + str(attribute_id) + '\t' + 'Involvement' +
                  ' ' + actor_id + ' ' + actor.involvement + '\n')
            attribute_id += 1

        for time_id in self.times:
            time = self.times[time_id]

            # T26 TIME_X3 413 429 novembro de 2015
            r += (time_id + '\t' + 'Time' + ' ' + str(time.character_span[0]) + ' ' + str(
                time.character_span[1]) + '\t' + time.text + '\n')

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

        for event_id in self.events:

            event = self.events[event_id]
            if len(event.text.strip()) == 0:
                continue
            # T22 EVENT 312 328 is strengthening
            r += (event_id + '\t' + 'Event' + ' ' + str(event.character_span[0]) + ' ' + str(
                event.character_span[1]) + '\t' + event.text + '\n')

            # E9 EVENT:T22
            #r += (f"{event_id}\tEVENT:T{str(self._id)}\n")

            r += (f"A{attribute_id}\tClass {event_id} {event.event_class}\n")
            attribute_id += 1

            r += (f"A{attribute_id}\tTense {event_id} {event.tense}\n")
            attribute_id += 1

            r += (f"A{attribute_id}\tPolarity {event_id} {event.polarity}\n")
            attribute_id += 1

            #r += (f"A{attribute_id}\tFactuality {event_id} {event.factuality}\n")
            #import pdb
            # pdb.set_trace()
            attribute_id += 1

        for objectal_link_id in self.obj_links:
            obj_link = self.obj_links[objectal_link_id]

            # R34 OBJ_REL_objIdentity Arg1:T3 Arg2:T1
            #r += (objectal_link_id + '\t' + 'OBJ_REL_' + obj_link.type + ' ' + 'Arg1:' + obj_link.arg1 + ' ' + 'Arg2:' + obj_link.arg2 + '\n')
            r += (objectal_link_id + '\t' + 'OLINK_' + obj_link.type + ' ' +
                  'Arg1:' + obj_link.arg1 + ' ' + 'Arg2:' + obj_link.arg2 + '\n')

        for sem_link_id in self.sem_links:
            sem_link = self.sem_links[sem_link_id]

            # R35 SEMROLE_theme Arg1:E1 Arg2:T1
        #	r += (f"{sem_link_id}\tSEMROLE_{sem_link.type} Arg1:{sem_link.event} Arg2:{sem_link.actor}\n")
            r += (f"{sem_link_id}\tSRLINK_{sem_link.type} Arg1:{sem_link.event} Arg2:{sem_link.actor}\n")

        return r

"""
	text2story.core.entity_structures

	Entity structures classes (participant, TimeX and Event)
"""
import sys


class ParticipantEntity:
    """
    Representation of an Participant entity.

    Attributes
    ----------
    text: str
        The textual representation of the participant.
    character_span: tuple[int, int]
        The character span of the participant.
    lexical_head: str
        The lexical head of the participant.
        Possible values are: 'Noun' or 'Pronoun'.
    type: str
        The type of the participant.
        Possible values are: 'Per', 'Org', 'Loc', 'Obj', 'Nat' or 'Other'.
    individuation: str
        Stipulation of whether the participant is a set, a single individual, or a mass quantity.
        Possible values are: 'Set', 'Individual' or 'Mass'.
        NOTE: For now, using the label 'Individual' to all participants.
    involvement: str
        The specification of how many entities or how much of the domain are/is participating in an event.
        Possible values are: '0', '1', '>1', 'All' or 'Und'.
        NOTE: For now, using the label '1' to all participants.
    """

    def __init__(self, text, character_span, lexical_head, participant_type):
        self.text = text
        self.character_span = character_span
        self.lexical_head = lexical_head
        self.type = participant_type
        self.individuation = 'Individual'
        self.involvement = '1'


class TimeEntity:
    """
    Representation of a time entity.

    Attributes
    ----------
    text: str
        The textual representation of the time.
    character_span: tuple[int, int]
        The character span of the time.
    value: str
        The value of the time.
        Possible values are: value ::=  Duration | Date | Time | WeekDate | WeekTime | Season | PartOfYear | PaPrFu(Past Present Future Reference)
    type: str
        The type of the time.
        Possible values are: 'Date', 'Time', 'Duration' and 'Set'.
    temporal_function: str
        Possible values are: 'None' or 'Publication_Time'.
    """

    def __init__(self, text, character_span, timex_type, temporal_function='Publication_Time'):
        self.text = text
        self.character_span = character_span
        #self.value = value
        self.type = timex_type
        self.temporal_function = temporal_function


class EventEntity:
    """
    Representation of an event entity.
        TODO: Annotations (A)
    """

    def __init__(self, text, character_span,**kwargs):
        self.text = text
        self.character_span = character_span

        for attr_name, attr_value in kwargs.items():
            if attr_name.lower() == "event_class":
                setattr(self, "Class", attr_value)
            elif attr_name.lower() == "event_type":
                setattr(self, "Event_Type", attr_value)
            else:
                setattr(self,attr_name, attr_value)

        if hasattr(self, "Event_Type"):
            event_type = self.__getattribute__("Event_Type")
            # deafult value for event_type
            setattr(self, "Event_Type","Process")

        #self.factuality = "Factual"
        #self.tense = "Pres"

class SpatialRelationEntity:
    def __init__(self, text, character_span,**kwargs):
        self.text = text
        self.character_span = character_span

        for attr_name, attr_value in kwargs.items():
            setattr(self,attr_name, attr_value)


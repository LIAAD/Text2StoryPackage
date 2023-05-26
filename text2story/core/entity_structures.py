"""
	text2story.core.entity_structures

	Entity structures classes (Actor, TimeX and Event)
"""


class ActorEntity:
    """
    Representation of an actor entity.

    Attributes
    ----------
    text: str
        The textual representation of the actor.
    character_span: tuple[int, int]
        The character span of the actor.
    lexical_head: str
        The lexical head of the actor.
        Possible values are: 'Noun' or 'Pronoun'.
    type: str
        The type of the actor.
        Possible values are: 'Per', 'Org', 'Loc', 'Obj', 'Nat' or 'Other'.
    individuation: str
        Stipulation of whether the actor is a set, a single individual, or a mass quantity.
        Possible values are: 'Set', 'Individual' or 'Mass'.
        NOTE: For now, using the label 'Individual' to all actors.
    involvement: str
        The specification of how many entities or how much of the domain are/is participating in an event.
        Possible values are: '0', '1', '>1', 'All' or 'Und'.
        NOTE: For now, using the label '1' to all actors.
    """

    def __init__(self, text, character_span, lexical_head, actor_type):
        self.text = text
        self.character_span = character_span
        self.lexical_head = lexical_head
        self.type = actor_type
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

    def __init__(self, text, character_span, value, timex_type, temporal_function='Publication_Time'):
        self.text = text
        self.character_span = character_span
        self.value = value
        self.type = timex_type
        self.temporal_function = temporal_function


class EventEntity:
    """
    Representation of an event entity.
        TODO: Annotations (A)
    """

    def __init__(self, text, character_span, event_class="Occurrence", polarity="Pos"):
        self.text = text
        self.character_span = character_span
        self.event_class = event_class
        self.polarity = polarity
        self.factuality = "Factual"
        self.tense = "Pres"

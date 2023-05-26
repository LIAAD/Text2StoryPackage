"""
	text2story.core.link_structures

	Link structure classes (Temporal, aspectual, subordination, semantic role and objectal)
"""


class TemporalLink:
    pass


class AspectualLink:
    pass


class SubordinationLink:
    pass


class SemanticRoleLink:
    def __init__(self, actor, event, type="theme"):
        self.type = type
        self.actor = actor
        self.event = event


class ObjectalLink:
    def __init__(self, arg1, arg2):
        self.type = 'objIdentity'
        self.arg1 = arg1
        self.arg2 = arg2

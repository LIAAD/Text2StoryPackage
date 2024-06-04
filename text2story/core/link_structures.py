"""
	text2story.core.link_structures

	Link structure classes (Temporal, aspectual, subordination, semantic role and objectal)
"""

from text2story.core.exceptions import  InvalidLink

class MeasureLink:
    def __init__(self, arg1, arg2, type='distance', id_rel = ""):
        self.type = type
        self.arg1 = arg1
        self.arg2 = arg2
        self.id = id_rel

    def __str__(self):
        return f"{self.id}\tMLINK_{self.type} Arg1:{self.arg1} Arg2:{self.arg2}"


class MovementLink:
    def __init__(self, arg1, arg2, type='spatialRelation', id_rel = ""):
        self.type = type
        self.arg1 = arg1
        self.arg2 = arg2
        self.id = id_rel

    def __str__(self):
        return f"{self.id}\tMOVELINK_{self.type} Arg1:{self.arg1} Arg2:{self.arg2}"


class QualitativeSpatialLink:
    def __init__(self, arg1, arg2, type='figure', id_rel = ""):
        self.type = type
        self.arg1 = arg1
        self.arg2 = arg2
        self.id = id_rel
    def __str__(self):
        return f"{self.id}\tQSLINK_{self.type} Arg1:{self.arg1} Arg2:{self.arg2}"



class SubordinationLink:

    def __init__(self, arg1, arg2, type='factive',id_rel = ""):
        self.type = type
        self.arg1 = arg1
        self.arg2 = arg2
        self.id = id_rel
    def __str__(self):
        return f"{self.id}\tSLINK_{self.type} Arg1:{self.arg1} Arg2:{self.arg2}"



class SemanticRoleLink:
    def __init__(self, actor, event, type="theme", id_rel=""):
        self.type = type
        self.arg1 = actor
        self.arg2 = event
        self.id = id_rel

    def __str__(self):
        return f"{self.id}\tSRLINK_{self.type} Arg1:{self.arg1} Arg2:{self.arg2}"


class ObjectalLink:
    def __init__(self, arg1, arg2, type='objIdentity', id_rel = ""):
        self.type = type
        self.arg1 = arg1
        self.arg2 = arg2
        self.id = id_rel
    def __str__(self):
        return f"{self.id}\tOLINK_{self.type} Arg1:{self.arg1} Arg2:{self.arg2}"


class TemporalLink:
    def __init__(self, arg1, arg2, type='includes', id_rel = ""):
        self.type = type
        self.arg1 = arg1
        self.arg2 = arg2
        self.id = id_rel

    def __str__(self):
        return f"{self.id}\tTLINK_{self.type} Arg1:{self.arg1} Arg2:{self.arg2}"

class AspectualLink:
    def __init__(self, arg1, arg2, type='initiates', id_rel = ""):
        self.type = type
        self.arg1 = arg1
        self.arg2 = arg2
        self.id = id_rel

    def __str__(self):
        return f"{self.id}\tALINK_{self.type} Arg1:{self.arg1} Arg2:{self.arg2}"

def createLinkObject( arg1, arg2, type, subtype, id_rel):
    if type == "TLINK":
        return TemporalLink(arg1, arg2, subtype, id_rel)
    elif type == "OLINK":
        return ObjectalLink(arg1, arg2, subtype, id_rel)
    elif type == "SRLINK":
        return SemanticRoleLink(arg1, arg2, subtype, id_rel)
    elif type == "SLINK":
        return SubordinationLink(arg1, arg2, subtype, id_rel)
    elif type == "QSLINK":
        return QualitativeSpatialLink(arg1, arg2, subtype, id_rel)
    elif type == "MOVELINK":
        return MovementLink(arg1, arg2, subtype, id_rel)
    elif type == "ALINK":
        return AspectualLink(arg1, arg2, subtype, id_rel)
    elif type == "MLINK":
        return MeasureLink(arg1, arg2, subtype, id_rel)
    else:
        raise InvalidLink(type)

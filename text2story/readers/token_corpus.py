class TokenCorpus:
    """
    This class should store the token text, and
    annotation information
    """

    def __init__(self, text=None, token_id=None, sentence=None):
        # fields in the Token object
        self.text = text
        self.id = token_id
        self.lemma = None
        self.ann = None
        self.sentence = sentence
        self.pos = None
        self.dep = None
        self.head = None
        self.head_pos = None
        self.head_lemma = None
        self.gov_verb = None
        self.gov_verb_idx = None
        self.srl = None
        self.sent_id = None
        self.clause_id = None

        self.offset = None
        self.attr = []
        self.ann_offset = []

        # a token can be annotated by more than one 
        # layer, therefore it can present more than one 
        # id
        self.id_ann = []

        # a list of items of type TokenRelation
        self.relations = []

    def get_attr_value(self,attr_name, token_type=None):
        for (type_, attr_map) in self.attr:
            if token_type != None and type_ == token_type:
                return attr_map.get(attr_name)
            else:
                return attr_map.get(attr_name)
    def is_type(self, token_type):
        for (type_, _) in self.attr:
            if type_ == token_type:
                return True
        return False

class TokenRelation:
    """
    Specifies a Relation between tokens
    """
    def __init__(self, rel_id, toks = [], rel_type = None, argn = None, arg_id = None):
        self.rel_id = rel_id
        self.toks = toks
        self.rel_type = rel_type
        self.argn = argn
        self.arg_id = arg_id

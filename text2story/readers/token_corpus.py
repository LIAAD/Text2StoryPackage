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

        self.offset = None
        self.attr = []

        # a token can be annotated by more than one 
        # layer, therefore it can present more than one 
        # id
        self.id_ann = []

        # a list of items of type TokenRelation
        self.relations = [] 

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

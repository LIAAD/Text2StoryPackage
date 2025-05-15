
from typing import List, Tuple
from text2story.readers.token_corpus import TokenCorpus

class Box:
    def __init__(self):
        self.variables = []
        self.box = []

    def __text2clauses(self, txt:str) -> List[Tuple[str,int,int]]:
        # TODO: write the function
        pass

    def __get_clause(self, tok:TokenCorpus, clause_lst:List[Tuple[str,int,int]]) -> Tuple[str,int,int]:
        # TODO: write the function
        # usar o atributo offset de tok para localizar qual oração
        # Busca binaria pela oracao a que o token pertence. Pesquisar tambem
        pass

    def __create_variable(self, entity:TokenCorpus)->Tuple[str,str,str]:
        pass

    def create_boxes(self, doc:List[TokenCorpus], doc_txt:str)-> List:
        clause_lst = self.__text2clauses(doc_txt)
        # TODO: ordenar esta lista de oracoes de acordo com o offset de inicio. Pesquisar
        # sobre ordenação de lista de tuplas
        box_lst = []
        current_box = Box()

        for tok in doc:

            # this means that tok has annotations
            if tok.attr != []: # [(ann_type, ann_map)] -> ('Event',{'Tense':'Past'...})
                variable = self.__create_variable(tok)
                clause = self.__get_clause(tok, clause_lst)
                # essa oração já foi analisada? Sim, então não preciso criar nova box. Não, crio nova box.
                # TODO 1): quando eu crio uma nova box. Atencao que pode ser recursiva!
                # TODO 2): adicionar a box a lista de box

                current_box.variables.append(variable)

        return box_lst





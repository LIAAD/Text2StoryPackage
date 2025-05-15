from markdown_it.rules_inline import entity


def get_all_entities(tok_lst, ann_type_lst=["time"]):
    entity_lst = {}

    for i in range(len(tok_lst)):
        tok = tok_lst[i]

        for attr_item in tok.attr:

            ann_type = attr_item[0]
            attr_map = attr_item[1]

            if ann_type.lower() in ann_type_lst:

                for id_ann in tok.id_ann:

                    if id_ann in entity_lst:
                        entity_lst[id_ann].append(tok)
                    else:
                        entity_lst[id_ann] = [tok]
                break

    return entity_lst
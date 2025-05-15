def get_all_participants_sent(tok_lst):
    """
    Get all participants from a document and index them by the sentence number
    @param tok_lst:
    @return:
    """

    participant_lst = {}
    participant_id_lst = set()

    for i in range(len(tok_lst)):
        tok = tok_lst[i]

        if tok.is_type("Participant") and tok.id_ann[0] not in participant_id_lst:
            if tok.sent_id in participant_lst:
                participant_lst[tok.sent_id].append(tok)
            else:
                participant_lst[tok.sent_id] = [tok]
            participant_id_lst.add(tok.id_ann[0])

    return participant_lst

def get_all_participants(tok_lst):
    participants_lst = {}

    for i in range(len(tok_lst)):
        tok = tok_lst[i]

        for attr_item in tok.attr:

            ann_type = attr_item[0]
            attr_map = attr_item[1]

            if "participant" in ann_type.lower():

                for id_ann in tok.id_ann:

                    if id_ann in participants_lst:
                        participants_lst[id_ann].append(tok)
                    else:
                        participants_lst[id_ann] = [tok]
                break

    return participants_lst
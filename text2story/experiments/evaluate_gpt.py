import os

from text2story.experiments.evaluation import process_evaluation, print_metrics_result


def run():
    narrative_elements = ["event", "participant", "time"]
    #target_dir = "/home/evelinamorim/UPorto/gpt_struct_me/resources/lusa_news/"
    #pred_dir = "/home/evelinamorim/UPorto/gpt_struct_me/brat/lusa_news/"
    target_dir = "/home/evelinamorim/UPorto/text2story2019/Data/AceISO_test/"
    pred_dir = "/home/evelinamorim/UPorto/gpt_struct_me/brat/ace"

    target_files = os.listdir(target_dir)
    pred_files = os.listdir(pred_dir)

    doc_lst = []
    for p_file in pred_files:
        if p_file in target_files:
            doc_lst.append((os.path.join(pred_dir, p_file),\
                           os.path.join(target_dir, p_file)))

    res = process_evaluation(narrative_elements, doc_lst)
    print_metrics_result(res)

if __name__ == "__main__":
    run()

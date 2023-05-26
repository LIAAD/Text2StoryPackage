import json
import sys
import os

from evaluation import start, build_evaluation, write_metrics_result



def read_config(config_file_name):

    with open(config_file_name, "r") as fd:
        config_exp = json.load(fd)
    return config_exp

def run():

    config_file_name = sys.argv[1]

    config_exp = read_config(config_file_name)

    for name_experiment in config_exp:

        if not(os.path.exists(name_experiment)):
            os.mkdir(name_experiment)

        language = config_exp[name_experiment]["language"]
        start(language)
        tools = config_exp[name_experiment]["tools"]

        narrative_elements = {"participant":tools["participant"],\
                          "time":tools["time"],\
                          "event":tools["event"],\
                          "srlink":tools["srlink"]}

        data_dir = config_exp[name_experiment]["data"]
        res = build_evaluation(narrative_elements=narrative_elements, language=language, data_dir=data_dir,
                           results_dir=name_experiment)

        with open("%s_metrics.txt" % name_experiment, "w") as fd:
            write_metrics_result(res, fd)


if __name__ == "__main__":
    run()

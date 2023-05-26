import os
import json
from flask import Flask, render_template, request, url_for
import parser as parser

app = Flask(__name__)

#BRAT_ANN_DIR = '../brat2drs/text_2_story_new'
BRAT_ANN_DIR = '../brat2drs/sample/'


@app.route('/brat2viz', methods=['GET', 'POST'])
def brat2viz():
    drs_files = parser.get_drs_files()

    selected_drs = request.form.get('select_drs')
    # nunca esta pegando nada aqui em select drs, pq?

    if selected_drs:
        drs_file = selected_drs
    else:
        drs_file = drs_files[0]

    selected_vis = request.form.get('select_vis')

    common_file_name = drs_file.split('/')[-1].split('_drs')[0]
    #print(selected_vis,selected_drs)

    if selected_vis == 'ann_text':
        vis = 'ann_text'
        ann_file = os.path.join(BRAT_ANN_DIR, common_file_name + '.ann')
        with open(ann_file, 'r') as fp:
            ann_text = fp.readlines()
        return render_template('index.html', drs_files=drs_files, selected_drs=drs_file, selected_vis=vis,
                               text=ann_text)

    elif selected_vis == 'drs_text':
        vis = 'drs_text'
        with open(drs_file, 'r') as fp:
            drs_text = fp.readlines()
        return render_template('index.html', drs_files=drs_files, selected_drs=drs_file, selected_vis=vis,
                               text=drs_text)

    elif selected_vis == 'msc':
        vis = 'msc'
        msc = parser.get_msc_data(drs_file)
        return render_template('index.html', drs_files=drs_files, selected_drs=drs_file, selected_vis=vis, msc=msc)

    elif selected_vis == 'graph':
        vis = 'graph'
        actors, non_ev_rels, ev_rels = parser.get_graph_data(drs_file)
        actors = json.dumps(actors)
        non_ev_rels = json.dumps(non_ev_rels)
        ev_rels = json.dumps(ev_rels)
        return render_template('index.html', drs_files=drs_files, selected_drs=drs_file, selected_vis=vis,
                               actors=actors, non_ev_rels=non_ev_rels, ev_rels=ev_rels)

    # Default
    else:
        vis = 'news_text'
        news_file = os.path.join(BRAT_ANN_DIR, common_file_name + '.txt')
        with open(news_file, 'r') as fp:
            news_text = fp.readlines()
        return render_template('index.html', drs_files=drs_files, selected_drs=drs_file, selected_vis=vis,
                               text=news_text)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5055, debug=True)

from text2story.select.bubble import BubbleMap, BigBubble, Bubble 
from text2story.readers.read_brat import ReadBrat
from pdf2image import convert_from_path

import os
import re
import sys

from latexcompiler import LC

tikz_doc = "\\documentclass{standalone}\n\\usepackage{pgf,tikz}\n\\usetikzlibrary{arrows}\n\\usepackage{listofitems} % for \\readlist to create arrays"

def bubble_header():
    return '''
\\usepackage[T1]{fontenc}
\\usetikzlibrary{calc}
\\usetikzlibrary{arrows.meta}
\\usepackage{relsize}

    % Create style for node circles
	           \\tikzstyle{state}=[circle,
                   minimum size =13cm,
                   draw=white,
                   thick
                ]

               \\tikzstyle{report}=[
                  circle,
                  minimum size =5cm,
                  draw=white,
                  fill=orange!80,
                  thick
                ]

                \\tikzstyle{agent}=[
                  rectangle,
                  draw=black,
                  fill=teal!80,
                  rounded corners=1pt,
                  thick
                ]

                \\tikzstyle{event}=[
                  circle,
                  minimum size =1cm,
                  draw=black,
                  fill=white,
                node distance=\\nodeDist,
                thick
              ]

    

             \\begin{document}

             \\begin{tikzpicture}


             % esse sera o container
            \\node[state] (A) at (0,0) {};'''

def compute_angles(nevents):
    if nevents < 4:
        count_angle = 90
    else:
        count_angle = 360 // nevents
    fst_angle = count_angle
    angle_lst = []

    while fst_angle >= -360 and len(angle_lst) < nevents:
        angle_lst.append(fst_angle)
        fst_angle = fst_angle - count_angle
    return angle_lst

def start_inside_angle(bubble_idx, nbubbles):

    
    angle_lst = compute_angles(nbubbles)

    bubble_angle = angle_lst[bubble_idx]

    if bubble_angle > -45 and bubble_angle < 45:
        return 0
    elif bubble_angle > - 135 and bubble_angle <= -45:
        return -90
    elif bubble_angle > -280 and bubble_angle <= -135:
        return -180
    else:
        return -270

def draw_big_bubble(big_bubble, type_event,  angle, map_id2nodes, has_agent):

    sent_id = big_bubble.bubble_.event.sent_id

    tikz_str = ""
    
    event_text = big_bubble.bubble_.get_event_text() 
    id_ann = big_bubble.bubble_.event.id_ann[0]

    sent_id_text = str(float(sent_id))
    whole, frac = sent_id_text.split(".") 

    name_node = "%s%s_%s" % (type_event.lower(), whole, frac[:1])
    map_id2nodes[id_ann] = name_node
    big_bubble.bubble_.name = name_node

    tikz_str += "\\node[report] (%s) at (A.%d) {\\footnotesize %s:%s};\n" % (name_node, angle, sent_id_text, event_text)
    tikz_str += "\n" 

    if has_agent and big_bubble.bubble_.agent.span != []:
        # draw agents
        shift = 0.5
        current_agent = big_bubble.bubble_.agent
        idx_agent = 0

        while current_agent != None:
            name_agent_node = "agent%s_%s_%d" % (whole, frac[:1], idx_agent)
            current_agent.name = name_agent_node
            tok_txt_agent = [tok.text for tok in current_agent.span]
            agent_txt = " ".join(tok_txt_agent)

            tikz_str += "\\node[agent] (%s) at ([shift=({90:%f cm})]%s) {\\footnotesize %s};\n" % (name_agent_node, shift, name_node, agent_txt)
            tikz_str += "\n" 

            shift += 0.5
            idx_agent += 1
            current_agent = current_agent.next_agent


    return tikz_str

def draw_little_bubbles(big_bubble, map_id2nodes):


    sent_id = big_bubble.bubble_.event.sent_id

    # draw the little bubbles
    little_bubble_lst_str = "\\def\\eventlst%d{{" % (sent_id)

    # sort little bubbles by their offset text position
    big_bubble.sort_by_offset()

    for lbubble in big_bubble.little_bubbles:

        little_bubble_lst_str =  little_bubble_lst_str + "\"%s\"," % lbubble.get_event_text()

    little_bubble_lst_str = little_bubble_lst_str + "}}\n"

    nevents = len(big_bubble.little_bubbles)

    for idx in range(nevents):
        id_ann = big_bubble.little_bubbles[idx].event.id_ann[0]
        name_node = "r%devent%d" % (sent_id,  idx)
        big_bubble.little_bubbles[idx].name = name_node

        map_id2nodes[id_ann] = name_node

    angle_lst = compute_angles(nevents)

    little_bubble_lst_str = little_bubble_lst_str + "\n\\foreach[count=\\i from 0] \\angle in {"

    angle_lst_str = ",".join(map(str,angle_lst))

    little_bubble_lst_str = little_bubble_lst_str + angle_lst_str + "}{\n"
    
    little_bubble_lst_str = little_bubble_lst_str + "\\node[event] (r%devent\\i) at ([shift=({\\angle:2 cm})]%s) {\\scriptsize \\pgfmathparse{\\eventlst%d[\\i]} \\pgfmathresult};}" % (sent_id,big_bubble.bubble_.name,sent_id)
    
    return little_bubble_lst_str

def draw_agent_relation(agent):

    rel_str = "\n"

    for rel_agent in agent.relations_agent:
        name_rel_agent = rel_agent.agent.name
        link_text = rel_agent.rel_type.split("_")[1]

        if rel_agent.out:
            rel_str = rel_str + "\n \\draw [<-, very thick, arrows = {-Latex[length=3mm, width=1.5mm]}] "
        else:
            rel_str = rel_str + "\n \\draw [->, very thick, arrows = {-Latex[length=3mm, width=1.5mm]}] "

        rel_str = rel_str + "(%s) edge [out=90,midway, above] node[sloped, fill=white] {\\tiny %s} (%s);\n" % (agent.name, link_text, rel_agent.agent.name)

    for rel_bubble in agent.relations_bubbles:
        if isinstance(rel_bubble.bubble, BigBubble):
            name_rel_bubble = rel_bubble.bubble.bubble_.name
        else:
            name_rel_bubble = rel_bubble.bubble.name

        link_text = rel_bubble.rel_type.split("_")[1]

        if rel_bubble.out:
            rel_str = rel_str + "\n \\draw [<-, very thick, arrows = {-Latex[length=3mm, width=1.5mm]}] "
        else:
            rel_str = rel_str + "\n \\draw [->, very thick, arrows = {-Latex[length=3mm, width=1.5mm]}] "

        rel_str = rel_str + "(%s) edge [out=90,midway, above, looseness = 3] node[sloped, fill=white] {\\tiny %s} (%s.text);\n" % (agent.name, link_text, name_rel_bubble)

    return rel_str 

def draw_bubble_relation(bubble1, rel, map_id2nodes, inside_angle_map, nevents = 0):

    
    if isinstance(bubble1, BigBubble):
        bubble1_event_id = bubble1.bubble_.event.id_ann[0]
        name_bubble1 = bubble1.bubble_.name
        # it gets the sent_id only if bubble1 is a big_buble
        sent_id = int(re.match('.*?([0-9]+)$', name_bubble1).group(1))

    else:
        bubble1_event_id = bubble1.event.id_ann[0]
        name_bubble1 = bubble1.name
        sent_id = int(re.match("r\d+", name_bubble1).group(0)[1:])

    if bubble1_event_id not in map_id2nodes:
        return

    bubble2 = rel.bubble_pointer

    if isinstance(bubble2, BigBubble):
        name_bubble2 = bubble2.bubble_.name
        bubble2_event_id = bubble2.bubble_.event.id_ann[0]
    else:
        name_bubble2 = bubble2.name
        bubble2_event_id = bubble2.event.id_ann[0]

    if bubble2_event_id not in map_id2nodes:
        return

    rel_str = ""
    

    edge_type = rel.edge_type
    name_edge = edge_type.split("_")[1]

    
    # get the index number (a counter that is the index of the event inside a big bubble)
    bubble_idx = int(re.match('.*?([0-9]+)$', name_bubble2).group(1))

    # bubble2 is in the same sentence of bubble1    
    if name_bubble2.startswith("r%d" % sent_id):

        #print(re.match('.*?([0-9]+)$', name_bubble_pointer).group())


        if name_bubble2 not in inside_angle_map:
            inside_angle_map[name_bubble2] = start_inside_angle(bubble_idx, nevents)

        inside_angle = inside_angle_map[name_bubble2]

        rel_str = rel_str + "\n \\draw [->, dashed, arrows = {-Latex[length=5mm, width=2mm]}] "


        if isinstance(bubble1, BigBubble):
            inside_angle1 = 0
            rel_str = rel_str + "(%s.center)" % name_bubble1
            rel_str = rel_str + " edge [out=%d,in=%d] node[midway, fill=white, sloped] {\\footnotesize %s} (%s);" % (
            inside_angle1, inside_angle, name_edge, name_bubble2)

        else:
            # TODO: testar com inside angle com o bubble1
            if name_bubble1 not in inside_angle_map:
                # get the index number (a counter that is the index of the event inside a big bubble)
                bubble_idx1 = int(re.match('.*?([0-9]+)$', name_bubble1).group(1))
                inside_angle_map[name_bubble1] = start_inside_angle(bubble_idx1, nevents)

            inside_angle1 = inside_angle_map[name_bubble1]
            #inside_angle_map[name_bubble1] -= 60
            #rel_str = rel_str + "(%s)" % name_bubble1
            if rel.out:
                rel_str = rel_str + " (%s) edge [out=%d,in=%d] node[midway, fill=white, sloped] {\\footnotesize %s} (%s);" % \
                          (name_bubble2, inside_angle1, inside_angle,name_edge,  name_bubble1)
            else:
                rel_str = rel_str + " (%s) edge [out=%d,in=%d] node[midway, fill=white, sloped] {\\footnotesize %s} (%s);" % \
                          (name_bubble1, inside_angle1, inside_angle, name_edge, name_bubble2)

        #inside_angle_map[name_bubble2] -= 60
               
    else:

        rel_str = rel_str + "\n \\draw [->, very thick, arrows = {-Latex[length=5mm, width=2mm]}] "

        isbubble = re.match("r\d+", name_bubble1)
        if isbubble:

            sent_id1 = isbubble.group() # the sent id, if bubble 1 is not a big bubble
            sent_id2 = re.match("r\d+", name_bubble2) # the sent_id , if bubble 2 is not a big bubble

            if sent_id2:
                sent_id2 = sent_id2.group()

                # two little bubbles inside the same big bubble?
                if sent_id1 == sent_id2:

                    if name_bubble2 not in inside_angle_map:
                        inside_angle_map[name_bubble2] = start_inside_angle(bubble_idx,nevents)
                        
                    inside_angle = inside_angle_map[name_bubble2]
                    rel_str = rel_str + "(%s) edge [out=0,in=%d] node[midway, fill=white, sloped] {\\footnotesize %s} (%s);" % (name_bubble1,inside_angle,name_edge,  name_bubble2)

                    inside_angle_map[name_bubble2] -= 60    

                else:
                    if rel.out:
                        rel_str = rel_str + "(%s) -- (%s) node[midway,fill=white,sloped] {\\footnotesize %s};" % (
                        name_bubble2, name_bubble1, name_edge)
                    else:
                        rel_str = rel_str + "(%s) -- (%s) node[midway,fill=white,sloped] {\\footnotesize %s};" % (name_bubble1, name_bubble2, name_edge)
            else:
                rel_str = rel_str + "(%s) -- (%s) node[midway,fill=white,sloped] {\\footnotesize %s};" % (name_bubble1, name_bubble2, name_edge)
        else:

            rel_str = rel_str + "(%s) -- (%s) node[midway,fill=white,sloped] {\\footnotesize %s};" % (name_bubble1, name_bubble2, name_edge)

    
    return rel_str


def build_fig(tok_lst, **kwargs):
    type_event = kwargs["type_event"]
    type_rel_lst = kwargs["type_rel_lst"]
    has_agent = kwargs["has_agent"]

    tikz_str = tikz_doc + "\n" + bubble_header()
    
    bubble_map = BubbleMap()
    bubble_map.build_map(tok_lst, type_event, type_rel_lst)

    nevents =  len(bubble_map.map)
    angle_lst = compute_angles(nevents)

    map_id2nodes = {} # a map of id_ann to nodes id
    
    for idx, sent_id in enumerate(bubble_map.map.keys()):

        big_bubble = bubble_map.map[sent_id]
        tikz_str = tikz_str + "\n" + draw_big_bubble(big_bubble, type_event, angle_lst[idx], map_id2nodes,has_agent)

        tikz_str = tikz_str + "\n" + draw_little_bubbles(big_bubble, map_id2nodes)

    
    tikz_str = tikz_str + "\n"

    
    inside_angle_map = {}
    #print("Big Bubbles: ", len(bubble_map.map))
    for sent_id in bubble_map.map:


        big_bubble = bubble_map.map[sent_id]
        
        bubble_event_id = big_bubble.bubble_.event.id_ann[0]


        if bubble_event_id not in map_id2nodes:
            continue

        nevents = len(big_bubble.little_bubbles)

        # draw agent relations
        if has_agent and big_bubble.bubble_.agent.span != []:
            current_agent = big_bubble.bubble_.agent

            while (current_agent != None):
                rel_str = draw_agent_relation(current_agent)
                if rel_str is not None:
                    tikz_str += rel_str
                current_agent = current_agent.next_agent

        # draw relations of the big bubble
        for rel in big_bubble.bubble_.relations: 

            #if rel.edge_type.startswith("TLINK") and rel.edge_type != "TLINK_identity":
            if rel.edge_type.startswith("TLINK"):
                continue
            rel_str = draw_bubble_relation(big_bubble, rel, map_id2nodes, inside_angle_map, nevents)  
            if rel_str is not None:
                tikz_str += rel_str

            
            tikz_str = tikz_str + "\n"
    
         # draw relations of the little bubbles
        for little_bubble in big_bubble.little_bubbles:

            lbubble_event_id = little_bubble.event.id_ann[0]

            if lbubble_event_id not in map_id2nodes:
                continue

            # is it in the same big bubble? adjust the angle inside
            for rel in little_bubble.relations:
                #print("-->", rel.bubble_pointer.event.id_ann)
                #if rel.edge_type.startswith("TLINK") and rel.edge_type != "TLINK_identity":
                #if rel.edge_type.startswith("TLINK"):
                #    continue
                rel_str = draw_bubble_relation(little_bubble, rel, map_id2nodes, inside_angle_map, nevents)
                if rel_str is not None:
                    tikz_str += rel_str

            print()

                

    tikz_str = tikz_str + "\n" + "\\end{tikzpicture}\n\\end{document}\n"

    return tikz_str, bubble_map

def build_fig_ann(ann_file, output_dir,**kwargs):

    lang = kwargs["lang"]
    reader = ReadBrat(lang)
    data = reader.process_file(ann_file)

    #Serao apenas as relações temporais que nao envolvam reporting events ligado pelo TLINK identity.
    tikz_str, bubble_map = build_fig(data, **kwargs)

    bubble_map.to_json(os.path.join(output_dir,"output.txt"))

    ann_file_base = os.path.basename(ann_file)

    output_file = os.path.join(output_dir, "%s.tex" % ann_file_base)
    with open(output_file, "w") as fd:
        fd.write(tikz_str)

    LC.compile_document(tex_engine="pdflatex", bib_engine="bibtex",no_bib=True, path=output_file, folder_name=output_dir)
    #pdfl = ptex.PDFLaTeX.from_texfile(output_file)
    #os.system("pdflatex -halt-on-error -output-directory %s %s" % (output_dir, output_file))

    pdf_file = os.path.join(output_dir, "%s.pdf" % ann_file_base)

    #pdf, log, completed_process = pdfl.create_pdf()
    #with open(pdf_file, 'wb') as pdfout:
    #    pdfout.write(pdf)

    png_file = os.path.join(output_dir, "%s.png" % ann_file_base)
    log_file = os.path.join(output_dir, "%s.log" % ann_file_base)
    aux_file = os.path.join(output_dir, "%s.aux" % ann_file_base)
    sync_file = os.path.join(output_dir, "%s.synctex.gz" % ann_file_base)

    pages = convert_from_path(pdf_file, 500)
    for page in pages:
        # usually only one page
        page.save(png_file, 'PNG')

    #os.system("convert  -density 600 %s  %s" % (pdf_file, png_file))
    os.system("rm %s %s %s %s" % (pdf_file, log_file, aux_file, sync_file))



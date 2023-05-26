from Text2Viz import Text2Viz, require_input_from_user

if __name__ == '__main__':
    visualizations_available = ['ann_text', 'drs', 'graph']

    visualization_required = input(f"Select the desired Visualization: {visualizations_available}\n")

    if visualization_required in visualizations_available:

        [language, text] = require_input_from_user()

        text2viz = Text2Viz('en', '', '2021-12-31', visualization_required)

        if language == 'pt':
            text2viz.narrative.lang = 'pt'
        elif language == 'en':
            text2viz.narrative.lang = 'en'

        text2viz.narrative.text = text

        text2viz.run()
    else:
        print("Visualization not valid")

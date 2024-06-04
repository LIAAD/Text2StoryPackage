from text2story.readers import read
import re

PARENTHESES = r"\([^()]*\)"
import pandas as pd

class ReadCSV(read.Read):

    def __init__(self, columns=None):
        self.columns = columns

    def _standardize_cols(self, columns):
        std_cols_map = {}
        for c in columns:
            if len(c.strip()) > 0:
                new_col_name = re.sub(PARENTHESES,"",c)
                new_col_name = new_col_name.strip().lower()
                new_col_name = new_col_name.replace(" ","_")
                std_cols_map[c] = new_col_name
            else:
                std_cols_map[c] = ''
        return  std_cols_map
    def process(self, data_dir):
        """
        Process a set of files of a corpus

        @param string: path of data corpus

        @return: a list of tokens
        """
        pass
    def get_data(self, data_lines, **kwargs):

        for data in data_lines:
            if 'span_id' in kwargs:
                if data['span_id'] == kwargs['span_id']:
                    return data
    def process_file(self, data_file):
        """
        Process only one file of a corpus.
        """
        df = pd.read_csv(data_file)
        col_names_nonstd = df.columns.tolist()
        col_names_map = self._standardize_cols(col_names_nonstd)

        data_lst = []

        for idx, row in df.iterrows():
            data = {}
            for col_name in  col_names_map:

                if self.columns:
                    if col_names_map[col_name] in self.columns:
                        data[col_names_map[col_name]] = row[col_name]
                else:
                    if col_name != '':
                        data[col_names_map[col_name]] = row[col_name]
            if data != {}:
                data_lst.append(data)


        return data_lst

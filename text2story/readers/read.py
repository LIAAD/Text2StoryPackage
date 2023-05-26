

class Read:
    """
    Abstract class to read all different kind of corpus
    """

    def process(self, data_dir):
        """
        Process a set of files of a corpus

        @param string: path of data corpus

        @return: a list of tokens
        """
        pass

    def process_file(self, data_file):
        """
        Process only one file of a corpus.
        """
        pass

    def __process_annotations(self, data_file):
        """
        Process specific annotations of the given data file
        """
        pass



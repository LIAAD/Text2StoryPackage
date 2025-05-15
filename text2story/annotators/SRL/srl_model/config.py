from transformers import PretrainedConfig

class SRLModelConfig(PretrainedConfig):
    model_type = "srl"
    
    def __init__(
        self,
        num_labels=0,
        bert_model_name="bert-base-uncased",
        embedding_dropout=0.0,
        label2id = {},
        id2label = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.bert_model_name = bert_model_name
        self.embedding_dropout = embedding_dropout
        self.label2id = label2id
        self.id2label = id2label

    def to_dict(self):
        config_dict = super().to_dict()

        config_dict["num_labels"] = self.num_labels
        # config_dict["bert_model_name"] = self.bert_model_name
        # config_dict["embedding_dropout"] = self.embedding_dropout

        return config_dict

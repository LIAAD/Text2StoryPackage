import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from .config import SRLModelConfig


class SRLModel(PreTrainedModel):
    config_class = SRLModelConfig

    def __init__(self, config):
        super().__init__(config)

        print(config.num_labels, config.bert_model_name, config.embedding_dropout)

        # Load pre-trained transformer-based model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)
        self.transformer = AutoModel.from_pretrained(
            config.bert_model_name,
            num_labels=config.num_labels,
            output_hidden_states=True,
        )
        self.transformer.config.id2label = config.id2label
        self.transformer.config.label2id = config.label2id

        # The roberta models do not have token_type_embeddings
        # (the type_vocab_size is 1)
        # but we use this to pass the verb's position
        # so we need to change the model and initialize the embeddings randomly
        if "xlm" in config.bert_model_name or "roberta" in config.bert_model_name:
            self.transformer.config.type_vocab_size = 2
            # Create a new Embeddings layer, with 2 possible segments IDs instead of 1
            self.transformer.embeddings.token_type_embeddings = nn.Embedding(
                2, self.transformer.config.hidden_size
            )
            # Initialize it
            self.transformer.embeddings.token_type_embeddings.weight.data.normal_(
                mean=0.0, std=self.transformer.config.initializer_range
            )

        # Linear layer for tag projection
        self.tag_projection_layer = nn.Linear(
            self.transformer.config.hidden_size, config.num_labels
        )

        # Dropout layer for embeddings
        self.embedding_dropout = nn.Dropout(p=config.embedding_dropout)

        # Number of labels
        self.num_labels = config.num_labels

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):

        # print("FORWARD")
        # print(labels)

        # Forward pass through the transformer model
        # Returns BaseModelOutputWithPoolingAndCrossAttentions
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract the [CLS] token representation
        # cls_output = outputs.pooler_output

        bert_embedding = outputs.last_hidden_state

        # Apply dropout to the embeddings
        embedded_text_input = self.embedding_dropout(bert_embedding)

        # Project to tag space
        logits = self.tag_projection_layer(embedded_text_input)

        reshaped_log_probs = logits.view(-1, self.num_labels)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
            logits.size(0), logits.size(1), -1
        )

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        output_dict["attention_mask"] = attention_mask
        output_dict["input_ids"] = input_ids
        # output_dict["start_offsets"] = start_offsets

        if labels is not None:
            # print("Input", logits.view(-1, self.num_labels).size())
            # print("Target", labels.view(-1).size())
            # print("C", self.num_labels)
            # print("AllenNLP function", logits.size(-1))
            # Could consider passing ignore_index as 0 (pad index) for minor optimization
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.num_labels), labels.view(-1)
            )
            output_dict["loss"] = loss
        return output_dict

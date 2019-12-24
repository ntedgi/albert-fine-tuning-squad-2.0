import torch
from transformers import *


class AlbertModelForQuestionAnswering:
    def __init__(self, path_to_model):
        self.tokenizer = AlbertTokenizer.from_pretrained(path_to_model)
        self.model = AlbertForQuestionAnswering.from_pretrained(path_to_model)

    def predict(self, question, paragraphs):
        answers = []
        for paragraph in paragraphs:
            input_text = "[CLS] " + question + " [SEP] " + paragraph['text'] + " [SEP]"
            input_ids = self.tokenizer.encode(input_text)
            token_type_ids = [0 if i <= input_ids.index(3) else 1 for i in range(len(input_ids))]
            start_scores, end_scores, hidden_states, output_attentions = self.model(torch.tensor([input_ids]),
                                                                                    token_type_ids=torch.tensor(
                                                                                        [token_type_ids]))
            all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            result = ' '.join(all_tokens[torch.argmax(start_scores):torch.argmax(end_scores) + 1]).replace("▁",'')
            answers.append((paragraph['id'], result))
        return answers

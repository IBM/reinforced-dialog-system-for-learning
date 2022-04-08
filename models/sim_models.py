import torch
from transformers import BertTokenizer, BertModel


class Unlearned:
    def __init__(self, device=None):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if not device:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.model = BertModel.from_pretrained("bert-base-uncased").to(self.device)

    def get_relv(self, sent1, sent2):
        encoded_input1 = self.tokenizer(sent1, return_tensors='pt').to(self.device)
        encoded_input2 = self.tokenizer(sent2, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output1 = self.model(**encoded_input1)[0]
            output2 = self.model(**encoded_input2)[0]
        rep1 = torch.mean(output1, 1).reshape(-1)
        rep2 = torch.mean(output2, 1).reshape(-1)
        relv = (torch.dot(rep1, rep2) / (torch.norm(rep1) * torch.norm(rep2))).item()
        return relv


class Pyramid:
    def __init__(self):
        pass

    def get_sim(self, sent1, sent2):
        return

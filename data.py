from fastNLP import DataSet, Instance, DataSetIter, RandomSampler, SequentialSampler
from transformers import BertTokenizer
import numpy as np
import json


class Dataset(object):
    def __init__(self, territory, template, batch_size, max_length):
        self.territory = territory
        self.template = template
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.batch_size = batch_size
        self.max_length = max_length

        with open(f'./data/{self.territory}/train.json', 'r', encoding='utf-8') as f:
            self.n_category, self.category, self.train_data = json.load(f)
        with open(f'./data/{self.territory}/test.json', 'r', encoding='utf-8') as f:
            _, _, self.test_data = json.load(f)

    def get_classification_data(self, data):
        data_set = DataSet()
        polarity = ['positive', 'neutral', 'negative']
        for item in data:
            labels = np.zeros((3, self.n_category))
            target1 = np.zeros(3)
            for c in item[1:]:
                for i in range(len(polarity)):
                    if c['polarity'] == polarity[i]:
                        labels[i][self.category.index(c['category'])] = 1
                        target1[i] = 1
            for j in range(len(polarity)):
                sentence = polarity[j] + '[SEP]' + item[0] + self.template
                length = len(self.tokenizer.tokenize(item[0] + self.template))
                data_set.append(Instance(raw_words=sentence, labels=labels[j], target1=target1[j], length=length))
        data_set.apply_more(
            lambda ins: dict(self.tokenizer.encode_plus(ins['raw_words'], add_special_tokens=True, return_tensors='pt')))

        data_set.set_target('labels', 'target1')
        data_set.set_input('input_ids', 'attention_mask', 'length')
        return data_set

    def process(self):
        train_set = self.get_classification_data(self.train_data)
        test_set = self.get_classification_data(self.test_data)
        train_batch = DataSetIter(batch_size=self.batch_size, dataset=train_set, sampler=RandomSampler())
        test_batch = DataSetIter(batch_size=self.batch_size, dataset=test_set, sampler=SequentialSampler())
        return train_batch, test_batch


if __name__ == '__main__':
    templates = 'The sentence talks about the [MASK]'
    data_set = Dataset('restaurants15', templates, 8, 108)
    train, _ = data_set.process()
    for x, y in train:
        print(x)

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

        with open(f'../data/{self.territory}/train.json', 'r', encoding='utf-8') as f:
            self.n_category, self.category, self.train_data = json.load(f)
        with open(f'../data/{self.territory}/test.json', 'r', encoding='utf-8') as f:
            _, _, self.test_data = json.load(f)

    def get_classification_data(self, data):
        data_set = DataSet()
        polarity = ['positive', 'neutral', 'negative']
        for item in data:
            category_label = np.zeros(self.n_category)
            sentiment_label = np.zeros((self.n_category, 3))
            for c in item[1:]:
                for i in range(self.n_category):
                    if c['category'] == self.category[i]:
                        category_label[i] = 1
                        sentiment_label[i][polarity.index(c['polarity'])] = 1
            for j in range(self.n_category):
                sentence = self.category[j] + '[SEP]' + item[0] + self.template
                length = len(item[0])
                data_set.append(Instance(raw_words=sentence, category_label=category_label[j], sentiment_label=sentiment_label[j], length=length))
        data_set.apply_more(
            lambda ins: dict(self.tokenizer.encode_plus(ins['raw_words'], add_special_tokens=True, return_tensors='pt')))

        data_set.set_target('category_label', 'sentiment_label')
        data_set.set_input('input_ids', 'attention_mask', 'length')
        return data_set

    def process(self):
        train_set = self.get_classification_data(self.train_data)
        print(len(train_set))
        test_set = self.get_classification_data(self.test_data)
        train_batch = DataSetIter(batch_size=self.batch_size, dataset=train_set, sampler=RandomSampler())
        test_batch = DataSetIter(batch_size=self.batch_size, dataset=test_set, sampler=SequentialSampler())
        return train_batch, test_batch


if __name__ == '__main__':
    templates = 'The sentence talks about the [MASK]'
    data_set = Dataset('restaurants15', templates, 8, 108)
    train, _ = data_set.process()


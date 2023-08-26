from data import Dataset
from transformers import BertForMaskedLM
from torch import nn
import numpy as np
import fitlog
import torch
import json
import argparse
import random


def attention(logits, mask):
    score = nn.functional.softmax(torch.matmul(logits, mask), dim=0)
    score_logit = torch.mul(logits, score.unsqueeze(1))
    att_logit = torch.sum(score_logit, dim=0)
    out = torch.cat((att_logit, mask))
    return out.unsqueeze(0)


class ClassModel(torch.nn.Module):
    def __init__(self, dropout, label_words, device):
        super(ClassModel, self).__init__()
        self.label_words = label_words
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.activation = nn.Tanh()
        self.device = device

        self.senti_linear = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(768, 2))

        self.dense = nn.Sequential(nn.Linear(768 * 2, 768), self.activation)

        self.prob_linear = nn.Sequential(nn.Linear(768, 30522), self.activation)
        self.prob_linear[0].weight = self.bert.cls.predictions.decoder.weight
        self.prob_linear[0].bias = self.bert.cls.predictions.decoder.bias

        self.w = nn.ModuleList(nn.Linear(len(label), 2, bias=False) for label in label_words)

    def forward(self, batches):
        # output:[batch_size, max_len, n_category]
        input_ids = batches['input_ids'].squeeze(1).to(self.device)
        attention_mask = batches['attention_mask'].squeeze(1).to(self.device)
        length = batches['length'].to(self.device)

        # bert_out = self.bert(input_ids, attention_mask).logits
        bert_out = self.bert(input_ids, attention_mask, output_hidden_states=True).hidden_states[-1]
        pooler = bert_out[:, 0]

        senti_out = self.senti_linear(pooler)  # senti_out: [batch_size, 2]

        a, b = input_ids.shape
        mask_logits = torch.cat([bert_out[i, j:j+1] for i in range(a) for j in range(b) if input_ids[i][j] == 103])

        attention_logits = torch.FloatTensor().to(self.device)
        for i in range(a):
            sentence_logits = bert_out[i, 3:3+length[i]]
            attention_out = attention(sentence_logits, mask_logits[i])
            attention_logits = torch.cat((attention_logits, attention_out), dim=0)

        attention_logits = self.dense(attention_logits)
        attention_logits = self.prob_linear(attention_logits)

        out = torch.FloatTensor().to(self.device)  # out: [batch_size, 2, n_cate]

        for i in range(len(self.label_words)):
            label_prob = torch.cat([attention_logits[:, ids:ids+1] for ids in self.label_words[i]], dim=1)
            prob = self.w[i](label_prob).unsqueeze(2)
            out = torch.cat((out, prob), dim=2)

        return senti_out, out

    def test(self, data_loader, ep):
        senti_pre_id = torch.DoubleTensor().to(self.device)
        category_pre_id = torch.DoubleTensor().to(self.device)
        category_true_id = torch.DoubleTensor().to(self.device)
        for batches_x, batches_y in data_loader:
            senti_pred, pred = self.forward(batches_x)

            senti_out = nn.functional.log_softmax(senti_pred, dim=1)
            senti_index = torch.argmax(senti_out, dim=1)

            pred_out = nn.functional.log_softmax(pred, dim=1)
            index = torch.argmax(pred_out, dim=1)
            '''
            pred = torch.sigmoid(pred)
            index = torch.where(pred > 1 / data_set.n_category, torch.ones_like(pred, dtype=torch.float64),
                                torch.zeros_like(pred, dtype=torch.float64))
            '''
            senti_pre_id = torch.cat((senti_pre_id, senti_index))
            category_pre_id = torch.cat((category_pre_id, index))
            category_true_id = torch.cat((category_true_id, batches_y['labels'].to(self.device)))
        f1 = evaluate(category_pre_id, category_true_id, ep)
        return f1


def evaluate(predict, true, ep):
    d0, d1 = true.shape
    s, g, tem = 0, 0, 0
    for k in range(d0):
        for j in range(d1):
            if true[k][j] != 0:
                g += 1
            # if senti_pre[k] == 1:
            if predict[k][j] != 0:
                s += 1
                if predict[k][j] == true[k][j]:
                    tem += 1

    p1 = 0 if tem == 0 else tem / s
    r1 = 0 if tem == 0 else tem / g
    f1 = 0 if p1 == 0 and r1 == 0 else 2 * p1 * r1 / (p1 + r1)

    print(f"epoch:{ep + 1}, predict:{s},{'':<5} true:{g},{'':<5} right:{tem},{'':<5} p:{p1:.4f},{'':<4} r:{r1:.4f},{'':<4} f1:{f1:.4f}")

    return f1


def train(opt):
    fitlog.set_log_dir(f'./logs/{opt.dataset}/')
    fitlog.add_hyper_in_file(__file__)

    data_set = Dataset(opt.dataset, opt.template, opt.batch_size, opt.max_lens)
    train_loader, test_loader = data_set.process()

    vocabulary = data_set.tokenizer.vocab
    with open(f'./label_words/{opt.dataset}_label_words.json', 'r', encoding='utf-8') as f:
        label_word = json.load(f)
    label_words = [[vocabulary.get(l) for l in label if vocabulary.get(l) is not None] for label in label_word.values()]

    device = torch.device('cuda')

    model = ClassModel(opt.dropout, label_words, device).to(device)

    param = [{'params': [param for name, param in model.named_parameters() if name.split('.')[0] != 'w'], 'lr': opt.learning_rate1},
             {'params': [param for name, param in model.named_parameters() if name.split('.')[0] == 'w'], 'lr': opt.learning_rate2}]
    optimizer = opt.optimizer(param, weight_decay=opt.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True,
                                                           threshold=0.001, threshold_mode='abs', cooldown=0, min_lr=0,
                                                           eps=1e-8)
    loss = opt.loss(reduction='sum').to(device)

    f1_list = []
    for i in range(opt.epoch):
        loss_1, loss_2, loss_3, loss_sum = [], [], [], []
        for batch_x, batch_y in train_loader:
            model.train()
            optimizer.zero_grad()
            senti_pres, pres = model.forward(batch_x)
            ls1 = loss(senti_pres, batch_y['target1'].long().to(device))
            ls2 = loss(pres, batch_y['labels'].long().to(device))
            ls_sum = ls1 + ls2
            ls_sum.backward()
            optimizer.step()
            loss_1.append(ls1)
            loss_2.append(ls2)
            loss_sum.append(ls_sum)
        fitlog.add_loss(loss_1[-1], name='Loss_1', step=i)
        fitlog.add_loss(loss_2[-1], name='Loss_2', step=i)
        fitlog.add_loss(loss_sum[-1], name='Loss_sum', step=i)
        model.eval()
        with torch.no_grad():
            f1 = model.test(test_loader, i)
            with open(opt.dataset+'_f1.txt', 'a', encoding='utf-8') as f:
                nstr = ' ' if i < 29 else '\n'
                f.write(str(f1) + nstr)
            f1_list.append(f1)
            if f1 > opt.best_f1[opt.dataset] and opt.save_params:
                opt.best_f1[opt.dataset] = f1
                print("the max f1:", f1)
                torch.save(model.state_dict(), opt.cls_params)
            scheduler.step(f1)
            f1_list.append(f1)
            print()

    print(max(f1_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='laptops15', type=str, help='restaurants15, restaurants16, laptops15, laptops16')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--learning_rate1', default=2e-5, type=float)
    parser.add_argument('--learning_rate2', default=2e-3, type=float)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--save_params', default=True, type=bool)
    parser.add_argument('--max_lens', default=108, type=int)
    parser.add_argument('--seed', default=None, type=int)
    opt = parser.parse_args()

    # template1:the sentence is about the [MASK]
    # template2:[MASK] is the category of the sentence
    # template3:it is about the [MASK] in the sentence
    opt.template = 'it is about the [MASK] in the sentence'
    opt.best_f1 = {'restaurants15': 0.6841, 'restaurants16': 0.7598, 'laptops15': 0.6288, 'laptops16': 0.5694}
    opt.cls_params = f'./save/{opt.dataset}_cls_params.pt'

    opt.optimizer = torch.optim.AdamW
    opt.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    opt.loss = nn.CrossEntropyLoss

    if opt.seed is not None:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train(opt)

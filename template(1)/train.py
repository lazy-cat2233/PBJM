from data import Dataset
from transformers import BertForMaskedLM
from torch import nn
import fitlog
import torch


templates = 'the sentence thinks it is [MASK]'

territory = 'laptops15'

best_f1 = {'restaurants15': 0.6841, 'restaurants16': 0.7598, 'laptops15': 0.6241, 'laptops16': 0.55}

cls_params = f'./save/{territory}_cls_params.pt'

fitlog.set_log_dir(f'../logs/{territory}/')
fitlog.add_hyper_in_file(__file__)

# ######hyper
batch_size = 8
epoch = 30
lr1 = 2e-5
lr2 = 2e-3
max_lens = 108
dropout = 0.1
weight_decay = 1e-5
load_para = False
save_para = False
# ######hyper

data_set = Dataset(territory, templates, batch_size, max_lens)
train_loader, test_loader = data_set.process()

vocabulary = data_set.tokenizer.vocab

label_word = {'positive': ['great', 'good', 'nice', 'right', 'excellent', 'fine', 'outstanding', 'positive'],
              'neutral': ['ok', 'indifferent', 'neutral', 'flat', 'insipid', 'ordinary'],
              'negative': ['terrible', 'bad', 'boring', 'poor', 'acute', 'negative', 'passive']}

label_words = [[vocabulary.get(l) for l in label if vocabulary.get(l) is not None] for label in label_word.values()]


def attention(logits, mask):
    score = nn.functional.softmax(torch.matmul(logits, mask), dim=0)
    score_logit = torch.mul(logits, score.unsqueeze(1))
    att_logit = torch.sum(score_logit, dim=0)
    out = torch.cat((att_logit, mask))
    return out.unsqueeze(0)


class ClassModel(torch.nn.Module):
    def __init__(self):
        super(ClassModel, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.activation = nn.Tanh()

        self.senti_linear = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(768, 2))

        self.dense = nn.Sequential(nn.Linear(768 * 2, 768), self.activation)

        self.prob_linear = nn.Sequential(nn.Linear(768, 30522), self.activation)
        self.prob_linear[0].weight = self.bert.cls.predictions.decoder.weight
        self.prob_linear[0].bias = self.bert.cls.predictions.decoder.bias

        self.w = nn.ModuleList(nn.Linear(len(label), 2, bias=False) for label in label_words)

    def forward(self, batches):
        # output:[batch_size, max_len, n_category]
        input_ids = batches['input_ids'].squeeze(1).to(device)
        attention_mask = batches['attention_mask'].squeeze(1).to(device)
        length = batches['length'].to(device)

        # bert_out = self.bert(input_ids, attention_mask).logits
        bert_out = self.bert(input_ids, attention_mask, output_hidden_states=True).hidden_states[-1]
        pooler = bert_out[:, 0]

        category_out = self.senti_linear(pooler)  # senti_out: [batch_size, 2]

        a, b = input_ids.shape
        mask_logits = torch.cat([bert_out[i, j:j+1] for i in range(a) for j in range(b) if input_ids[i][j] == 103])

        attention_logits = torch.FloatTensor().to(device)
        for i in range(a):
            sentence_logits = bert_out[i, 3:3+length[i]]
            attention_out = attention(sentence_logits, mask_logits[i])
            attention_logits = torch.cat((attention_logits, attention_out), dim=0)

        attention_logits = self.dense(attention_logits)
        attention_logits = self.prob_linear(attention_logits)

        out = torch.FloatTensor().to(device)  # out: [batch_size, 2, n_cate]

        for i in range(len(label_words)):
            label_prob = torch.cat([attention_logits[:, ids:ids+1] for ids in label_words[i]], dim=1)
            prob = self.w[i](label_prob).unsqueeze(2)
            out = torch.cat((out, prob), dim=2)

        return category_out, out


device = torch.device('cuda')  # if torch.cuda.is_available() else 'cpu'

model = ClassModel().to(device)
if load_para:
    model.load_state_dict(torch.load(cls_params))


param = [{'params': [param for name, param in model.named_parameters() if name.split('.')[0] != 'w'], 'lr': lr1},
         {'params': [param for name, param in model.named_parameters() if name.split('.')[0] == 'w'], 'lr': lr2}]
optimizer = torch.optim.AdamW(param, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True,
                                                       threshold=0.001, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-8)


loss_sen = nn.CrossEntropyLoss(reduction='sum').to(device)


def test(data_loader, ep):
    cate_pre_id = torch.DoubleTensor().to(device)
    senti_pre_id = torch.DoubleTensor().to(device)
    senti_true_id = torch.DoubleTensor().to(device)
    for batches_x, batches_y in data_loader:
        cate_pred, pred = model.forward(batches_x)

        cate_out = nn.functional.log_softmax(cate_pred, dim=1)
        cate_index = torch.argmax(cate_out, dim=1)

        pred_out = nn.functional.log_softmax(pred, dim=1)
        index = torch.argmax(pred_out, dim=1)
        '''
        pred = torch.sigmoid(pred)
        index = torch.where(pred > 1 / data_set.n_category, torch.ones_like(pred, dtype=torch.float64),
                            torch.zeros_like(pred, dtype=torch.float64))
        '''
        cate_pre_id = torch.cat((cate_pre_id, cate_index))
        senti_pre_id = torch.cat((senti_pre_id, index))
        senti_true_id = torch.cat((senti_true_id, batches_y['sentiment_label'].to(device)))
    f1 = evaluate(senti_pre_id, senti_true_id, ep, cate_pre_id)
    return f1


def evaluate(predict, true, ep, senti_pre):
    d0, d1 = true.shape
    s, g, tem = 0, 0, 0
    for k in range(d0):
        for j in range(d1):
            if true[k][j] != 0:
                g += 1
            if senti_pre[k] == 1:
                if predict[k][j] != 0:
                    s += 1
                    if predict[k][j] == true[k][j]:
                        tem += 1

    p1 = 0 if tem == 0 else tem / s
    r1 = 0 if tem == 0 else tem / g
    f1 = 0 if p1 == 0 and r1 == 0 else 2 * p1 * r1 / (p1 + r1)

    print(f"epoch:{ep + 1}, predict:{s},{'':<5} true:{g},{'':<5} right:{tem},{'':<5} p:{p1:.4f},{'':<4} r:{r1:.4f},{'':<4} f1:{f1:.4f}")

    return f1


f1_list = []
for i in range(epoch):
    loss_1, loss_2, loss_3, loss_sum = [], [], [], []
    for batch_x, batch_y in train_loader:
        model.train()
        optimizer.zero_grad()
        cate_pres, pres = model.forward(batch_x)
        ls1 = loss_sen(cate_pres, batch_y['category_label'].long().to(device))
        ls2 = loss_sen(pres, batch_y['sentiment_label'].long().to(device))
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
        f1 = test(test_loader, i)
        f1_list.append(f1)
        if f1 > best_f1[territory] and save_para:
            best_f1[territory] = f1
            print("the max f1:", f1)
            torch.save(model.state_dict(), cls_params)
        scheduler.step(f1)
        f1_list.append(f1)
        print()

print(max(f1_list))

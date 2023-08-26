import xml.etree.ElementTree as ET
import json

path_train = "D:/code/prompt-ABSA/dataset/original data/ABSA16_Laptops_Train_SB1_v2.xml"
path_test = 'D:/code/prompt-ABSA/dataset/original data/EN_LAPT_SB1_TEST_.xml'
terr = 'laptops16'


def get_path(territory, data_type):
    return f'./dataset/data/{territory}/{data_type}.json'


def judge(p):
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[j]['category'] == p[i]['category'] and p[j]['polarity'] != p[i]['polarity']:
                return False
    return True


def extract_data(path):
    tree = ET.parse(path)
    root = tree.getroot()
    data = []

    # 从xml文件中提取数据
    for review in root:
        for sentences in review:
            for sentence in sentences:
                piece = []
                for t in sentence.iter('text'):
                    piece.append(t.text)
                for o in sentence.iter('Opinion'):
                    d = {'category': o.attrib['category'], 'polarity': o.attrib['polarity']}
                    piece.append(d)
                # 所有沒有category分類的句子以及所有一個category卻多個情感的句子
                if len(piece) > 1 and judge(piece[1:]):
                    data.append(piece)

    n_category = 0
    category = []

    # 进行数据统计
    for e in data:
        for i in range(1, len(e)):
            c, s = e[i].values()
            if c not in category:
                n_category += 1
                category.append(c)

    all_data = [n_category, category, data]
    return all_data


train_data = extract_data(path_train)
test_data = extract_data(path_test)
# 将train中没有而test中有的category從test中刪除

over_list = [elem for elem in test_data[1] if elem not in train_data[1]]
move_list = [elem for cate in over_list for elem in test_data[2] for e in elem[1:] if e['category'] == cate]
test_data[2] = [elem for elem in test_data[2] if elem not in move_list]
test_data[1] = [elem for elem in test_data[1] if elem not in over_list]
test_data[0] = len(test_data[1])
print(over_list)

with open(get_path(terr, 'train'), 'w', encoding='utf-8') as f:
    json.dump(train_data, f)
with open(get_path(terr, 'test'), 'w', encoding='utf-8') as f:
    json.dump(test_data, f)

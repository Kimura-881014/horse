import pandas as pd
import time
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from torch import torch, nn, optim
from torch.utils.data import TensorDataset, DataLoader
from imblearn.under_sampling import RandomUnderSampler
import requests
from bs4 import BeautifulSoup
import re
from sklearn.metrics import roc_auc_score

# スクレイピング
year = str(2019)
try:
    pre_race_results = pd.read_pickle(year + '-results.pickle')
    no_results = pd.read_pickle(year + '-no.pickle')
except FileNotFoundError:
    url = 'https://db.netkeiba.com/race/' + year + '01010101'
    race_id = year + '01010101'
    race_results = {}
    race_infos = {}
    url = 'https://db.netkeiba.com/race/' + race_id
    html = requests.get(url)
    html.encoding = 'EUC-JP'
    soup = BeautifulSoup(html.text, 'html.parser')
    texts_table = str(soup.find('table', attrs = {'class':'race_table_01 nk_tb_common'}))
    race_results[race_id] = pd.read_html(texts_table)[0]
    texts = soup.find('div', attrs={'class':'data_intro'}).find_all('p')[0].text + \
        soup.find('div', attrs={'class':'data_intro'}).find_all('p')[1].text
    info = re.findall(r'\w+',texts)
    info_dict = {}
    for text in info:
        if text in ['芝','ダート']:
            info_dict['race_type'] = text
        if '障' in text:
            info_dict['race_type'] = '障害'
        if 'm' in text:
            info_dict['course_len'] = re.findall(r'\d+',text)[0]
        if text in ['良','稍重','重','不良']:
            info_dict['ground_state'] = text
        if text in['曇','晴','雨','小雨','小雪','雪']:
            info_dict['weather'] = text
        if '年' in text:
            info_dict['data'] = text
    race_infos[race_id] = info_dict
    test = race_results

    for key in test:
        test[key].index = [key] * len(test[key])
    info_df = pd.DataFrame(race_infos).T
    results_new_1 = pd.concat([test[key] for key in test], sort = False)

    pre_race_results = results_new_1.merge(info_df,left_index=True,right_index=True,how ='inner')

    no_ex = [year + '01010702']
    no_results = pd.DataFrame()
    no_results['id'] = no_ex

pre_race_id_list = list(dict.fromkeys(pre_race_results.index.tolist()))

no_id_list = list(dict.fromkeys(no_results['id'].tolist()))


all_race_id_list = []
for place in range(1, 11):
    for kai in range(1, 6):
        for day in range(1,13):
            for r in range(1,13):
                race_id = year + str(place).zfill(2) + str(kai).zfill(2) + str(day).zfill(2) + str(r).zfill(2)
                all_race_id_list.append(race_id)

for a in pre_race_id_list:
    for p in all_race_id_list:
        if a == p:
            all_race_id_list.remove(p)

for an in no_id_list:
    for n in all_race_id_list:
        if an == n:
            all_race_id_list.remove(n)

race_results = {}
race_infos = {}
no_data_li =[]
for race_id in tqdm(all_race_id_list):
    try:
        time.sleep(1)
        url = 'https://db.netkeiba.com/race/' + race_id
        html = requests.get(url)
        html.encoding = 'EUC-JP'
        soup = BeautifulSoup(html.text, 'html.parser')
        texts_table = str(soup.find('table', attrs = {'class':'race_table_01 nk_tb_common'}))
        race_results[race_id] = pd.read_html(texts_table)[0]
        texts = soup.find('div', attrs={'class':'data_intro'}).find_all('p')[0].text + \
            soup.find('div', attrs={'class':'data_intro'}).find_all('p')[1].text
        info = re.findall(r'\w+',texts)
        info_dict = {}
        for text in info:
            if text in ['芝','ダート']:
                info_dict['race_type'] = text
            if '障' in text:
                info_dict['race_type'] = '障害'
            if 'm' in text:
                info_dict['course_len'] = re.findall(r'\d+',text)[0]
            if text in ['良','稍重','重','不良']:
                info_dict['ground_state'] = text
            if text in['曇','晴','雨','小雨','小雪','雪']:
                info_dict['weather'] = text
            if '年' in text:
                info_dict['data'] = text
        race_infos[race_id] = info_dict
    except IndexError:
        no_data_li.append(race_id)
        continue
    except:
        break

test = race_results
new_no_df = pd.DataFrame()
new_no_df['id'] = no_data_li
try:
    no_df = pd.concat([no_results,new_no_df])
except ValueError:
    pass
no_df.drop_duplicates(inplace=True)
no_df.to_pickle(year + '-no.pickle')

for key in test:
    test[key].index = [key] * len(test[key])
try:
    info_df = pd.DataFrame(race_infos).T
    results_new_1 = pd.concat([test[key] for key in test], sort = False)
    results_new = results_new_1.merge(info_df,left_index = True,right_index=True,how = 'inner')
    try:
        results = pd.concat([pre_race_results,results_new])
    except ValueError:
        pass
    # results.drop_duplicates(inplace=True)
    results.to_pickle(year + '-results.pickle')
except ValueError:
    pass

# 前処理
results = pd.read_pickle(year + '-results.pickle')

results = results[~(results['着順'].astype(str).str.contains('\D'))]
results['着順'] = results['着順'].astype(int)
results['性'] = results['性齢'].map(lambda x:str(x)[0])
results['年齢'] = results['性齢'].map(lambda x:str(x)[1:]).astype(int)
results['体重'] = results['馬体重'].str.split('(', expand = True)[0].astype(int)
results['体重変化'] = results['馬体重'].str.split('(', expand = True)[1].str[:-1].astype(int)
results['単勝'] = results['単勝'].astype(float)
results['course_len'] = results['course_len'].astype(int)
results['data'] = pd.to_datetime(results['data'],format='%Y年%m月%d日')

clip_rank = lambda x:0 if x < 4 else 1
results['rank'] = results['着順'].map(clip_rank)
results.drop(['タイム','着差','調教師','性齢','馬体重','着順','馬名'],axis = 1, inplace = True)
results_d = pd.get_dummies(results)

sorted_id_list = results_d.sort_values('data').index.unique()
train_id_list = sorted_id_list[:round(len(sorted_id_list) * 0.7)]
test_id_list = sorted_id_list[round(len(sorted_id_list) * 0.7):]


train = results_d.loc[train_id_list]
test = results_d.loc[test_id_list]

rank_0 = train['rank'].value_counts()[0]
rank_1 = train['rank'].value_counts()[1]
# rank_3 = train['rank'].value_counts()[3]

min_rank = min([rank_0, rank_1])
rus = RandomUnderSampler(sampling_strategy={0:min_rank,1:min_rank},random_state=14)

X_train = train.drop(['data','rank'], axis=1)
y_train = train['rank']
X_test = test.drop(['data','rank'],axis=1)
y_test = test['rank']

X_train_rus,y_train_rus = rus.fit_sample(X_train,y_train)

# 学習
model = LogisticRegression()
model.fit(X_train_rus, y_train_rus)
print('LRのtrain精度:' + str(model.score(X_train,y_train)) + ' ' + 'LRのtest精度:' + str(model.score(X_test,y_test)))

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train_rus,y_train_rus)
print('RFのtrain精度:' + str(clf.score(X_train,y_train)) + ' ' + 'RFのtest精度:' + str(clf.score(X_test,y_test)))

params = {
    'min_samples_split':200,
    'max_depth':None,
    # 'n_estimators':None,
    'criterion':'entropy',
    'random_state':200,
    'class_weight':'balanced'
}

rf = RandomForestClassifier(**params)
rf.fit(X_train_rus,y_train_rus)
print('RF2のtrain精度:' + str(rf.score(X_train,y_train)) + ' ' + 'RF2のtest精度:' + str(rf.score(X_test,y_test)))

lgb_clf = lgb.LGBMClassifier()
lgb_clf.fit(X_train_rus.values,y_train_rus.values)
print('LGBのtrain精度:' + str(lgb_clf.score(X_train,y_train)) + ' ' + 'LGBのtest精度:' + str(lgb_clf.score(X_test,y_test)))

params = {
    'num_leaves':2,
    'n_estimators':10,
    'class_weight':'balanced',
    'random_state':100
}

lgb_clf2 = lgb.LGBMClassifier(**params)
lgb_clf2.fit(X_train_rus.values,y_train_rus.values)
print('LGB2のtrain精度:' + str(lgb_clf2.score(X_train,y_train)) + ' ' + 'LGB2のtest精度:' + str(lgb_clf2.score(X_test,y_test)))

# ニューラルネットワーク
X_train = torch.Tensor(X_train_rus.values)
t_train = torch.Tensor(y_train_rus.values)
X_test = torch.Tensor(X_test.values)
t_test = torch.Tensor(y_test.values)

t_train = t_train.reshape(-1, 1)
t_test = t_test.reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(219,128),
    nn.Sigmoid(),
    nn.Linear(128,64),
    nn.Sigmoid(),
    nn.Linear(64,1),
    nn.Sigmoid()
)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

model.train()
for epoch in tqdm(range(100)):
    optimizer.zero_grad()
    y_train = model(X_train)
    loss = loss_fn(y_train, t_train)
    
    loss.backward()
    optimizer.step()

y_train = model(X_train)
y_test = model(X_test)

roc_auc_score(t_train.detach().numpy(), y_train.detach().numpy())
print('nnのtrain精度:' + str(roc_auc_score(t_train.detach().numpy(), y_train.detach().numpy())) + ' ' \
    + 'nnのtest精度:' + str(roc_auc_score(t_test.detach().numpy(), y_test.detach().numpy())))

model = nn.Sequential(
    nn.Linear(219,128),
    nn.BatchNorm1d(128),
    nn.Sigmoid(),
    nn.Linear(128,64),
    nn.BatchNorm1d(64),
    nn.Sigmoid(),
    nn.Linear(64,1),
    nn.Sigmoid()
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

dataset = TensorDataset(X_train, t_train)
loader = DataLoader(dataset, batch_size = 128, shuffle = True)

model.train()
for epoch in tqdm(range(10)):
    for X, t in loader:
        optimizer.zero_grad()
        y = model(X)
        loss = loss_fn(y, t)
        loss.backward()
        optimizer.step()
        
model.eval()
y_train = model(X_train)
y_test = model(X_test)
print('nn2のtrain精度:' + str(roc_auc_score(t_train.detach().numpy(), y_train.detach().numpy())) + ' ' \
    + 'nn2のtest精度:' + str(roc_auc_score(t_test.detach().numpy(), y_test.detach().numpy())))
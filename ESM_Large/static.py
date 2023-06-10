
train_news = []
dev_news = []
test_news = []
with open('../MIND-500k/train/news.tsv', encoding='utf-8') as ftr:
    for line in ftr:
        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
        train_news.append(news_ID)

with open('../MIND-500k/dev/news.tsv', encoding='utf-8') as ftr:
    for line in ftr:
        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
        dev_news.append(news_ID)

with open('../MIND-500k/test/news.tsv', encoding='utf-8') as ftr:
    for line in ftr:
        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
        test_news.append(news_ID)

print(f'train_news_length:{len(train_news)},{len(set(train_news))}')
print(f'dev_news_length:{len(dev_news)},{len(set(dev_news))}')
print(f'test_news_length:{len(test_news)},{len(set(test_news))}')

total_news = set(train_news+dev_news+test_news)
print(f'total_news_length:{len(total_news)}')




train_user = []
dev_user = []
test_user = []
train_impressions = 0
dev_impressions = 0
test_impressions = 0
train_pos = []

train_click_impressions = []
train_non_click_impressions = []
dev_click_impressions = []
dev_non_click_impressions = []
test_click_impressions = []
test_non_click_impressions = []

with open('../MIND-500k/train/behaviors.tsv', encoding='utf-8') as train_behaviors_f:
    for line in train_behaviors_f:
        impression_ID, user_ID, time, history, impressions = line.split('\t')
        train_user.append(user_ID)
        train_impressions += 1
        for impression in impressions.strip().split(' '):
            if impression[-2:] == '-1':
                train_click_impressions.append(impression[:-2])
            else:
                train_non_click_impressions.append(impression[:-2])


with open('../MIND-500k/dev/behaviors.tsv', encoding='utf-8') as dev_behaviors_f:
    for line in dev_behaviors_f:
        impression_ID, user_ID, time, history, impressions = line.split('\t')
        dev_user.append(user_ID)
        dev_impressions += 1
        for impression in impressions.strip().split(' '):
            if impression[-2:] == '-1':
                dev_click_impressions.append(impression[:-2])
            else:
                dev_non_click_impressions.append(impression[:-2])

with open('../MIND-500k/test/behaviors.tsv', encoding='utf-8') as test_behaviors_f:
    for line in test_behaviors_f:
        impression_ID, user_ID, time, history, impressions = line.split('\t')
        test_user.append(user_ID)
        test_impressions+=1
        for impression in impressions.strip().split(' '):
            if impression[-2:] == '-1':
                test_click_impressions.append(impression[:-2])
            else:
                test_non_click_impressions.append(impression[:-2])

print('#######################')
print(f'train_userlength:{len(train_user)},{len(set(train_user))}')
print(f'dev_user_length:{len(dev_user)},{len(set(dev_user))}')
print(f'test_user_length:{len(test_user)},{len(set(test_user))}')
total_user = set(train_user+dev_user+test_user)
print(f'total_user_length:{len(total_user)},{len(set(total_user))}')
print(f'train_impressions:{train_impressions}')
print(f'dev_impressions:{dev_impressions}')
print(f'test_impressions:{test_impressions}')
print('####################')
print('click')
print(f'train_click:{len(train_click_impressions)}')
print(f'train_no_click:{len(train_non_click_impressions)}')
print(f'dev_click:{len(dev_click_impressions)}')
print(f'dev_no_click:{len(dev_non_click_impressions)}')
print(f'test_click:{len(test_click_impressions)}')
print(f'test_no_click:{len(test_non_click_impressions)}')

print('######################large#######################')

train_news = []
dev_news = []
test_news = []
with open('../MIND-large/train/news.tsv', encoding='utf-8') as ftr:
    for line in ftr:
        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
        train_news.append(news_ID)

with open('../MIND-large/dev/news.tsv', encoding='utf-8') as ftr:
    for line in ftr:
        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
        dev_news.append(news_ID)

with open('../MIND-large/test/news.tsv', encoding='utf-8') as ftr:
    for line in ftr:
        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
        test_news.append(news_ID)

print(f'train_news_length:{len(train_news)},{len(set(train_news))}')
print(f'dev_news_length:{len(dev_news)},{len(set(dev_news))}')
print(f'test_news_length:{len(test_news)},{len(set(test_news))}')

total_news = set(train_news+dev_news+test_news)
print(f'total_news_length:{len(total_news)}')




train_user = []
dev_user = []
test_user = []
train_impressions = 0
dev_impressions = 0
test_impressions = 0
train_pos = []

train_click_impressions = []
train_non_click_impressions = []
dev_click_impressions = []
dev_non_click_impressions = []
test_click_impressions = []
test_non_click_impressions = []

with open('../MIND-large/train/behaviors.tsv', encoding='utf-8') as train_behaviors_f:
    for line in train_behaviors_f:
        impression_ID, user_ID, time, history, impressions = line.split('\t')
        train_user.append(user_ID)
        train_impressions += 1
        for impression in impressions.strip().split(' '):
            if impression[-2:] == '-1':
                train_click_impressions.append(impression[:-2])
            else:
                train_non_click_impressions.append(impression[:-2])



with open('../MIND-large/dev/behaviors.tsv', encoding='utf-8') as dev_behaviors_f:
    for line in dev_behaviors_f:
        impression_ID, user_ID, time, history, impressions = line.split('\t')
        dev_user.append(user_ID)
        dev_impressions += 1
        for impression in impressions.strip().split(' '):
            if impression[-2:] == '-1':
                dev_click_impressions.append(impression[:-2])
            else:
                dev_non_click_impressions.append(impression[:-2])

with open('../MIND-large/test/behaviors.tsv', encoding='utf-8') as test_behaviors_f:
    for line in test_behaviors_f:
        impression_ID, user_ID, time, history, impressions = line.split('\t')
        test_user.append(user_ID)
        test_impressions+=1
        for impression in impressions.strip().split(' '):
            if impression[-2:] == '-1':
                test_click_impressions.append(impression[:-2])
            else:
                test_non_click_impressions.append(impression[:-2])

print('#######################')
print(f'train_userlength:{len(train_user)},{len(set(train_user))}')
print(f'dev_user_length:{len(dev_user)},{len(set(dev_user))}')
print(f'test_user_length:{len(test_user)},{len(set(test_user))}')
total_user = set(train_user+dev_user+test_user)
print(f'total_user_length:{len(total_user)},{len(set(total_user))}')
print(f'train_impressions:{train_impressions}')
print(f'dev_impressions:{dev_impressions}')
print(f'test_impressions:{test_impressions}')
print('####################')
print('click')
print(f'train_click:{len(train_click_impressions)}')
print(f'train_no_click:{len(train_non_click_impressions)}')
print(f'dev_click:{len(dev_click_impressions)}')
print(f'dev_no_click:{len(dev_non_click_impressions)}')





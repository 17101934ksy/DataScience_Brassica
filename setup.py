from modular import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

with open("./pickle/meta_df_t.pickle","rb") as fr:
    meta_df_t = pickle.load(fr)

with open("./pickle/meta_test_df_te","rb") as fr:
    meta_df_te = pickle.load(fr)

meta_df_t = meta_df_t[['0시 EC관측치', '12시 EC관측치', '0시 온도차','0시 습도차', '12시 온도차', '12시 습도차', 'Case', 'Name']]
meta_df_te = meta_df_te[['0시 EC관측치', '12시 EC관측치', '0시 온도차','0시 습도차', '12시 온도차', '12시 습도차', 'Name']]

# 기본 경로
train_path = './train'
test_path = './test'

# all_img_path, all_label = get_train_data(train_path)
# test_img_path = get_test_data(test_path)

with open('./pickle/all_img_path.pickle', 'rb') as f:
    all_img_path = pickle.load(f)

with open('./pickle/all_label.pickle', 'rb') as f:
    all_label = pickle.load(f)

with open('./pickle/test_img_path.pickle', 'rb') as f:
    test_img_path = pickle.load(f)

train_len = int(len(all_img_path)*0.8)

train_img_path = all_img_path[:train_len]
train_meta_ = meta_df_t[:train_len]
train_label = all_label[:train_len]

vali_img_path = all_img_path[train_len:]
vali_meta_ = meta_df_t[train_len:]
vali_label = all_label[train_len:]

with open('./pickle/train_img_mask_0_64.pickle', 'rb') as f:
    train_img_mask_ = pickle.load(f)

with open('./pickle/test_img_mask_0_64.pickle', 'rb') as f:
    test_img_mask_ = pickle.load(f)

train_np_sum = []
test_np_sum = []

for s in train_img_mask_:
    train_np_sum.append(np.sum(s))

for s in test_img_mask_:
    test_np_sum.append(np.sum(s))


    # 데이터 순서 체크
meta_df_t['CaseName'] = meta_df_t['Case'] + '_' + meta_df_t['Name']
test_seq = True
for i, j in zip(range(len(all_img_path)), meta_df_t['CaseName']):
    if all_img_path[i][-9:-4] != j:
        print(j)
        test_seq = False
    print(f'Sequence Checking Train: {test_seq}')


test_seq = True
for i, j in zip(range(len(test_img_path)), meta_df_te['Name']):
    if test_img_path[i][-7:-4] != j:
        print(j)
        test_seq = False
    print(f'Sequence Checking Test: {test_seq}')

meta_df_t = meta_df_t.drop(['Case', 'Name', 'CaseName'], axis=1)

meta_df_te = meta_df_te.drop(['Name'], axis=1)

# 이상치 제거
idx = -1
for i, p in enumerate(all_img_path):
    if p[-9:-4] == '45_17':
        idx = i

meta_df_t = meta_df_t.drop(idx, axis=0)
train_np_sum.pop(idx)

meta_df_t['행렬 합'] =  train_np_sum
meta_df_te['행렬 합'] =  test_np_sum

meta_df_t = meta_df_t.to_numpy()
meta_df_te = meta_df_te.to_numpy()
print(f'image_path_remove: {all_img_path.pop(idx)}')
print(f'image_mask_remove:{train_img_mask_.pop(idx)}')
print(f'label_remove:{all_label.pop(idx)}')
print(f'meta_df_t.shape: {meta_df_t.shape}')
print(f'meta_df_te.shape: {meta_df_te.shape}')

scaler = StandardScaler()
meta_df_t = scaler.fit_transform(meta_df_t) 
meta_df_te = scaler.transform(meta_df_te)

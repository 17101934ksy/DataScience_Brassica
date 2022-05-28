import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from moduler import *

def get_train_data(data_dir):
    img_path_list = []
    label_list = []
    meta_list = pd.DataFrame()

    for case_name in os.listdir(data_dir):
        current_path = os.path.join(data_dir, case_name)
        if os.path.isdir(current_path):
            # get image path
            img_path_list.extend(sorted((glob(os.path.join(current_path, 'image', '*.jpg')))))
            img_path_list.extend(sorted((glob(os.path.join(current_path, 'image', '*.png')))))
            
            # get label
            label_df = pd.read_csv(current_path+'/label.csv')
            label_list.extend(label_df['leaf_weight'])
              
    return img_path_list, label_list

def get_test_data(data_dir):
    # get image path
    img_path_list = glob(os.path.join(data_dir, 'image', '*.jpg'))
    img_path_list.extend(glob(os.path.join(data_dir, 'image', '*.png')))
    img_path_list.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
    return img_path_list

def transform_train_resize(img_path_list, img_size=(CFG['IMG_SIZE'], CFG['IMG_SIZE'])):
  """
  이미지의 여백에 해당하는 특정 지점의 배경색 평균 계산
  img_path_list : 이미지의 경로 저장 리스트
  return  사이즈 조정된 이미지 리스트, 추출된 특정지점의 배경 벡터값 (3, )
  """
  w, h = img_size[0], img_size[1]

  img_vector = []
  resized_img_ = []
  for idx, case in enumerate(img_path_list):
    if not(idx % 30):
      print(f'{idx} 진행 중')
    
    img = cv2.resize(cv2.imread(case), dsize=(w, h), interpolation=cv2.INTER_AREA)
    resized_img_.append(img)

    img_vector.append(img[w-1:, h-1:, :])
  img_vector = np.array(img_vector)
  img_vector = np.mean(img_vector, axis=0).reshape(-1)

  return resized_img_, img_vector


def transform_test_resize(img_path_list, img_size=(CFG['IMG_SIZE'], CFG['IMG_SIZE'])):
  w, h = img_size[0], img_size[1]
  resized_img_ = []
  for idx, case in enumerate(img_path_list):
    if not(idx % 30):
      print(f'{idx} 진행 중')
      
    img = cv2.resize(cv2.imread(case), dsize=(w, h), interpolation=cv2.INTER_AREA)
    resized_img_.append(img)

  return resized_img_


def transform_background(image_list, extracted_mean_vector=(243.09112333, 219.92615868, 223.38177533)):
  """
  특정 지점의 배경색을 모두 평균과 같도록 이미지에 numpy 연산을 시행하여 명암의 분산이 줄어들도록 이미지 전처리
  resized_img_: 사이즈 조정된 이미지
  extracted_mean_vector: 추출된 특정지점의 배경 벡터값 (3, )
  return 모든 이미지가 추출된 특정지점의 평균과 동일하도록 정규화된 이미지
  """
  w, h = image_list[0].shape[0], image_list[0].shape[1]
  v1, v2, v3 =  extracted_mean_vector
  new_image_ = []
  for image in image_list:
    c1, c2, c3 = image[w-1:, h-1:, :].reshape(-1)
    
    condition_1 = np.where((image[:, :, 0] - (c1 - v1)) > 0 , (image[:, :, 0] - (c1 - v1)), 0)
    condition_2 = np.where((image[:, :, 1] - (c2 - v2)) > 0 , (image[:, :, 1] - (c2 - v2)), 0)
    condition_3 = np.where((image[:, :, 2] - (c3 - v3)) > 0 , (image[:, :, 2] - (c3 - v3)), 0)

    new_c1 = np.where(condition_1 < 255 , condition_1, 255)[..., np.newaxis] / 255.
    new_c2 = np.where(condition_2 < 255 , condition_2, 255)[..., np.newaxis] / 255.
    new_c3 = np.where(condition_3 < 255 , condition_3, 255)[..., np.newaxis] / 255.

    new_image = np.concatenate((new_c1, new_c2, new_c3), axis=2)
    new_image_.append(new_image)

  return new_image_


def transform_mask(image_list, threshold=0.64, alpha=0.7):
  """
  마스킹 처리
  image: 정규화된 이미지
  threshold: 마스킹 임계점
  alpha: 노이즈 처리(임계점으로 해결이 안될 경우 조정하는 변수)
  return 마스킹된 이미지 리스트
  """
  mask_ = []
  for image in image_list:
    ret, mask = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY) 
    mask = np.abs(mask - 1)
    mask = np.where(mask > alpha, mask, 0)
    a, b = np.where(((mask[:, :, 0] == 0) & (mask[:, :, 1] == 1) & (mask[:, :, 2] == 1)) |\
                    ((mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 1)) |\
                    ((mask[:, :, 0] == 0) & (mask[:, :, 1] == 1) & (mask[:, :, 2] == 0)))

    mask[a, b, :] = 0
    
    a, b = np.where(((mask[:, :, 0] == 1) & (mask[:, :, 1] == 1) & (mask[:, :, 2] == 0)) |\
                    ((mask[:, :, 0] == 1) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)) |\
                    ((mask[:, :, 0] == 1) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 1)))
    mask[a, b, :] = 1

    mask_.append(mask)
  
  return mask_

def plot_img(img_list, first_idx_and_last_idx=(0, 9)):
  plt.figure(figsize=(10, 10))
  for idx, img in enumerate(img_list[first_idx_and_last_idx[0]:first_idx_and_last_idx[1]]):
    plt.subplot(3, 3, idx+1)
    plt.imshow(img)
  plt.show()

def plot_mask_img(mask_, all_img_path_, label_):
  plt.figure(figsize=(100, 100))
  for idx, (mask, name, label) in enumerate(zip(mask_, all_img_path_, label_)):
    plt.subplot(30, 50, idx+1)
    plt.imshow(mask)
    plt.xticks()
    plt.yticks()
    plt.title(name[-9:-4]+'_'+str(f'{label:.1f}'))
  plt.show()

def plot_mask_test_img(mask_, all_img_path_):
  plt.figure(figsize=(100, 100))
  for idx, (mask, name) in enumerate(zip(mask_, all_img_path_)):
    plt.subplot(30, 50, idx+1)
    plt.imshow(mask)
    plt.xticks()
    plt.yticks()
    plt.title(name[-9:-4])
  plt.show()



def data_processing(df, train_set=True):
  
  path = './pickle'
  # 타입 변환

  df = df.reset_index(level=None, drop=True, inplace=False, col_level=0, col_fill='')
  df['시간'] = pd.to_datetime(df['시간'], format='%Y-%m-%d %H:%M:%S', errors='raise')
 
  df['m_d'] = df['시간'].dt.strftime("%m-%d")
  df['group'] = df['시간'].dt.strftime("%y-%m-%d-%H")


  # 광추정량 (공식)
  
  lay_white_nan_idx = np.where(df['백색광추정광량'].isnull())[0]
  lay_blue_nan_idx = np.where(df['청색광추정광량'].isnull())[0]
  lay_red_nan_idx = np.where(df['적색광추정광량'].isnull())[0]

  dlw, dlb, dlr = len(lay_white_nan_idx), len(lay_blue_nan_idx), len(lay_red_nan_idx)
  blw = blb = blr = -1

  df.loc[lay_white_nan_idx, '백색광추정광량'] = df.loc[lay_white_nan_idx, '화이트 LED동작강도'] * 0.01 * 309.41
  df.loc[lay_blue_nan_idx, '청색광추정광량'] = df.loc[lay_blue_nan_idx, '블루 LED동작강도'] * 0.01 * 156.65
  df.loc[lay_red_nan_idx, '적색광추정광량'] = df.loc[lay_red_nan_idx, '레드 LED동작강도'] * 0.01 * 165.48

  LED_white_nan_idx = np.where(df['화이트 LED동작강도'].isnull())[0]
  LED_blue_nan_idx = np.where(df['블루 LED동작강도'].isnull())[0]
  LED_red_nan_idx = np.where(df['레드 LED동작강도'].isnull())[0]

  dLw, dLb, dLr = len(LED_white_nan_idx), len(LED_blue_nan_idx), len(LED_red_nan_idx)
  bLw = bLb = bLr = -1

  df.loc[LED_white_nan_idx, '화이트 LED동작강도'] = df.loc[LED_white_nan_idx, '백색광추정광량'] * 100 / 309.41
  df.loc[LED_blue_nan_idx, '블루 LED동작강도'] = df.loc[LED_blue_nan_idx, '청색광추정광량'] * 100 / 156.65
  df.loc[LED_red_nan_idx, '레드 LED동작강도'] = df.loc[LED_red_nan_idx, '적색광추정광량'] * 100 / 165.48

  total_estimation = np.where(~df['백색광추정광량'].isnull() & ~df['청색광추정광량'].isnull() & ~df['적색광추정광량'].isnull() & df['총추정광량'].isnull())[0]
  df.loc[total_estimation, '총추정광량'] = df.loc[total_estimation, '백색광추정광량'] + df.loc[total_estimation, '청색광추정광량'] + df.loc[total_estimation, '적색광추정광량']

  dte = len(total_estimation)
  bte = -1

  white_error_ohter_nan_idx = np.where(df['백색광추정광량'].isnull() & ~df['청색광추정광량'].isnull() & ~df['적색광추정광량'].isnull())[0]
  blue_error_other_nan_idx = np.where(df['청색광추정광량'].isnull() & ~df['백색광추정광량'].isnull() & ~df['적색광추정광량'].isnull())[0]
  red_error_other_nan_idx = np.where(df['적색광추정광량'].isnull() & ~df['백색광추정광량'].isnull() & ~df['청색광추정광량'].isnull())[0]

  dwe, dbe, dre = len(white_error_ohter_nan_idx), len(blue_error_other_nan_idx), len(red_error_other_nan_idx)
  bwe = bbe = bre = -1
  cnt = 0

  while True:
    cnt += 1
    print(cnt)

    if cnt > 20:
      break # Error 대비

    df.loc[white_error_ohter_nan_idx, '백색광추정광량'] = df.loc[white_error_ohter_nan_idx, '총추정광량'] - df.loc[white_error_ohter_nan_idx, '청색광추정광량'] - df.loc[white_error_ohter_nan_idx, '적색광추정광량']
    df.loc[blue_error_other_nan_idx, '청색광추정광량'] = df.loc[blue_error_other_nan_idx, '총추정광량'] - df.loc[blue_error_other_nan_idx, '백색광추정광량'] - df.loc[blue_error_other_nan_idx, '적색광추정광량']
    df.loc[red_error_other_nan_idx, '적색광추정광량'] = df.loc[red_error_other_nan_idx, '총추정광량'] - df.loc[red_error_other_nan_idx, '백색광추정광량'] - df.loc[red_error_other_nan_idx, '청색광추정광량']

    bwe, bbe, bre = len(white_error_ohter_nan_idx), len(blue_error_other_nan_idx), len(red_error_other_nan_idx)
    
    white_error_ohter_nan_idx = np.where(df['백색광추정광량'].isnull() & ~df['청색광추정광량'].isnull() & ~df['적색광추정광량'].isnull())[0]
    blue_error_other_nan_idx = np.where(df['청색광추정광량'].isnull() & ~df['백색광추정광량'].isnull() & ~df['적색광추정광량'].isnull())[0]
    red_error_other_nan_idx = np.where(df['적색광추정광량'].isnull() & ~df['백색광추정광량'].isnull() & ~df['청색광추정광량'].isnull())[0]

    dwe, dbe, dre = len(white_error_ohter_nan_idx), len(blue_error_other_nan_idx), len(red_error_other_nan_idx)

    # repeat
    blw, blb, blr = len(lay_white_nan_idx), len(lay_blue_nan_idx), len(lay_red_nan_idx)
    bLw, bLb, bLr = len(LED_white_nan_idx), len(LED_blue_nan_idx), len(LED_red_nan_idx)
    bte = len(total_estimation)

    lay_white_nan_idx = np.where(df['백색광추정광량'].isnull())[0]
    lay_blue_nan_idx = np.where(df['청색광추정광량'].isnull())[0]
    lay_red_nan_idx = np.where(df['적색광추정광량'].isnull())[0]

    dlw, dlb, dlr = len(lay_white_nan_idx), len(lay_blue_nan_idx), len(lay_red_nan_idx)

    df.loc[lay_white_nan_idx, '백색광추정광량'] = df.loc[lay_white_nan_idx, '화이트 LED동작강도'] * 0.01 * 309.41
    df.loc[lay_blue_nan_idx, '청색광추정광량'] = df.loc[lay_blue_nan_idx, '블루 LED동작강도'] * 0.01 * 156.65
    df.loc[lay_red_nan_idx, '적색광추정광량'] = df.loc[lay_red_nan_idx, '레드 LED동작강도'] * 0.01 * 165.48

    LED_white_nan_idx = np.where(df['화이트 LED동작강도'].isnull())[0]
    LED_blue_nan_idx = np.where(df['블루 LED동작강도'].isnull())[0]
    LED_red_nan_idx = np.where(df['레드 LED동작강도'].isnull())[0]

    dLw, dLb, dLr = len(LED_white_nan_idx), len(LED_blue_nan_idx), len(LED_red_nan_idx)

    df.loc[LED_white_nan_idx, '화이트 LED동작강도'] = df.loc[LED_white_nan_idx, '백색광추정광량'] * 100 / 309.41
    df.loc[LED_blue_nan_idx, '블루 LED동작강도'] = df.loc[LED_blue_nan_idx, '청색광추정광량'] * 100 / 156.65
    df.loc[LED_red_nan_idx, '레드 LED동작강도'] = df.loc[LED_red_nan_idx, '적색광추정광량'] * 100 / 165.48

    total_estimation = np.where(~df['백색광추정광량'].isnull() & ~df['청색광추정광량'].isnull() & ~df['적색광추정광량'].isnull() & df['총추정광량'].isnull())[0]
    df.loc[total_estimation, '총추정광량'] = df.loc[total_estimation, '백색광추정광량'] + df.loc[total_estimation, '청색광추정광량'] + df.loc[total_estimation, '적색광추정광량']

    dte = len(total_estimation)

    if (bwe == dwe) and (bbe == dbe) and (bre == dre) and (blw == dlw) and (blb == dlb) and (blr == dlr) and \
      (bLw == dLw) and (bLb == dLb) and (bLr == dLr) and (bte == dte):
      break

  # Cooling Error

  cooling_error_idx = np.where(df['냉방부하'].isnull())[0]

  cooling_error_zero_meaning_columns = ['최근분무량', '화이트 LED동작강도', '레드 LED동작강도', '블루 LED동작강도', \
                                      '냉방온도', '냉방부하', '난방온도', '난방부하', '총추정광량', '백색광추정광량', \
                                      '적색광추정광량', '청색광추정광량']

  df.loc[cooling_error_idx, cooling_error_zero_meaning_columns] = 0


  g_inf = df['내부온도관측치'].groupby(df['m_d']).mean()
  m_d_list = g_inf.index

  g_inf = pd.Series(g_inf.to_numpy()).interpolate('polynomial', order=5)
  g_of = pd.Series(df['외부온도관측치'].groupby(df['m_d']).mean().to_numpy()).interpolate('polynomial', order=5)

  g_ih = pd.Series(df['내부습도관측치'].groupby(df['m_d']).mean().to_numpy()).interpolate('polynomial', order=3)
  g_oh = pd.Series(df['외부습도관측치'].groupby(df['m_d']).mean().to_numpy()).interpolate('polynomial', order=1)

  g_co2 = pd.Series(df['CO2관측치'].groupby(df['m_d']).mean().to_numpy()).interpolate('polynomial', order=1)
  g_ec = df['EC관측치'].groupby(df['m_d']).mean().to_numpy()

  for idx, m_d in enumerate(m_d_list):
    df.loc[np.where((df['m_d']==m_d) & (df['내부온도관측치'].isnull()))[0], '내부온도관측치'] = g_inf[idx]
    df.loc[np.where((df['m_d']==m_d) & (df['외부온도관측치'].isnull()))[0], '외부온도관측치'] = g_of[idx]
    df.loc[np.where((df['m_d']==m_d) & (df['내부습도관측치'].isnull()))[0], '내부습도관측치'] = g_ih[idx]
    df.loc[np.where((df['m_d']==m_d) & (df['외부습도관측치'].isnull()))[0], '외부습도관측치'] = g_oh[idx]
    df.loc[np.where((df['m_d']==m_d) & (df['CO2관측치'].isnull()))[0], 'CO2관측치'] = g_co2[idx]
    df.loc[np.where((df['m_d']==m_d) & (df['EC관측치'].isnull()))[0], 'EC관측치'] = g_ec[idx]



  for i in ['내부온도관측치', '외부온도관측치', '내부습도관측치', '외부습도관측치', 'CO2관측치', '최근분무량', '화이트 LED동작강도', '레드 LED동작강도', '블루 LED동작강도', \
            '냉방온도', '냉방부하', '난방온도', '난방부하', '총추정광량', '백색광추정광량', '적색광추정광량', '청색광추정광량']:

    df[i] = df[i].fillna(method='ffill')


  # 냉방난방
  df['냉방-난방'] = df['냉방온도'] - df['난방온도']
  
  # 내부 - 외부 온도가 크다면, 외부에 비해 따뜻하다는 것 => 청경채는 고랭지 채소이므로 더우면 안된다. 따라서, 이상치로 판단
  df['온도차']= df['내부온도관측치'] - df['외부온도관측치']

  # 습도차
  df['습도차'] = df['내부습도관측치'] - df['외부습도관측치']

  # 냉온방 부하율
  df['냉방부하율'] = (df['냉방부하'] / (df['냉방온도'] + 1e-12))
  df['난방부하율'] = (df['난방부하'] / (df['난방부하'] + 1e-12))
  df['총부하량'] = df['냉방부하'] + df['난방부하']

  # 맨 마지막 EC 예측
  filename = path + '/xgb_model.model'

  # 모델 불러와서 EC 예측
  xgb_model = pickle.load(open(filename, 'rb'))

  tdf = df[['내부습도관측치','외부습도관측치','CO2관측치', '최근분무량', '총추정광량', '내부온도관측치', '외부온도관측치', 'EC관측치']]

  test_idx = np.where(tdf['EC관측치'].isnull())[0]

  X_test = df.iloc[test_idx]
  y_test = X_test.pop('EC관측치')

  X_test = X_test[['내부습도관측치','외부습도관측치','CO2관측치', '최근분무량', '총추정광량', '내부온도관측치', '외부온도관측치']]
  prediction = xgb_model.predict(X_test)
  df.loc[test_idx, 'EC관측치'] = prediction

  # 0시, 12시만 활용
  df['hour'] = df['시간'].dt.hour
  df['minute'] = df['시간'].dt.minute

  df_0 = df.iloc[np.where(((df['hour'] == 0) & (df['minute'] == 0)))[0]]
  df_12 = df.iloc[np.where(((df['hour'] == 12) & (df['minute'] == 0)))[0]]

  df_0 = df_0.drop(['시간', 'm_d', 'group', 'hour', 'minute'], axis=1)
  df_12 = df_12.drop(['시간', 'm_d', 'group', 'hour', 'minute'], axis=1)

  new_df_0_columns_ = []
  new_df_12_columns_ = []


  # Trainset은 케이스 및 이름이 존재하므로, 데이터가 섞이지 않도록 Case, Name으로 구분, Testset에는 Case가 없으므로 Name으로만 구분  
  if train_set:

    for i, j in zip(df_0.columns, df_12.columns):
      if i in ['Case', 'Name']:
        new_df_0_columns_.append(i)
      else:
        new_df_0_columns_.append('0시 ' + i)
      new_df_12_columns_.append('12시 ' + j)

    df_0.columns = new_df_0_columns_
    df_12.columns = new_df_12_columns_

    df_0 = df_0.reset_index(level=None, drop=True, inplace=False, col_level=0, col_fill='')
    df_12 = df_12.reset_index(level=None, drop=True, inplace=False, col_level=0, col_fill='')

    df_t = pd.concat([df_0, df_12], axis=1)
    if np.sum(df_t['Case'] == df_t['12시 Case']) == np.sum(df_t['Name'] == df_t['12시 Name']) == len(df_t):
      df_t = df_t.drop(['12시 Case', '12시 Name'], axis=1)
    else:
      print('error')

  else:
    
    for i, j in zip(df_0.columns, df_12.columns):
      if i == 'Name':
        new_df_0_columns_.append(i)
      else:
        new_df_0_columns_.append('0시 ' + i)
      new_df_12_columns_.append('12시 ' + j)

    df_0.columns = new_df_0_columns_
    df_12.columns = new_df_12_columns_

    df_0 = df_0.reset_index(level=None, drop=True, inplace=False, col_level=0, col_fill='')
    df_12 = df_12.reset_index(level=None, drop=True, inplace=False, col_level=0, col_fill='')

    df_t = pd.concat([df_0, df_12], axis=1)
    if np.sum(df_t['Name'] == df_t['12시 Name']) == len(df_t):
      df_t = df_t.drop(['12시 Name'], axis=1)
    else:
      print('error')

  return df, df_t
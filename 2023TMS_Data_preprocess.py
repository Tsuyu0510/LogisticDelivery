#!/usr/bin/env python
# coding: utf-8

# In[1]:


## get raw data 
import numpy as np 
import pandas as pd 

raw = pd.read_csv('2020823.csv')
raw


# In[2]:


## get key data
tms = pd.read_excel('0823.xlsx',engine="openpyxl")
tms


# In[3]:


## join data
result = pd.merge(raw, tms, how="inner", on=["店名", "店名"])
#result.to_csv('newtms0827.csv',encoding='utf-8')


# In[4]:


result


# ## 需確認一件事：用已經發生的資料去join 貨量訊息是否恰當。（會造成後續部署特徵取得的困難度）

# ## Modeling Start From Here
# 

# In[5]:


import pandas as pd 
import numpy as np 
## get data 
raw_data_0823 = pd.read_csv('newtms0823.csv')
raw_data_0824 = pd.read_csv('newtms0824.csv')
raw_data_0825 = pd.read_csv('newtms0825.csv')
raw_data_0826 = pd.read_csv('newtms0826.csv')
raw_data_0827 = pd.read_csv('newtms0827.csv')


# In[6]:


df_1 = raw_data_0824.append(raw_data_0825,ignore_index=True)
df_2 = df_1.append(raw_data_0826,ignore_index=True)
df_3 = df_2.append(raw_data_0827,ignore_index= True)
df_4 = df_3.append(raw_data_0823,ignore_index=True) ## 0823 作為testdata


# In[7]:


df_4 = df_4.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1)
df_4


# In[8]:


df_4['區域'] =""

for i in range(0,len(df_4)):
    if str(df_4.iloc[i,20]) == '3.49噸配送車':
        df_4.iloc[i,9] = 7.2
    elif str(df_4.iloc[i,20]) == '11噸配送車':
        df_4.iloc[i,9] = 14.4
    else:
        df_4.iloc[i,9] = 10
    
    if len(df_4.iloc[i,16])==3:
        df_4.iloc[i,11] = 1
    else:
        df_4.iloc[i,11] = 0
        
    df_4['區域'][i] = df_4['店名'][i][0:2]
    if len(df_4['店名'][i]) == 3:
        df_4['區域'][i] = '北市'

        
        


# In[9]:



## get dummys





df_4 = pd.DataFrame(df_4,columns=['車次','店名','區域','表定時間','預定時間','貨量','裝載率','總貨量','貨量差值','單店裝載率',
                                 '車輛裝載上限','特殊店註記','爆量車次','配送日期','基地','路徑名稱','路徑號碼','司機ID',
                                  '司機名字','車牌','噸型','溫度帶','店號','前室','後室','計畫時間','預測時間','實際時間'
                                  ,'到店差異(分)','到店狀況','爆量註記'])
df_4


# In[10]:


## filter featres
df_4_drop = df_4.drop(['表定時間','預定時間','特殊店註記','溫度帶','司機名字','前室','後室','基地','車牌'],axis = 1)
#df_4_drop  = pd.get_dummies(df_4_drop, columns=['噸型'])
#df_4_drop_drop = pd.get_dummies(df_4_drop, columns=['路徑名稱'])
#df_4_drop_drop
df_total_data = pd.DataFrame(df_4_drop)
df_total_data['計畫時間'] = df_total_data['計畫時間'].apply(pd.Timestamp) ## convert ogject to timestamp
df_total_data['預測時間'] = df_total_data['預測時間'].apply(pd.Timestamp) ## convert ogject to timestamp
df_total_data['實際時間'] = df_total_data['實際時間'].apply(pd.Timestamp) ## convert ogject to timestamp

## convert to int 
df_total_data['計畫時間_int'] = pd.to_datetime(df_total_data['計畫時間']).astype(np.int64) ## convert timestamp to int
df_total_data['預測時間_int'] = pd.to_datetime(df_total_data['計畫時間']).astype(np.int64) ## convert timestamp to int
df_total_data['實際時間_int'] = pd.to_datetime(df_total_data['計畫時間']).astype(np.int64) ## convert timestamp to int



df_total_data


# In[11]:


## 計算單線總工時
import math
df_total_data_test = pd.DataFrame(df_total_data,columns=['爆量車次','計畫時間_int'])
df_total_data_test_sum = df_total_data_test.groupby('爆量車次').sum()
df_total_data_test_sum['計畫時間_log'] = ''
for i in range(0,len(df_total_data_test_sum)):
    df_total_data_test_sum['計畫時間_log'][i] = math.log10(df_total_data_test_sum['計畫時間_int'][i])
df_total_data_test_sum


# In[12]:


## merge data

df_total_data = pd.merge(df_total_data,df_total_data_test_sum,left_on= '爆量車次', right_on= '爆量車次',how='left')


# In[13]:


df_total_data_d = df_total_data.drop(['路徑號碼'],axis =1)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df_total_data_d['區域'] = labelencoder.fit_transform(df_total_data_d['區域'])
df_total_data_d['店名'] = labelencoder.fit_transform(df_total_data_d['店名'])

df_total_data_d['司機ID'] = labelencoder.fit_transform(df_total_data_d['司機ID'])
df_total_data_d['路徑名稱'] = labelencoder.fit_transform(df_total_data_d['路徑名稱'])
#df_total_data_d['噸型'] = labelencoder.fit_transform(df_total_data_d['噸型'])

df_total_data_d = df_total_data_d.drop(['配送日期'],axis=1)


#df_total_data_d['司機ID'] = labelencoder.fit_transform(df_total_data_d['司機ID'])
#df_total_data_d['車牌'] = labelencoder.fit_transform(df_total_data_d['車牌'])


## object to datetime to second(int)


print (df_total_data_d.dtypes)
df_total_data_d


# In[14]:


## re for get tons 
tons_raw = df_total_data_d['噸型'].tolist()
tons_raw = str(tons_raw)


import re 

tons = re.compile(r'\d+\.\d+|\d+')
tons_res = tons.findall(tons_raw)
len(tons_res)

## make a df and concat to df_total_data_d
res_tons = pd.DataFrame(tons_res,columns=['噸型_數值'])
res_tons = res_tons['噸型_數值'].astype(float)
res_tons.dtypes


# In[15]:


## merge 
df_total_data_t = pd.concat([df_total_data_d,res_tons],axis = 1)
df_total_data_t_d = pd.DataFrame(df_total_data_t,columns=['區域','店名','貨量','裝載率','總貨量','貨量差值','單店裝載率','車輛裝載上限',
                                                           '路徑名稱','司機ID','噸型_數值','店號','計畫時間_log',
                                                          '爆量註記'])

df_total_data_t_d_a = df_total_data_t_d.fillna(0)
print (df_total_data_t_d_a.dtypes)
df_total_data_t_d_a


# In[ ]:





# In[20]:


## train, test data

train_x = df_total_data_t_d_a.iloc[0:2851,0:13]
train_y = df_total_data_t_d_a.iloc[0:2851,13]
test_x = df_total_data_t_d_a.iloc[2851:,0:13]
test_y = df_total_data_t_d_a.iloc[2851:,13]
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# In[21]:


print(df_total_data_t_d_a.isnull().any())


# In[22]:


df_total_data_t_d_a


# In[ ]:





# In[23]:


import sklearn
from xgboost import XGBClassifier
xg1 = XGBClassifier(colsample_bytree= 0.3, learning_rate=0.01, max_depth= 5, n_estimators=1000)
xg1=xg1.fit(train_x, train_y)
predxgb = xg1.predict(test_x)
#print (predxgb)

from sklearn.metrics import confusion_matrix

cf_xgb = confusion_matrix(test_y,predxgb)
cf_xgb

## confusion matrix
from sklearn.metrics import confusion_matrix
cf_1 = confusion_matrix(test_y,predxgb)

## visuallize confusion matrix 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_xgb, display_labels = [0, 1])

#ax = sns.heatmap(cf_xgb, annot=True, cmap='Blues')

#ax.set_title('Seaborn Confusion Matrix with labels');
#ax.set_xlabel('Predicted Values')
#ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
#ax.xaxis.set_ticklabels(['False','True'])
#ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.



print (sklearn.metrics.accuracy_score(test_y, predxgb)*100 )
print (sklearn.metrics.recall_score(test_y, predxgb)*100)
print (sklearn.metrics.precision_score(test_y, predxgb)*100)
print (sklearn.metrics.f1_score(test_y, predxgb)*100)

cm_display.plot()

plt.show()


# In[24]:


## try tree
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(train_x,train_y)
tree_pred = clf_tree.predict(test_x)
from sklearn.metrics import confusion_matrix
cf_1 = confusion_matrix(test_y,tree_pred)

## visuallize confusion matrix 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn

get_ipython().run_line_magic('matplotlib', 'inline')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_1, display_labels = [0, 1])

#ax = sns.heatmap(cf_xgb, annot=True, cmap='Blues')

#ax.set_title('Seaborn Confusion Matrix with labels');
#ax.set_xlabel('Predicted Values')
#ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
#ax.xaxis.set_ticklabels(['False','True'])
#ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
cm_display.plot()

plt.show()


print (sklearn.metrics.accuracy_score(test_y, tree_pred)*100 )
print (sklearn.metrics.recall_score(test_y, tree_pred)*100)
print (sklearn.metrics.precision_score(test_y, tree_pred)*100)
print (sklearn.metrics.f1_score(test_y, tree_pred)*100)


# ### 計算爆量車趟數

# In[25]:


## 結合預測結果和test_x, test_y 數量判斷需要多少台專車
#print (tree_pred)
pred_df = pd.DataFrame(tree_pred,columns=['預測爆量註記'])
#print (test_y)
#print (pred_df)
print (test_x)
df_test = pd.DataFrame(test_x).join(test_y)
df_test.reset_index(inplace=True)
df_test_model2 = pd.DataFrame(df_test).join(pred_df)
df_test_model2_drop = df_test_model2.drop(['index'],axis=1)
df_test_model2_drop


# In[26]:


## 篩選出爆量店鋪並加總貨量計算當日所需專車趟數
## 實際數據結果
fill = (df_test_model2_drop['爆量註記'] == 1.0)
df_car_real = df_test_model2_drop.loc[fill, ['貨量', '爆量註記']]
cargo_real = df_car_real['貨量'].sum()
cargo_num_real = cargo_real/7.2
print (cargo_num_real)


# In[27]:


##預測結果
fill_2 = (df_test_model2_drop['預測爆量註記'] == 1.0)
df_car_pred = df_test_model2_drop.loc[fill_2, ['貨量', '預測爆量註記']]
cargo_pred = df_car_pred['貨量'].sum()
cargo_num_pred = cargo_pred/7.2
print (cargo_num_pred)


# ### 實際該日最終派出40台專車

# ### PART  2 

# In[28]:


## 結合預測結果和test_x, test_y 數量判斷需要多少台專車
#print (tree_pred)
pred_df = pd.DataFrame(predxgb,columns=['預測爆量註記'])
#print (test_y)
#print (pred_df)
print (test_x)
df_test = pd.DataFrame(test_x).join(test_y)
df_test.reset_index(inplace=True)
df_test_model2 = pd.DataFrame(df_test).join(pred_df)
df_test_model2_drop = df_test_model2.drop(['index'],axis=1)
df_test_model2_drop


# In[29]:


## 篩選出爆量店鋪並加總貨量計算當日所需專車趟數
## 實際數據結果
fill = (df_test_model2_drop['爆量註記'] == 1.0)
df_car_real = df_test_model2_drop.loc[fill, ['貨量', '爆量註記']]
cargo_real = df_car_real['貨量'].sum()
cargo_num_real = cargo_real/7.2
print (cargo_num_real)


# In[30]:


##預測結果
fill_2 = (df_test_model2_drop['預測爆量註記'] == 1.0)
df_car_pred = df_test_model2_drop.loc[fill_2, ['貨量', '預測爆量註記']]
cargo_pred = df_car_pred['貨量'].sum()
cargo_num_pred = cargo_pred/7.2
print (cargo_num_pred)


# ### model 3 

# In[110]:


## 先取得區域和店名中文
df_total_data_test = pd.DataFrame(df_total_data, columns=['店名','區域'])
df_total_data_test = df_total_data_test.iloc[2851:,:]
len(df_total_data_test)
df_total_data_test = df_total_data_test.rename(columns={'店名':'店名_中','區域':'區域_中'})
df_total_data_test.reset_index(inplace=True)
df_total_data_test_drop = df_total_data_test.drop(['index'],axis=1)
df_total_data_test_drop


# In[111]:


df_test_model2_drop


# In[112]:


## join 中英文店名
test_data_col = pd.DataFrame(df_test_model2_drop, columns=  ['區域','店名'])
mapping_col = df_total_data_test_drop.join(test_data_col)


# In[114]:


fill_3 = (df_test_model2_drop['預測爆量註記']==1)
df_car_pred_cluster = df_test_model2_drop.loc[fill_3,['區域','店名','貨量','裝載率','總貨量','貨量差值','單店裝載率','車輛裝載上限',
                                                      '路徑名稱','司機ID','噸型_數值','店號','計畫時間_log']]
df_car_pred_cluster.reset_index(inplace=True)
df_car_pred_cluster.drop(['index'],axis =1 )


# In[ ]:





# In[115]:


## 分29群
from sklearn.cluster import KMeans

kmeansModel = KMeans(n_clusters=35)
clusters_pred = kmeansModel.fit_predict(df_car_pred_cluster)
print (clusters_pred)


# In[116]:


df_cluster_pred = pd.DataFrame(clusters_pred, columns=['分群預測'])
df_cluster_pred


# In[117]:


## 合併分群結果
pd.set_option('display.max_columns', None) # 設定字元顯示寬度

df_test_model2_cluster = pd.DataFrame(df_car_pred_cluster).join(df_cluster_pred)
df_test_model2_cluster.drop(['index'],axis=1).head(20)


# In[130]:


pd.set_option('display.max_rows', None) # 設定字元顯示寬度
cluster_result = pd.merge(df_test_model2_cluster,mapping_col, on='店名',how = 'inner')
cluster_result = cluster_result.sort_values(by=['分群預測'])
cluster_result_drop = cluster_result.drop(['index','區域_y'],axis=1)
#cluster_result_drop.to_csv('model3_result.csv',encoding='utf-8')
cluster_result_drop


# In[ ]:





# In[ ]:





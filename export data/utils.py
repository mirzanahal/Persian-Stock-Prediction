import json
import os
import re
import emoji
import pandas as pd
import numpy as np
import datetime
from hazm import *

shaspa = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/shaspa.csv')
kmase = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/kmase.csv')
vpars = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/vpars.csv')
vtejarat = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/vtejarat.csv')
famli = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/famli.csv')
foolad = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/foolad.csv')
kgol = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/kgol.csv')
kama = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/kama.csv')
ghesabat = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/ghesabat.csv')
vmelt = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/vmelt.csv')
khodro = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/khodro.csv')
fakhooz = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/fakhooz.csv')
khsaipa = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/khsaipa.csv')
kachad = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/kachad.csv')
shapna = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/shapna.csv')
shbandar = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/shbandar.csv')
shtran = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/shtran.csv')
rmapna = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/rmapna.csv')
khpars = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/khpars.csv')
shabriz = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/shabriz.csv')
fars = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/fars.csv')
vghadir = pd.read_csv('/content/drive/MyDrive/nlp project/tsetmc/vghadir.csv')

nemad_list = {'خودرو' :khodro   ,
              'خساپا'  :khsaipa  ,
              'فولاد'  :foolad   ,
              'فملی'  :famli    ,
              'وتجارت':vtejarat ,
              'فخوز'  :fakhooz  ,
              'وبملت'  :vmelt,
              'شستا'   : shaspa ,
              'وپارس' : vpars,
              'کگل' : kgol,
              'کاما' : kama,
              'قثابت' : ghesabat,
              'کماسه': kmase,
              'کچاد' : kachad,
              'شپنا' : shapna,
              'شبندر' : shbandar,
              'شتران' : shtran,
              'رمپنا' : rmapna,
              'خپارس' : khpars,
              'شبریز' : shabriz,
              'فارس' : fars,
              'وغدیر' : vghadir}

nemad_num = {'خودرو' :0   ,
              'خساپا'  :1  ,
              'فولاد'  :2   ,
              'فملی'  :3    ,
              'وتجارت':4 ,
              'فخوز'  :5  ,
              'وبملت'  :6,
              'شستا'   : 7 ,
              'وپارس' : 8,
              'کگل' : 9,
              'کاما' : 10,
              'قثابت' : 11,
              'کماسه': 12,
              'کچاد' : 13,
              'شپنا' : 14,
              'شبندر' : 15,
              'شتران' : 16,
              'رمپنا' : 17,
              'خپارس' : 18,
              'شبریز' : 19,
             'فارس' : 20,
              'وغدیر' : 21}

def give_emoji_free_text(text):
    return emoji.get_emoji_regexp().sub(r'', text)

def convert_str_to_datetime(day):
    time = (day[:10]).replace('-','')
    y = int(time[:4])
    m = int(time[4:6])
    d = int(time[6:])

    return datetime.datetime(y , m , d)  

def concat_df(df1, df2):
    """Concatenate dataframes"""
    if df1.shape[0] == 0:
        df1 = df2.copy()
    else:
        frames = [df1, df2]
        df1 = pd.concat(frames, ignore_index=True, join="inner")

    return df1
  

def get_binary_labesl( next_day , today , dataframe):
    data_list = []
    check = True
    c = 0
    y = -1
    while check:
      if next_day > today:
        break
      next_day = next_day + datetime.timedelta(days=1)
      d = str(next_day)[:10].replace('-','')
      c += 1
      if int(d) in list(dataframe['<DTYYYYMMDD>']):
        index_label = dataframe.loc[lambda df: df['<DTYYYYMMDD>'] == int(d)].index[0]
        if dataframe.iloc[index_label-1]['<CLOSE>'] < dataframe.iloc[index_label]['<CLOSE>']:
            #price get increase
            y = 1
        else:
            y = 0
        
        data_list += [y] # label [0-1]
        data_list += [next_day] #t2 label_time = first day after news_time that we have price
        data_list += [c] # period between news time and label time
        
        check = False
    # print(data_list)
    return data_list

def get_period_prices(next_day , today , before_day , dataframe):
    check = False
    check2 = False
    c = 0
    data_list = []
    prices_after  = [ 0 for i in range(9)]
    prices_before = [ 0 for i in range(9)]

    while c!=9:
      if c == 8 and check and check2 :
        data_list += [prices_after] # prices for 9 days after news_time
        data_list += [prices_before] # prices for 9 days before news_time
        return data_list

      if next_day > today:
        break
      # after prices 
      next_day = next_day + datetime.timedelta(days=1)
      d = str(next_day)[:10].replace('-','')

      if int(d) in list(dataframe['<DTYYYYMMDD>']):
        check = True
        index_label = dataframe.loc[lambda df: df['<DTYYYYMMDD>'] == int(d)].index[0]

        prices_after[c] = dataframe.iloc[index_label]['<CLOSE>']
      # before prices
      before_day = before_day + datetime.timedelta(days=-1)
      dd = str(before_day)[:10].replace('-','')
      if int(dd) in list(dataframe['<DTYYYYMMDD>']):
        check2 = True
        index_label = dataframe.loc[lambda df: df['<DTYYYYMMDD>'] == int(dd)].index[0]
        prices_before[c] = dataframe.iloc[index_label]['<CLOSE>']
      
      c += 1 
    return data_list


def read_data(channel_name , mode):
    # read data
    path = channel_name + '.json'
    with open(path , 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # process data
    total_list = []
    for ii , id in enumerate(data):
          msg = data[id]['message']
          if msg != None:
            normal_msg = give_emoji_free_text(msg.replace('ي', 'ی').replace('ك', 'ک').replace('ة', 'ه'))
            words = normal_msg.split()
            for w in words:
               if '.com' in w or '@' in w:
                  normal_msg = normal_msg.replace(w , '')
            hashtag = [w for w in words if w[0] == '#' and w[1:] in nemad_list.keys()]
            if len(hashtag) == 1:
                  nemad = hashtag[0][1:]
                  dataframe = nemad_list[nemad]
                  day = convert_str_to_datetime(data[id]['data'])

                  next_day = day 
                  before_day = day
                  today = datetime.datetime.now()

                  # normalization
                  normalizer = Normalizer()
                  normal_msg = normalizer.normalize(normal_msg)

                  li = [channel_name] # channel name
                  li += [nemad] # nemad
                  li += [normal_msg] # normalized message
                  li += [day]
                            
                  if mode == 'binary':
                      li += get_binary_labesl(next_day , today , dataframe)
                      if len(li) >= 5:
                          total_list += [li]
                  if mode == '9days':
                      li += get_period_prices(next_day , today , before_day , dataframe)
                      if len(li) >= 5:
                          total_list += [li]
    if mode == 'binary':                           
        df = pd.DataFrame(np.array(total_list),
                      columns=['channel_name' ,'nemad', 'news', 'news_time', 'label_day'  , 'label_time' , 'c'])
    if mode == '9days':
        df = pd.DataFrame(np.array(total_list),
                      columns=['channel_name' ,'nemad', 'news' , 'news_time' , 'prices after' , 'prices before'])     
    return df

def construct_data(channel , mode):
    DATA = pd.DataFrame()
    for ch in channel:
        df = read_data(ch , mode)
        DATA = concat_df(DATA , df)
    return DATA



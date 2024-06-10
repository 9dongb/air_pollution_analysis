import os
import pandas as pd
import streamlit as st

@st.cache_data()
def load_ap_data(country='korea'):
    k_apData = pd.DataFrame()

    folder_path = f'data/{country}_air_pollution'

    for k in os.listdir(folder_path):
        file_path = folder_path+'/'+k
        area = k.split('-')[0]
        k_apData = pd.concat([k_apData, df_preprocessing(file_path, area)])
    k_apData = k_apData[k_apData['측정일시'] <= '2024-05-31']
    return k_apData
    
def df_preprocessing(file_path, area):        
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df.rename(columns= {'date':'측정일시', ' pm25':'초미세먼지농도(㎍/㎥)', ' pm10':'미세먼지농도(㎍/㎥)', ' o3':'오존농도(ppm)', ' no2':'이산화질소농도(ppm)', ' so2':'아황산가스농도(ppm)', ' co':'일산화탄소농도(ppm)'})
        df['측정소명'] = area

        df1 = df[['측정일시', '측정소명']] # '측정일시'와 '측정소명' 컬럼 선택
        df2 = df.drop(columns=['측정일시', '측정소명'])
        df2 = df2.replace(' ', float('nan'))

        for column in list(df2.columns):
            df2[f'{column}'] = df2[f'{column}'].str.strip().astype(float)

        df2 = df2.fillna(round(df2.mean(), 2))                                   # 결측치를 평균값으로 대체
        data = pd.concat([df1, df2], axis=1)                                     # '측정일시'와 '측정소명'과 대기오염 변수를 합침
        data = data[data['측정일시'].dt.year != 2013]
        return data
    
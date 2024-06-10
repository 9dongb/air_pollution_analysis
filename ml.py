import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from prophet import Prophet
from prophet.plot import plot_plotly

import os
import joblib

def run_ml(apData, c_apData):
    st.write('### 대기 오염 물질 농도 예측')
    predict_ap(apData, c_apData)


@st.cache_data()
def train_model(train_df):
    model = Prophet()
    model.fit(train_df)
    return model


def predict_ap(apData, c_apData):
    # 한글 폰트 설정
    path = "font/H2HDRM.TTF"
    fontprop = fm.FontProperties(fname=path, size=12)

    col1, col2, col3, col4 = st.columns(4)

    sgg_names = list(apData['측정소명'].unique())
    c_sgg_names = list(c_apData['측정소명'].unique())
    sgg_names = [sgg_name for sgg_name in sgg_names if sgg_name is not np.nan]
    
    ap_list_u = apData.columns.to_list()[2:]
    ap_list = [al[:al.find('(')] for al in ap_list_u]

    with col1:
        selected_country = st.selectbox('국가', ['한국', '중국'])
    with col2:
        if selected_country == '한국':
            selected_sgg = st.selectbox('행정구역', sgg_names)
        else:
            selected_sgg = st.selectbox('행정구역', c_sgg_names)
    with col3:
        selected_ap = st.selectbox('대기오염 물질', ap_list)
    with col4:
        periods = int(st.number_input("예측 기간 설정", min_value=1, max_value=30))
    
    ap_index = ap_list.index(selected_ap)
    
    model = Prophet() # 모델 객체 생성

    train_df = apData.groupby("측정일시")[ap_list_u[ap_index]].agg('mean').reset_index()
    train_df = train_df.rename(columns={'측정일시': 'ds', ap_list_u[ap_index]:'y'})
    
    model_path = f"models/prophet_model_{selected_country}_{selected_sgg}_{selected_ap}.pkl"
    
    if os.path.exists(model_path):                  # 경로에 모델이 존재할 경우
        model = joblib.load(model_path)             # 모델 로드
    else:                                           # 경로에 모델이 존재하지 않을 경우
        model = train_model(train_df)      # 모델 훈련
        joblib.dump(model, model_path)              # 모델 저장

    future = model.make_future_dataframe(periods=periods) # 예측 결과를 저장할 ds만 존재하는 데이터 프레임(ds만 존재) 만들기 (2014~2023 + 예측일 크기)
    
    forcast = model.predict(future) # 예측 데이터 프레임 생성 (여러 컬럼 존재)

    easy_forcast = forcast[forcast['ds'] >= '2023-01-01']
    # st.write(forcast)
    fig = plot_plotly(model, forcast)

    fig.update_layout(
        title=dict(
            text=f"한국 {selected_sgg} {selected_ap} 평균값 예측 {periods} 일간",
            font=dict(size=20),
            yref="paper",
        ),
        xaxis_title="날짜",
        yaxis_title=ap_list_u[ap_index],
        autosize=False,
        width=700,
        height=800,
    )
    fig.update_yaxes(tickformat="000")
    st.plotly_chart(fig)
    
    

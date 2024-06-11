import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

import matplotlib.pyplot as plt
import seaborn as sns

from pingouin import ttest
import pingouin as pg

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def showStat(k_apData, c_apData):

    selected_analysis = st.sidebar.selectbox('분석 방법', ['차이 검정', '상관 분석'])
    selected_china = st.sidebar.selectbox('비교 대상', c_apData['측정소명'].unique())
    years = k_apData['측정일시'].dt.year.unique()
    
    if selected_analysis == '차이 검정':
        selected_year = st.sidebar.selectbox('비교 연도', years)
        differenceTest(k_apData, c_apData, selected_china, selected_year, selected_analysis)
        
    if selected_analysis == '상관 분석':
        all_correlation(k_apData, c_apData, selected_china, selected_analysis)


def differenceTest(apData, c_apData, selected_china, selected_year, selected_analysis):
    st.write(f'### {selected_year}년 한국과 {selected_china}의 대기오염 물질 {selected_analysis}')
    ap_list = sorted(apData.columns.to_list()[2:]) # 대기오염 물질 목록

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_area = st.selectbox('행정구역', np.insert(apData['측정소명'].unique(),0, ['전체']))
    with col2:
        selected_ap = st.selectbox('대기 오염 물질 선택', ap_list)
    with col3:
            selected_view = st.radio('보기 방식', ['간단히 보기', '자세히 보기'])
    apData['year'] = apData['측정일시'].dt.year
    c_apData['year'] = c_apData['측정일시'].dt.year

    ap_unit_index = selected_ap.find('(')
    ap = selected_ap[:ap_unit_index]
    ap_unit = selected_ap[ap_unit_index:]

    df = pd.DataFrame()
    if selected_area == '전체':
        area_list = apData['측정소명'].unique()
        
    else:
        area_list = [selected_area]

    for area in area_list:
        filtered_data = apData[(apData['측정소명'] == area) & (apData['year'] == selected_year) ]
        c_filtered_data = c_apData[(c_apData['측정소명'] == selected_china) & (c_apData['year'] == selected_year)]
        
        ttest_result = ttest(filtered_data[selected_ap], c_filtered_data[selected_ap], paired=False)
        
        p_value = round(ttest_result['p-val'].values[0], 2)

        status_num = 0 if p_value > 0.05 else 1

        status1 = ['이상', '미만'][status_num]
        status2 = ['통계적으로 유의하지않다.','통계적으로 유의하다.'][status_num]
        status3 = ['X','O'][status_num]
        
        area_mean = filtered_data[selected_ap].mean().round(0)
        area_df = pd.DataFrame({'행정구역':[area], f'{selected_ap} 평균': [area_mean], 'p-value':[p_value],'통계적 유의성':[status3]})
        df = pd.concat([df, area_df])

        if selected_view != '간단히 보기':
            st.markdown(f'- **{area}**와(과) **{selected_china}**의 p-val 값이 0.05 {status1}으로 {selected_ap} 농도 차이는 **{status2}**')
            with st.expander('- 자세히 보기') :
                st.dataframe(ttest_result, use_container_width=True)
                st.markdown(
                            f'  - **{area}** {ap} 평균: {area_mean} {ap_unit}\n'
                            f'  - **{selected_china}**  {ap}평균: { c_filtered_data[selected_ap].mean().round(0)}{ap_unit}\n'
                            f'  - **p-value**: {p_value}')
                            #f'- {area}와 {selected_china}의 {ap} 농도 차이 검정\n'

    df = df.reset_index(drop=True)
    df.index = df.index+1
    if selected_view == '간단히 보기':
        st.dataframe(df)

def all_correlation(apData, c_apData, selected_china, selected_analysis):
    st.write(f'### 한국과 {selected_china}의 대기오염 물질 {selected_analysis}')
    col1, col2, col3 = st.columns(3)
    

    with col1:
        selected_area = st.selectbox('행정구역', np.insert(apData['측정소명'].unique(),0, ['전체']))
    with col2:
        selected_ap = st.selectbox('대기오염 물질', apData.columns.unique()[2:])
    with col3:
        selected_view = st.radio('보기 방식', ['간단히 보기', '자세히 보기'])

    df = pd.DataFrame()
    if selected_area == '전체':
        area_list = apData['측정소명'].unique()
        
    else:
        area_list = [selected_area]

    for area in area_list:
        c_filtered_data = c_apData[c_apData['측정소명'] == selected_china]
        c_filtered_data = c_filtered_data.rename(columns = {selected_ap: f'{selected_china} {selected_ap}'})
        c_filtered_data['year'] = c_filtered_data['측정일시'].dt.year
        c_filtered_data['month'] = c_filtered_data['측정일시'].dt.month
        c_filtered_data = c_filtered_data.groupby(['year', 'month'])[f'{selected_china} {selected_ap}'].agg('mean')
        c_filtered_data = c_filtered_data.reset_index()
        
        filtered_data = apData[apData['측정소명']==area]
        filtered_data['year'] = filtered_data['측정일시'].dt.year
        filtered_data['month'] = filtered_data['측정일시'].dt.month
        filtered_data = filtered_data.rename(columns={selected_ap:f'{area} {selected_ap}'})
        filtered_data = filtered_data.groupby(['year', 'month'])[f'{area} {selected_ap}'].agg('mean')
        filtered_data = filtered_data.reset_index()

        total_data = pd.merge(filtered_data, c_filtered_data)
        corr_r = pg.corr(total_data[f'{area} {selected_ap}'], total_data[f'{selected_china} {selected_ap}']).round(3)['r']
        
        corr_r = corr_r.item()
        status3 = '양의 상관관계' if corr_r >= 0.5 else ('음의 상관관계' if corr_r <= -0.5 else '관계성 적음')

        # status3 = ['X','O']
        area_mean = filtered_data[f'{area} {selected_ap}'].mean().round(0)
        area_df = pd.DataFrame({'행정구역':[area], f'{selected_ap} 평균': [area_mean], '상관계수':[corr_r],'상관관계':[status3]})
        df = pd.concat([df, area_df])

        if selected_view =='자세히 보기':
            ######################################################################################

            
            # st.write('### 피어슨 상관관계 계수 및 검정')

            if (corr_r > 0.5):
                st.markdown(f'- 피어슨 상관계수는 **{corr_r}**이며, **중국 {selected_china}**의 미세먼지 농도가 **증가**할수록 **{area}**미세먼지 농도도 **증가**하는 경향성을 가진다.')
            elif (corr_r < -0.5):
                st.markdown(f'- 피어슨 상관계수는 {corr_r}이며, **중국 {selected_china}**의 미세먼지 농도가 **증가**할수록 **{area}** 미세먼지 농도는 **감소**하는 경향성을 가진다.')
            else:
                st.markdown(f'- 피어슨 상관계수는 {corr_r}이며, **중국 {selected_china}**의 미세먼지 농도와 **{area}** 미세먼지 농도의 **관계성**은 비교적 **적다.**')

            

            ######################################################################################
        
                
            with st.expander('- 자세히 보기') :
                st.write( pg.corr( total_data[f'{area} {selected_ap}'], total_data[f'{selected_china} {selected_ap}'] ) )
                st.write(f'### {selected_ap[:selected_ap.find("(")]} 상관관계 분석 시각화 (산포도)')
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.rcParams['font.family'] = 'Malgun Gothic'
                sns.scatterplot(x=f'{area} {selected_ap}', y=f'{selected_china} {selected_ap}', data=total_data, ax=ax)
                st.pyplot(fig)


                st.write('### 월별 그래프 (2014 ~ 2023)')
                # Subplots 생성
                fig2 = make_subplots(rows=3, cols=4, subplot_titles=[str(year) for year in range(2014, 2024)])
                
                for idx, year in enumerate(range(2014, 2024)):
                    row = idx // 4 + 1
                    col = idx % 4 + 1
                    
                    # 한국 미세먼지 농도 라인 플롯
                    fig2.add_trace(
                        go.Scatter(x=total_data[total_data['year'] == year]['month'], 
                                y=total_data[total_data['year'] == year][f'{area} {selected_ap}'], 
                                mode='lines', 
                                name=f'한국 {area} {year}'),
                        row=row, col=col
                    )
                    
                    # 중국 미세먼지 농도 라인 플롯
                    fig2.add_trace(
                        go.Scatter(x=total_data[total_data['year'] == year]['month'], 
                                y=total_data[total_data['year'] == year][f'{selected_china} {selected_ap}'], 
                                mode='lines', 
                                name=f'중국 {selected_china} {year}'),
                        row=row, col=col
                    )

                # 레이아웃 업데이트
                fig2.update_layout(height=900, width=1200)
                st.plotly_chart(fig2)
    df = df.reset_index(drop=True)
    df.index = df.index+1
    if selected_view == '간단히 보기':
        
        st.write(f'### 한국 {selected_area}와 {selected_china} {selected_ap} 상관 분석')
        st.dataframe(df)



def correlation(apData, c_apData):
    selected_china = st.sidebar.selectbox('비교 대상', ['베이징', '지난시'])
    st.write(f'## {selected_china}의 미세먼지와 서울 미세먼지 분석')
    st.write(f'중국 {selected_china}의 미세 먼지 농도와 한국 서울시 미세먼지 분석')
    st.write('### 상관관계 분석 시각화 (산포도)')
    c_filtered_data = c_apData[c_apData['측정일시'].dt.year != 2024]
    c_filtered_data = c_filtered_data[c_filtered_data['측정소명'] == selected_china]
    c_filtered_data = c_filtered_data.rename(columns = {'미세먼지농도(㎍/㎥)': f'{selected_china} 미세먼지농도(㎍/㎥)'})
    c_filtered_data['year'] = c_filtered_data['측정일시'].dt.year
    c_filtered_data['month'] = c_filtered_data['측정일시'].dt.month
    c_filtered_data = c_filtered_data.groupby(['year', 'month'])[f'{selected_china} 미세먼지농도(㎍/㎥)'].agg('mean')
    c_filtered_data = c_filtered_data.reset_index()
    

    apData = apData[apData['측정소명']=='서울']
    apData['year'] = apData['측정일시'].dt.year
    apData['month'] = apData['측정일시'].dt.month
    filtered_data = apData.rename(columns={'미세먼지농도(㎍/㎥)':'서울 미세먼지농도(㎍/㎥)'})
    filtered_data = filtered_data.groupby(['year', 'month'])['서울 미세먼지농도(㎍/㎥)'].agg('mean')
    filtered_data = filtered_data.reset_index()

    total_data = pd.merge(filtered_data, c_filtered_data)
    ######################################################################################
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='서울 미세먼지농도(㎍/㎥)', y=f'{selected_china} 미세먼지농도(㎍/㎥)', data=total_data, ax=ax)
    st.pyplot(fig)


    st.write('### 월별 그래프 (2014 ~ 2023)')
    # Subplots 생성
    fig2 = make_subplots(rows=3, cols=4, subplot_titles=[str(year) for year in range(2014, 2024)])
    
    for idx, year in enumerate(range(2014, 2024)):
        row = idx // 4 + 1
        col = idx % 4 + 1
        
        # 서울 미세먼지 농도 라인 플롯
        fig2.add_trace(
            go.Scatter(x=total_data[total_data['year'] == year]['month'], 
                       y=total_data[total_data['year'] == year]['서울 미세먼지농도(㎍/㎥)'], 
                       mode='lines', 
                       name=f'서울 {year}'),
            row=row, col=col
        )
        
        # 베이징 미세먼지 농도 라인 플롯
        fig2.add_trace(
            go.Scatter(x=total_data[total_data['year'] == year]['month'], 
                       y=total_data[total_data['year'] == year]['베이징 미세먼지농도(㎍/㎥)'], 
                       mode='lines', 
                       name=f'베이징 {year}'),
            row=row, col=col
        )

    # 레이아웃 업데이트
    fig2.update_layout(height=900, width=1200)
    st.plotly_chart(fig2)



    ######################################################################################
    corr_r = pg.corr(total_data['서울 미세먼지농도(㎍/㎥)'], total_data[f'{selected_china} 미세먼지농도(㎍/㎥)']).round(3)['r']
    st.write('### 피어슨 상관관계 계수 및 검정')

    if (corr_r.item() > 0.5):
        st.markdown(f'- 피어슨 상관계수는 **{corr_r.item()}**이며, 중국 {selected_china}의 미세먼지 농도가 **증가**할수록 서울 미세먼지 농도도 **증가**하는 경향성을 가진다.')
    elif (corr_r.item() < -0.5):
        st.markdown(f'- 피어슨 상관계수는 {corr_r.item()}이며, 중국 {selected_china}의 미세먼지 농도가 **증가**할수록 서울 미세먼지 농도는 **감소**하는 경향성을 가진다.')
    else:
        st.markdown(f'- 피어슨 상관계수는 {corr_r.item()}이며, 중국 {selected_china}의 미세먼지 농도와 서울 미세먼지 농도의 **관계성**은 비교적 **적다.**')

    st.write( pg.corr( total_data['서울 미세먼지농도(㎍/㎥)'], total_data[f'{selected_china} 미세먼지농도(㎍/㎥)'] ) )

    
# def differenceTest(apData, c_apData):   
#     apData = apData[apData['측정소명'] == '서울']
#     ap_list = sorted(apData.columns.to_list()[2:])
#     c_s = st.sidebar.selectbox('중국 지역 선택', ['베이징', '지난시'])
#     selected_ap = st.sidebar.selectbox('오염 물질 선택', ap_list)

    
#     apData['year'] = apData['측정일시'].dt.year
#     c_apData['year'] = c_apData['측정일시'].dt.year
    
#     year_list = sorted(apData['year'].unique(), reverse=True)
#     selected_year = st.sidebar.selectbox('검증 연도 선택', year_list)

#     ap_unit_index = selected_ap.find('(')
#     ap = selected_ap[:ap_unit_index]
#     ap_unit = selected_ap[ap_unit_index:]

#     st.markdown(f'### 서울과 {c_s}의 {selected_year}년 {ap} 농도 차이 검증')



#     c_filtered_data = c_apData[(c_apData['측정소명'] == c_s) & (c_apData['year'] == selected_year)]
#     filtered_data = c_apData[c_apData['year'] == selected_year]


#     st.markdown(f"- 서울 {selected_year}년 {ap}\n"
#                 f"   - 평균 농도: { filtered_data[selected_ap].mean().round(0)}{ap_unit}")
#     st.markdown(f"- {c_s} {selected_year}년 {ap}\n" 
#                 f"  - 평균 농도: { c_filtered_data[selected_ap].mean().round(0)}{ap_unit}")
    
#     ttest_result = ttest(filtered_data[selected_ap], c_filtered_data[selected_ap], paired=False)
#     st.dataframe(ttest_result, use_container_width=True)

#     status_num = 0 if ttest_result['p-val'].values[0] > 0.05 else 1

#     status1 = ['이상', '미만'][status_num]
#     status2 = ['통계적으로 유의지않다.','통계적으로 유의하다.'][status_num]
    
#     st.markdown(f'p-val 값이 0.05 {status1}으로 서울과 {c_s}의 {selected_year}년 {selected_ap} 농도 차이는 **{status2}**')

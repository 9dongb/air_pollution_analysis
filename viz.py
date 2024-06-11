import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from millify import prettify

def showViz(k_apData, c_Data):
    
    with st.sidebar:
        selected_radio = st.radio('데이터 셋',['한국 대기오염 농도', '중국 대기오염 농도'])

    if selected_radio == '한국 대기오염 농도':
        with st.sidebar:
            selected_radio2 = st.radio('선택',['데이터셋 탐색', '조건별 평균 대기오염 농도', '행정구역별 대기오염 농도'])
        if selected_radio2 == '데이터셋 탐색':
            defaultAirPollution(k_apData)
        if selected_radio2 == '조건별 평균 대기오염 농도':
            meanAirPollution(k_apData)
        if selected_radio2 == '행정구역별 대기오염 농도':
            regionAirPollution(k_apData)

    if selected_radio == '중국 대기오염 농도':
        with st.sidebar:
            selected_radio2 = st.radio('선택',['데이터셋 탐색', '조건별 평균 대기오염 농도', '행정구역별 대기오염 농도'])
        if selected_radio2 == '데이터셋 탐색':
            defaultAirPollution(c_Data)
        if selected_radio2 == '조건별 평균 대기오염 농도':
            meanAirPollution(c_Data)
        if selected_radio2 == '행정구역별 대기오염 농도':
            regionAirPollution(c_Data)
    if selected_radio == '3':
        st.write('예이예')

####대기오염 기본 데이터셋 정보 반환 함수##########################################################################################
def defaultAirPollution(apData):
    apData['month'] = apData['측정일시'].dt.month
    apData['year'] = apData['측정일시'].dt.year

    sgg_nm = sorted(apData['측정소명'].unique())
    year = sorted(apData['year'].unique())
    month = sorted(apData['month'].unique())
    
    c_nm = '중국' if '베이징' in sgg_nm else '한국'

    col1, col2, col3, col4 = st.columns(4)
    with col2:
        selected_sgg = st.selectbox('행정구역', sgg_nm)
    with col3:
        selected_year = st.selectbox('연도 선택', year)
    with col4: 
        selected_month = st.selectbox('월 선택', month)
    # selected_month = st.radio('자료 형태', ['월 별', '그래프'], horizontal=True)
    with col1:
        ouput_form = st.radio('출력 형태', ['표', '그래프'], horizontal=True)

    filtered_data = apData[(apData['측정소명']==selected_sgg) & (apData['측정일시'].dt.year==selected_year) & (apData['측정일시'].dt.month==selected_month)]
    filtered_data.reset_index(drop=True, inplace=True)
    filtered_data['측정일시'] = filtered_data['측정일시'].dt.date

    y= ''
    m=''
    d=''  
    col5, col6 = st.columns(2)

    # 그래프에서 미세먼지가 가장 높은 곳에 텍스트 추가
    max_value = max(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 값 반환환
    max_index =np.argmax(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
    filtered_data['미세먼지농도(㎍/㎥)'].iloc[max_index]

    # 그래프에서 초미세먼지가 가장 높은 곳에 텍스트 추가
    max_value_2 = max(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 값 반환
    max_index_2 =np.argmax(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
    
    y = '**('+str(filtered_data['측정일시'].iloc[max_index])+')**'

   # 최대값을 맨 위에 표시
    with col5:
        st.metric(label=f'최대 미세먼지 농도\n {y}{m}{d} ', value=f'{round(max_value, 2)}(㎍/㎥)')
    with col6:
        st.metric(label=f'최대 초미세먼지 농도\n {y}{m}{d}', value=f'{round(max_value_2, 2)}(㎍/㎥)')                            
    if ouput_form == '표':

        st.markdown(f'### {c_nm} {selected_sgg} {selected_year}년 {selected_month}월 대기오염 농도 표')
        st.dataframe(filtered_data.drop(['측정소명'], axis=1)) # 위에서 측정소명은 기재되어 있기 때문에 데이터 프레임에서 제외
    elif ouput_form =='그래프':

        st.markdown(f'### {c_nm} {selected_sgg} {selected_year}년 {selected_month}월 대기오염 농도 그래프 (㎍/㎥)')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_data['측정일시'], y=filtered_data['미세먼지농도(㎍/㎥)'],
                    mode='lines', # Line plot만 그리기
                    name='미세먼지농도(㎍/㎥)'))
        fig.add_trace(go.Scatter(x=filtered_data['측정일시'], y=filtered_data['초미세먼지농도(㎍/㎥)'],
                    mode='lines', # Line plot만 그리기
                    name='초미세먼지농도(㎍/㎥)'))
        
        # 그래프에서 미세먼지가 가장 높은 곳에 텍스트 추가
        max_value = max(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 값 반환환
        max_index =np.argmax(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
        
        fig.add_annotation(
            x=filtered_data['측정일시'].iloc[max_index], y=filtered_data['미세먼지농도(㎍/㎥)'].iloc[max_index],
            text=f"최대:{max_value}(㎍/㎥)",  # 원하는 텍스트 입력
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40
        )

        # 그래프에서 초미세먼지가 가장 높은 곳에 텍스트 추가
        max_value = max(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 값 반환
        max_index =np.argmax(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
        # 그래프에 텍스트 추가
        fig.add_annotation(
            x=filtered_data['측정일시'].iloc[max_index], y=filtered_data['초미세먼지농도(㎍/㎥)'].iloc[max_index],
            text=f"최대:{max_value}(㎍/㎥)",  # 원하는 텍스트 입력
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-30
        )
        st.plotly_chart(fig)

        st.markdown(f'### {c_nm} {selected_sgg} {selected_year}년 {selected_month}월 대기오염 농도 그래프 (ppm)')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=filtered_data['측정일시'], y=filtered_data['이산화질소농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='이산화질소농도(ppm)'))
        fig2.add_trace(go.Scatter(x=filtered_data['측정일시'], y=filtered_data['오존농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='오존농도(ppm)'))
        fig2.add_trace(go.Scatter(x=filtered_data['측정일시'], y=filtered_data['일산화탄소농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='일산화탄소농도(ppm)'))
        fig2.add_trace(go.Scatter(x=filtered_data['측정일시'], y=filtered_data['아황산가스농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='아황산가스농도(ppm)'))
        
        st.plotly_chart(fig2)

    csv = filtered_data.drop(['측정소명'], axis=1).to_csv(index=False).encode('utf-8')
    st.sidebar.download_button('결과 다운로드(CSV)', csv, f'AP.csv', 'text/csv', key='download-csv')


def meanAirPollution(apData):
    


    sgg_nm = sorted(apData['측정소명'].unique())
    year = sorted(apData['측정일시'].dt.year.unique())
    month = sorted(apData['측정일시'].dt.month.unique())

    apData['month'] = apData['측정일시'].dt.month
    apData['year'] = apData['측정일시'].dt.year

    c_nm = '중국' if '베이징' in sgg_nm else '한국'
    
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        selected_ym = st.radio('구분', ['연도', '월', '일'])

    with col2:
            selected_sgg = st.selectbox('행정구역', sgg_nm)
    with col3:
        endYear = int(np.argmax(year))
        if selected_ym == '연도':
            selected_startYear = st.selectbox('시작 연도 선택', year, index= endYear-5)
        else:
            selected_startYear = st.selectbox('연도 선택', year, index= endYear-5)
    with col4:
        end_year_container = st.empty() # 빈 컨테이너 생성
        if selected_ym == '연도': # 연도일 때만 끝년 선택을 표시
            selected_endYear = st.selectbox('끝 연도 선택', year, index= endYear)
        elif selected_ym == '월':
            selected_endYear = None
        else:
            selected_endYear = None # selected_endYear가 없어 다른 코드에서 발생하는 오류를 방지를 위해
            with col4:
                selected_month = st.selectbox('월 선택', month)
    max_value = 0
    max_value_2 = 0
    
    max_index = 0
    max_index_2 = 0
    y = ''
    m = ''
    d = ''
    col5, col6 = st.columns(2)
        # 필터링 및 최대값 계산
    if selected_ym == '월':
        filtered_data = apData[(apData['측정소명'] == selected_sgg) & (apData['year'] == selected_startYear)]
        filtered_data = filtered_data.groupby('month')[['미세먼지농도(㎍/㎥)', '초미세먼지농도(㎍/㎥)']].agg('mean')
        filtered_data = filtered_data.reset_index()
        max_value = filtered_data['미세먼지농도(㎍/㎥)'].max()
        max_value_2 = filtered_data['초미세먼지농도(㎍/㎥)'].max()
        max_index =np.argmax(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
        max_index_2 =np.argmax(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
        y = '**('+str(filtered_data['month'].iloc[max_index])+'월)**'
    elif selected_ym == '연도':
        filtered_data = apData[(apData['측정소명'] == selected_sgg) & (apData['year'].between(selected_startYear, selected_endYear))]
        filtered_data = filtered_data.groupby('year')[['미세먼지농도(㎍/㎥)', '초미세먼지농도(㎍/㎥)']].agg('mean')
        filtered_data = filtered_data.reset_index()
        max_value = filtered_data['미세먼지농도(㎍/㎥)'].max()
        max_value_2 = filtered_data['초미세먼지농도(㎍/㎥)'].max()
        max_index =np.argmax(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
        max_index_2 =np.argmax(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
        y = '**('+str(filtered_data['year'].iloc[max_index])+'년)**'
    elif selected_ym == '일':
        filtered_data = apData[(apData['측정소명']==selected_sgg) & (apData['year'] == selected_startYear) & (apData['month'] == selected_month)]
        filtered_data['day'] = filtered_data['측정일시'].dt.day
        filtered_data = filtered_data.reset_index()
        max_value = filtered_data['미세먼지농도(㎍/㎥)'].max()
        max_value_2 = filtered_data['초미세먼지농도(㎍/㎥)'].max()
        max_index =np.argmax(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
        max_index_2 =np.argmax(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
        y = '**('+str(filtered_data['day'].iloc[max_index])+'일)**'
    # 최대값을 맨 위에 표시
    with col5:
        st.metric(label=f'최대 미세먼지 농도\n {y}{m}{d} ', value=f'{round(max_value, 2)}(㎍/㎥)')
    with col6:
        st.metric(label=f'최대 초미세먼지 농도\n {y}{m}{d}', value=f'{round(max_value_2, 2)}(㎍/㎥)')

    if selected_ym == '월':
        filtered_data = apData[(apData['측정소명']==selected_sgg) & (apData['year'] == selected_startYear)]
        filtered_data = filtered_data.groupby('month')[['측정일시', '미세먼지농도(㎍/㎥)', '초미세먼지농도(㎍/㎥)', '이산화질소농도(ppm)', '오존농도(ppm)', '일산화탄소농도(ppm)', '아황산가스농도(ppm)']].agg('mean')
        filtered_data['month'] = filtered_data['측정일시'].dt.month

        # max_value = max(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 값 반환
        # max_value_2 = max(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 값 반환
        
        # max_index =np.argmax(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
        # max_index_2 =np.argmax(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_data['month'], y=filtered_data['미세먼지농도(㎍/㎥)'],
                    mode='lines', # Line plot만 그리기
                    name='미세먼지농도(㎍/㎥)'))
        fig.add_trace(go.Scatter(x=filtered_data['month'], y=filtered_data['초미세먼지농도(㎍/㎥)'],
                    mode='lines', # Line plot만 그리기
                    name='초미세먼지농도(㎍/㎥)'))
        # 그래프 레이아웃 설정 (제목 포함)
        fig.update_layout(
            title= f"{c_nm} {selected_sgg} {selected_startYear}년 월 평균 (초)미세먼지 농도 추이 ",
            xaxis_title="월",
            yaxis_title="농도(㎍/㎥)")
                # 그래프에서 초미세먼지가 가장 높은 곳에 텍스트 추가

        

        # 그래프에 텍스트 추가
        fig.add_annotation(
            x=filtered_data['month'].iloc[max_index], y=filtered_data['미세먼지농도(㎍/㎥)'].iloc[max_index],
            text=f"최대:{round(max_value, 2)}(㎍/㎥)",  # 원하는 텍스트 입력
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-30)
        
        fig.add_annotation(
        x=filtered_data['month'].iloc[max_index_2], y=filtered_data['초미세먼지농도(㎍/㎥)'].iloc[max_index_2],
            text=f"최대:{round(max_value_2, 2)}(㎍/㎥)",  # 원하는 텍스트 입력
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-30)

        st.plotly_chart(fig)

        ### 기타 대기오염 그래프 (ppm)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=filtered_data['month'], y=filtered_data['이산화질소농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='이산화질소농도(ppm)'))
        fig2.add_trace(go.Scatter(x=filtered_data['month'], y=filtered_data['오존농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='오존농도(ppm)'))
        fig2.add_trace(go.Scatter(x=filtered_data['month'], y=filtered_data['일산화탄소농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='일산화탄소농도(ppm)'))
        fig2.add_trace(go.Scatter(x=filtered_data['month'], y=filtered_data['아황산가스농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='아황산가스농도(ppm)'))
        # 그래프 레이아웃 설정 (제목 포함)
        fig2.update_layout(
            title= f"{c_nm} {selected_sgg} {selected_startYear}년 월 평균 4종 대기오염 물질 농도 추이 ",
            xaxis_title="월",
            yaxis_title="농도(㎍/㎥)")
        

        st.plotly_chart(fig2)
    elif selected_ym == '연도':
        filtered_data = apData[(apData['측정소명']==selected_sgg) & (apData['year'].between(selected_startYear, selected_endYear))]
        
        filtered_data = filtered_data.groupby('year')[['측정일시', '미세먼지농도(㎍/㎥)', '초미세먼지농도(㎍/㎥)', '이산화질소농도(ppm)', '오존농도(ppm)', '일산화탄소농도(ppm)', '아황산가스농도(ppm)']].agg('mean')
        filtered_data['year'] = filtered_data['측정일시'].dt.year
        ### (초)미세먼지 농도 그래프 (㎍/㎥)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_data['year'], y=filtered_data['미세먼지농도(㎍/㎥)'],
                    mode='lines', # Line plot만 그리기
                    name='미세먼지농도(㎍/㎥)'))
        fig.add_trace(go.Scatter(x=filtered_data['year'], y=filtered_data['초미세먼지농도(㎍/㎥)'],
                    mode='lines', # Line plot만 그리기
                    name='초미세먼지농도(㎍/㎥)'))
        # 그래프 레이아웃 설정 (제목 포함)
        
        fig.update_layout(
            title= f"{c_nm} {selected_sgg} {selected_startYear}년 ~ {selected_endYear}년 사이 연 평균 (초)미세먼지 농도 추이 ",
            xaxis_title="연도",
            yaxis_title="농도(㎍/㎥)",)
        
        max_value = max(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 값 반환
        max_value_2 = max(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 값 반환
        
        max_index =np.argmax(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
        max_index_2 =np.argmax(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
        

        # 그래프에 텍스트 추가
        fig.add_annotation(
            x=filtered_data['year'].iloc[max_index], y=filtered_data['미세먼지농도(㎍/㎥)'].iloc[max_index],
            text=f"최대:{round(max_value, 2)}(㎍/㎥)",  # 원하는 텍스트 입력
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-30)
        
        fig.add_annotation(
        x=filtered_data['year'].iloc[max_index_2], y=filtered_data['초미세먼지농도(㎍/㎥)'].iloc[max_index_2],
            text=f"최대:{round(max_value_2, 2)}(㎍/㎥)",  # 원하는 텍스트 입력
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-30)
        st.plotly_chart(fig)

        ### 4종 대기오염 그래프 (ppm)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=filtered_data['year'], y=filtered_data['이산화질소농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='이산화질소농도(ppm)'))
        fig2.add_trace(go.Scatter(x=filtered_data['year'], y=filtered_data['오존농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='오존농도(ppm)'))
        fig2.add_trace(go.Scatter(x=filtered_data['year'], y=filtered_data['일산화탄소농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='일산화탄소농도(ppm)'))
        fig2.add_trace(go.Scatter(x=filtered_data['year'], y=filtered_data['아황산가스농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='아황산가스농도(ppm)'))
        # 그래프 레이아웃 설정 (제목 포함)
        fig2.update_layout(
            title= f"{c_nm} {selected_sgg} {selected_startYear}년 ~ {selected_endYear}년 사이 연 평균 4종 대기오염 물질 농도 추이 ",
            xaxis_title="연도",
            yaxis_title="농도(㎍/㎥)")
        st.plotly_chart(fig2)
    
    else:
        filtered_data = apData[(apData['측정소명']==selected_sgg) & (apData['year'] == selected_startYear) & (apData['month'] == selected_month)]
        filtered_data['day'] = filtered_data['측정일시'].dt.day

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_data['day'], y=filtered_data['미세먼지농도(㎍/㎥)'],
                    mode='lines', # Line plot만 그리기
                    name='미세먼지농도(㎍/㎥)'))
        fig.add_trace(go.Scatter(x=filtered_data['day'], y=filtered_data['초미세먼지농도(㎍/㎥)'],
                    mode='lines', # Line plot만 그리기
                    name='초미세먼지농도(㎍/㎥)'))
        # 그래프 레이아웃 설정 (제목 포함)
        fig.update_layout(
            title= f"{c_nm} {selected_sgg} {selected_startYear}년 {selected_month}월 (초)미세먼지 농도 추이 ",
            xaxis_title="일",
            yaxis_title="농도(㎍/㎥)",)

        
        # max_value = max(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 값 반환
        # max_value_2 = max(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 값 반환
        
        # max_index =np.argmax(filtered_data['미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환
        # max_index_2 =np.argmax(filtered_data['초미세먼지농도(㎍/㎥)']) # 미세먼지 농도 값이 제일 큰 index 반환

        # 그래프에 텍스트 추가
        fig.add_annotation(
            x=filtered_data['day'].iloc[max_index], y=filtered_data['미세먼지농도(㎍/㎥)'].iloc[max_index],
            text=f"최대:{round(max_value, 2)}(㎍/㎥)",  # 원하는 텍스트 입력
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-30)
        
        fig.add_annotation(
        x=filtered_data['day'].iloc[max_index_2], y=filtered_data['초미세먼지농도(㎍/㎥)'].iloc[max_index_2],
            text=f"최대:{round(max_value_2, 2)}(㎍/㎥)",  # 원하는 텍스트 입력
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-30)
        
        st.plotly_chart(fig)

        ### 기타 대기오염 그래프 (ppm)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=filtered_data['day'], y=filtered_data['이산화질소농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='이산화질소농도(ppm)'))
        fig2.add_trace(go.Scatter(x=filtered_data['day'], y=filtered_data['오존농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='오존농도(ppm)'))
        fig2.add_trace(go.Scatter(x=filtered_data['day'], y=filtered_data['일산화탄소농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='일산화탄소농도(ppm)'))
        fig2.add_trace(go.Scatter(x=filtered_data['day'], y=filtered_data['아황산가스농도(ppm)'],
                    mode='lines', # Line plot만 그리기
                    name='아황산가스농도(ppm)'))
        # 그래프 레이아웃 설정 (제목 포함)
        fig2.update_layout(
            title= f"{c_nm} {selected_sgg} {selected_startYear}년 {selected_month}월 4종 대기오염 물질 농도 추이 ",
            xaxis_title="일",
            yaxis_title="농도(㎍/㎥)")
        st.plotly_chart(fig2)


def regionAirPollution(apData):
    apData['month'] = apData['측정일시'].dt.month
    apData['year'] = apData['측정일시'].dt.year

    sgg_nm = sorted(apData['측정소명'].unique())
    year = sorted(apData['year'].unique())
    month = sorted(apData['month'].unique())

    c_nm = '중국' if '베이징' in sgg_nm else '한국'

    col1, col2, col3, col4  = st.columns(4)

    with col1:
        ouput_form = st.radio('구분', ['연도', '월'], horizontal=True)
    with col2:
        selected_ap = st.selectbox('대기 오염 물질', ['미세먼지', '초미세먼지', '이산화질소', '오존', '일산화탄소', '아황산가스'])
    with col3:
        selected_year = st.selectbox('연도 선택', year)
    with col4:
        if ouput_form == '월':
            selected_month = st.selectbox('월 선택', month)


    max_value = 0
    max_value_2 = 0
    
    max_index = 0
    max_index_2 = 0
    y = ''
    m = ''
    d = ''
    col5, col6 = st.columns(2)
        # 필터링 및 최대값 계산
    if ouput_form == '월':
        ap_list = ['미세먼지농도(㎍/㎥)', '초미세먼지농도(㎍/㎥)', '이산화질소농도(ppm)', '오존농도(ppm)', '일산화탄소농도(ppm)', '아황산가스농도(ppm)']

        sal = len(selected_ap)
        selected_ap_index = [i for i, a in enumerate(ap_list) if selected_ap[:sal] == a[:sal]][0]
        selected_ap_name = ap_list[selected_ap_index]
        
        start_index = selected_ap_name.find('(')
        ap_unit_name = selected_ap_name[start_index+1 :-1]
        
        filtered_data = apData[(apData['year'] == selected_year)&(apData['month'] == selected_month)]

        filtered_data = filtered_data.groupby(['측정소명'])[selected_ap_name].agg('mean').reset_index()
        
        max_value = round(max(filtered_data[selected_ap_name]),2) # 미세먼지 농도 값이 제일 큰 값 반환환
        max_index =np.argmax(filtered_data[selected_ap_name]) # 미세먼지 농도 값이 제일 큰 index 반환

        y = '**('+str(filtered_data['측정소명'].iloc[max_index])+')**'
    elif ouput_form == '연도':
        ap_list = ['미세먼지농도(㎍/㎥)', '초미세먼지농도(㎍/㎥)', '이산화질소농도(ppm)', '오존농도(ppm)', '일산화탄소농도(ppm)', '아황산가스농도(ppm)']

        sal = len(selected_ap)
        selected_ap_index = [i for i, a in enumerate(ap_list) if selected_ap[:sal] == a[:sal]][0]
        selected_ap_name = ap_list[selected_ap_index]
        
        start_index = selected_ap_name.find('(')
        ap_unit_name = selected_ap_name[start_index+1 :-1]
        filtered_data = apData[apData['year'] == selected_year]

        filtered_data = filtered_data.groupby(['측정소명'])[selected_ap_name].agg('mean').reset_index()

        max_value = round(max(filtered_data[selected_ap_name]),2) # 미세먼지 농도 값이 제일 큰 값 반환환
        max_index =np.argmax(filtered_data[selected_ap_name]) # 미세먼지 농도 값이 제일 큰 index 반환
        
        y = '**('+str(filtered_data['측정소명'].iloc[max_index])+')**'

    # 최대값을 맨 위에 표시
    with col5:
        st.metric(label=f'최대 미세먼지 농도\n {y} ', value=f'{round(max_value, 2)}(㎍/㎥)')

    if ouput_form == '연도':
        ap_list = ['미세먼지농도(㎍/㎥)', '초미세먼지농도(㎍/㎥)', '이산화질소농도(ppm)', '오존농도(ppm)', '일산화탄소농도(ppm)', '아황산가스농도(ppm)']

        sal = len(selected_ap)
        selected_ap_index = [i for i, a in enumerate(ap_list) if selected_ap[:sal] == a[:sal]][0]
        selected_ap_name = ap_list[selected_ap_index]
        
        start_index = selected_ap_name.find('(')
        ap_unit_name = selected_ap_name[start_index+1 :-1]
        filtered_data = apData[apData['year'] == selected_year]


        filtered_data = filtered_data.groupby(['측정소명'])[selected_ap_name].agg('mean').reset_index()

        fig = px.bar(filtered_data, x='측정소명', y=[selected_ap_name], title=f'{selected_year}년 {c_nm} 행정구역별 {selected_ap_name} 농도 평균')
        fig.update_layout(xaxis_title="행정구역", yaxis_title=ap_unit_name)

        max_value = round(max(filtered_data[selected_ap_name]),2) # 미세먼지 농도 값이 제일 큰 값 반환환
        max_index =np.argmax(filtered_data[selected_ap_name]) # 미세먼지 농도 값이 제일 큰 index 반환
        
        # 그래프에서 미세먼지가 가장 높은 곳에 텍스트 추가
        fig.add_annotation(
            x=filtered_data['측정소명'].iloc[max_index], y=filtered_data[selected_ap_name].iloc[max_index],
            text=f"최대:{max_value}{ap_unit_name}",  # 원하는 텍스트 입력
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40
        )

        st.plotly_chart(fig)
    else:
        ap_list = ['미세먼지농도(㎍/㎥)', '초미세먼지농도(㎍/㎥)', '이산화질소농도(ppm)', '오존농도(ppm)', '일산화탄소농도(ppm)', '아황산가스농도(ppm)']

        sal = len(selected_ap)
        selected_ap_index = [i for i, a in enumerate(ap_list) if selected_ap[:sal] == a[:sal]][0]
        selected_ap_name = ap_list[selected_ap_index]
        
        start_index = selected_ap_name.find('(')
        ap_unit_name = selected_ap_name[start_index+1 :-1]
        
        filtered_data = apData[(apData['year'] == selected_year)&(apData['month'] == selected_month)]

        filtered_data = filtered_data.groupby(['측정소명'])[selected_ap_name].agg('mean').reset_index()
        
        

        fig = px.bar(filtered_data, x='측정소명', y=[selected_ap_name], title=f'{selected_year}년 {selected_month}월 {c_nm} 행정구역별 {selected_ap_name} 농도 평균')
        fig.update_layout(xaxis_title="행정구역", yaxis_title=ap_unit_name)

        
        max_value = round(max(filtered_data[selected_ap_name]),2) # 미세먼지 농도 값이 제일 큰 값 반환환
        max_index =np.argmax(filtered_data[selected_ap_name]) # 미세먼지 농도 값이 제일 큰 index 반환
        
        # 그래프에서 미세먼지가 가장 높은 곳에 텍스트 추가
        fig.add_annotation(
            x=filtered_data['측정소명'].iloc[max_index], y=filtered_data[selected_ap_name].iloc[max_index],
            text=f"최대:{max_value}{ap_unit_name}",  # 원하는 텍스트 입력
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40
        )
        st.plotly_chart(fig)
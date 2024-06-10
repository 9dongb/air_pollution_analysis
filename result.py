import pandas as pd
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.font_manager as fm
import seaborn as sns
from plotly.subplots import make_subplots
from pingouin import ttest
import pingouin as pg

import plotly.graph_objects as go

def run_result(k_apData, c_apData):

    

    selected_analysis = st.sidebar.radio('분석 방법', ['1. 행정구역별 미세먼지 농도 차이', '2. 연도별 미세먼지 농도 차이', '3. 상관분석'])
    if selected_analysis == '2. 연도별 미세먼지 농도 차이':
        st.markdown('### 2. 연도별 미세먼지 농도 차이\n'
        '- 1월부터 12월까지 데이터가 모두 존재하는 최근 5년 간의 데이터(2019~2023)로 시각화 진행\n')

        fig_k = ap_analysis_by_year(k_apData, '한국')
        fig_c = ap_analysis_by_year(c_apData, '중국')
        st.pyplot(fig_k)
        st.pyplot(fig_c)
        
        st.markdown('### 분석 결과\n'
            '1. 2019년에 비해 2020년의 미세먼지 농도가 큰 폭으로 하락하였음.\n'
            '- 이는 코로나19가 유행하기 시작했던 2020년 초이기에 중국 공장 가동률이 낮았기 때문이라고 추측.\n'
            '- 따라서, 한국의 미세먼지 농도는 중국의 영향을 많이 받는다고 추측.')

    elif selected_analysis == '1. 행정구역별 미세먼지 농도 차이':
        
        st.markdown('## 1. 행정구역별 (초)미세먼지 농도 차이\n'
                    '- 한국의 전반적인 (초)미세먼지 데이터를 **분석**하고 **시각화**하고자 함\n'
                    '- 한국의 **시군구별**로 데이터를 분석하기엔 시군구의 수는 **260개**로 **현실적인 어려움**이 있음\n'
                    '- **행정구역별**로 분석을 진행할 시 **17개**로 한국의 전반적인 데이터를 **분석하기에 적합**하다고 판단\n'
                    '- 도 단위의 데이터는 **가장 인구가 많은 도시 1개**를 **대표 데이터**로 지정함\n')
        with st.expander('분석한 도시 정보') :
            st.markdown('- 서울특별시\n'
                        '- 대전광역시\n'
                        '- 인천광역시\n'
                        '- 광주광역시\n'
                        '- 대구광역시\n'
                        '- 부산광역시\n'
                        '- 울산광역시\n'
                        '- 강원도 - 원주(명륜동)\n'
                        '- 경상북도 - 포항\n'
                        '- 경기도 - 수원\n'
                        '- 전라북도 - 전주\n'
                        '- 전라남도 - 여수\n'
                        '-  충청북도 - 청주\n'
                        '- 충청남도 - 천안\n'
                        '- 경상남도 - 김해\n'
                        '- 제외 도시\n'
                        '   - 세종특별자치시(데이터 수집 불가)\n'

                        )

        fig1 = ap_analysis_by_sido(k_apData, '미세먼지농도(㎍/㎥)')
        fig2 = ap_analysis_by_sido(k_apData, '초미세먼지농도(㎍/㎥)')
        st.pyplot(fig1)
        st.pyplot(fig2)
        st.markdown('### 분석 결과\n'
                    '1. 전반적으로 **서쪽 지역**의 (초)미세먼지 농도가  **높다**는 것을 알 수 있음.\n'
                    '- **서쪽 지역**은 다른 지역보다 **중국과 비교적 가까운 곳**에 위치해 있기에\n '
                    '- 중국 미세먼지의 **영향을 받을 것으로 추측**'
                    '2. 하지만 경북(포항), 부산, 울산은 서쪽 지역 못지않게 (초)미세먼지 농도가 높음\n'
                    '- 이는 중국의 미세먼지 영향을 받기 보다는 다른 이유가 있을 것으로 추측\n')

    elif selected_analysis == '3. 상관분석':
        all_correlation(k_apData, c_apData, '상관분석')
        st.markdown('### 분석 결과\n'
            '1. 중국 상하이의 (초)미세먼지와 서쪽지역(인천, 경기도, 서울, 충청남도)의 미세먼지의 상관관계를 분석한 결과 강한 양의 상관관계를 보였음\n'
            '2. 중국 산둥성의 (초)미세먼지와 서쪽지역(인천, 경기도, 서울, 충청남도)의 미세먼지의 상관관계를 분석한 결과 강한 양의 상관관계를 보였음\n'
            '3. 중국 베이징의 미세먼지와 서쪽지역(인천, 경기도, 서울, 충청남도)의 미세먼지의 상관관계를 분석한 결과 강한 양의 상관관계를 보였음\n'
            '- 3-1. 하지만 중국 베이징의 초미세먼지와 서쪽지역(인천, 경기도, 서울, 충청남도)의 미세먼지의 상관관계를 분석한 결과 **인천**만 강한 양의 상관관계를 보였음\n')
    # elif selected_analysis =='4. 차이검정':
    #     differenceTest(k_apData, c_apData)

# def differenceTest(apData, c_apData):
    
#     st.write(f'### 한국과 중국의 (초)미세먼지 대기오염 물질 차이검정')
#     ap_list = sorted(apData.columns.to_list()[2:]) # 대기오염 물질 목록

#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         selected_year = st.radio('연도', [2019, 2020, 2021, 2022, 2023], horizontal=True)
#     with col2:
#         selected_china = st.radio('중국', ['베이징', '산둥성', '상하이'], horizontal=True)
#     with col3:
#         selected_ap = st.radio('대기오염 물질', ['미세먼지농도(㎍/㎥)','초미세먼지농도(㎍/㎥)'])
#     with col4:
#         selected_view = st.radio('보기 방식', ['간단히 보기', '자세히 보기'])

#     apData['year'] = apData['측정일시'].dt.year
#     c_apData['year'] = c_apData['측정일시'].dt.year

#     ap_unit_index = selected_ap.find('(')
#     ap = selected_ap[:ap_unit_index]
#     ap_unit = selected_ap[ap_unit_index:]

#     df = pd.DataFrame()
#     area_list = ['경기도', '충청남도', '인천', '서울']

#     for area in area_list:
#         filtered_data = apData[(apData['측정소명'] == area) & (apData['year'] == selected_year) ]
#         c_filtered_data = c_apData[(c_apData['측정소명'] == selected_china) & (c_apData['year'] == selected_year)]
        
#         ttest_result = ttest(filtered_data[selected_ap], c_filtered_data[selected_ap], paired=False)
        
#         p_value = round(ttest_result['p-val'].values[0], 2)

#         status_num = 0 if p_value > 0.05 else 1

#         status1 = ['이상', '미만'][status_num]
#         status2 = ['통계적으로 유의하지않다.','통계적으로 유의하다.'][status_num]
#         status3 = ['X','O'][status_num]
        
#         area_mean = filtered_data[selected_ap].mean().round(0)
#         area_df = pd.DataFrame({'행정구역':[area], f'{selected_ap} 평균': [area_mean], 'p-value':[p_value],'통계적 유의성':[status3]})
#         df = pd.concat([df, area_df])

#         if selected_view != '간단히 보기':
#             st.markdown(f'- **{area}**와(과) **{selected_china}**의 p-val 값이 0.05 {status1}으로 {selected_ap} 농도 차이는 **{status2}**')
#             with st.expander('- 자세히 보기') :
#                 st.dataframe(ttest_result, use_container_width=True)
#                 st.markdown(
#                             f'  - **{area}** {ap} 평균: {area_mean} {ap_unit}\n'
#                             f'  - **{selected_china}**  {ap}평균: { c_filtered_data[selected_ap].mean().round(0)}{ap_unit}\n'
#                             f'  - **p-value**: {p_value}')
#                             #f'- {area}와 {selected_china}의 {ap} 농도 차이 검정\n'

#     df = df.reset_index(drop=True)
#     df= df.sort_values('p-value')
#     df.index = df.index+1
#     if selected_view == '간단히 보기':
#         st.dataframe(df)

def all_correlation(apData, c_apData, selected_analysis):
    st.write(f'### 한국 서쪽지역과 중국의 (초)미세먼지 {selected_analysis}')

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_china = st.radio('중국', ['베이징', '산둥성', '상하이'], horizontal=True)
    with col2:
        selected_ap = st.radio('대기오염 물질', ['미세먼지농도(㎍/㎥)','초미세먼지농도(㎍/㎥)'])
    with col3:

        selected_view = st.radio('보기 방식', ['간단히 보기', '자세히 보기'])
    
    df = pd.DataFrame()

    area_list = apData['측정소명'].unique()
        
    
    area_list = ['경기도', '충청남도', '인천', '서울']

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

            
            st.write(f'### 피어슨 상관관계 계수 및 검정 ({area})')

            if (corr_r > 0.5):
                st.markdown(f'- 피어슨 상관계수는 **{corr_r}**이며, 중국 {selected_china}의 미세먼지 농도가 **증가**할수록 {area} 미세먼지 농도도 **증가**하는 경향성을 가진다.')
            elif (corr_r < -0.5):
                st.markdown(f'- 피어슨 상관계수는 {corr_r}이며, 중국 {selected_china}의 미세먼지 농도가 **증가**할수록 {area} 미세먼지 농도는 **감소**하는 경향성을 가진다.')
            else:
                st.markdown(f'- 피어슨 상관계수는 {corr_r}이며, 중국 {selected_china}의 미세먼지 농도와 {area} 미세먼지 농도의 **관계성**은 비교적 **적다.**')

            

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
        
        st.write(f'### 한국 서쪽지역과 {selected_china} {selected_ap} 상관 분석')
        df = df.sort_values('상관계수', ascending=False).reset_index(drop=True)
        df.index += 1
        st.dataframe(df)

def ap_analysis_by_year(k_apData, country):

    k_apData['year'] = k_apData['측정일시'].dt.year
    k_apData = k_apData[(k_apData['year']>=2019)&(k_apData['year']<=2023)]
    filtered_data = k_apData.groupby(['측정소명', 'year'])['미세먼지농도(㎍/㎥)'].agg('mean')
    filtered_data = filtered_data.reset_index()

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=filtered_data, x='측정소명', y='미세먼지농도(㎍/㎥)', hue='year', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title(f'연도별 미세먼지농도 차이 ({country})')
    #st.pyplot(fig)
    return fig
    

    
    
    

def ap_analysis_by_sido(k_apData, ap_name):

    # main_ap = ['미세먼지농도(㎍/㎥)', '초미세먼지농도(㎍/㎥)']
    # selected_ap = st.selectbox('대기 오염 물질', sorted(k_apData.columns.to_list()[2:8]))

    # 한글 폰트 설정
    path = "font/H2HDRM.TTF"
    fontprop = fm.FontProperties(fname=path, size=12)

    k_gpd = load_map('korea', k_apData)
    fig, ax = plt.subplots(figsize=(15, 15), ncols=3, nrows=2)

    r = 0
    for i, year in enumerate(zip(range(2019, 2025))):

        
        if i >= 3:
            r = 1
            i -= 3
        
        k_merge_1  = df_preprocessing(k_apData, k_gpd, year, ap_name)

        k_merge_1.plot(ax=ax[r][i], column='mean', cmap='Greens', legend=False, alpha=0.7, edgecolor='gray')

        ax[r][i].set_title(f' {year}년 평균 {ap_name}', fontproperties=fontprop)
    
    # ax[-1, -1].axis('off') # 마지막 칸 제거
    #st.pyplot(fig)
    return fig

def load_map(country, apData):
    y_gpd = gpd.read_file(f'data/map/{country}_administrative_division_all.gpkg') # 한국 지도 데이터 불러오기

    y_gpd=y_gpd.set_crs(epsg='5178', allow_override=True)            # set_crs(): 좌표계 정의 (kad.crs 확인 결과 초기엔 좌표계가 정의되어있지 않음 )
    y_gpd['center_point'] = y_gpd['geometry'].geometry.centroid      # 좌표의 중심점 컬럼 추가

    y_gpd['geometry'] = y_gpd['geometry'].to_crs(epsg=4326)          # to_crs(): 좌표계 변환기능 (epsg: 5178-> epsg: 4326)
    y_gpd['center_point'] = y_gpd['center_point'].to_crs(epsg=4326)

    y_gpd['lat'] = y_gpd['center_point'].map(lambda x: x.xy[1][0])   # 위도(latitude)
    y_gpd['lon'] = y_gpd['center_point'].map(lambda x: x.xy[0][0])   # 경도(longitude)

    y_gpd = y_gpd.rename(columns={"CTP_KOR_NM":"측정소명"})           # 컬럼명 변경
    
    apData_area = apData['측정소명'].tolist()
    y_gpd['측정소명'] = y_gpd['측정소명'].apply(lambda x: replace_name(x, apData_area))
    return y_gpd

def df_preprocessing(apData, y_gpd, year, selected_ap):
        apData['year'] = apData['측정일시'].dt.year
        apData['month'] = apData['측정일시'].dt.month

        filtered_data = apData[apData['year']==year]
        summary_df = filtered_data.groupby(['측정소명', 'year'])[selected_ap].agg(['mean', 'std', 'size'])

        merge_df = y_gpd.merge(summary_df, on='측정소명')

        return merge_df

def replace_name(x, apData_area):
    for area in apData_area:
        if area in x:
            return area
    return x




    
                           

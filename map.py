import pandas as pd
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.font_manager as fm



def replace_name(x, apData_area):
    for area in apData_area:
        if area in x:
            return area
    return x


def showMap(apData, c_apData):
    col1, col2, col3 = st.columns(3)

    k_gpd = gpd.read_file('data/map/korea_administrative_division_all.gpkg') # 한국 지도 데이터 불러오기

    k_gpd=k_gpd.set_crs(epsg='5178', allow_override=True)            # set_crs(): 좌표계 정의 (kad.crs 확인 결과 초기엔 좌표계가 정의되어있지 않음 )
    k_gpd['center_point'] = k_gpd['geometry'].geometry.centroid      # 좌표의 중심점 컬럼 추가

    k_gpd['geometry'] = k_gpd['geometry'].to_crs(epsg=4326)          # to_crs(): 좌표계 변환기능 (epsg: 5178-> epsg: 4326)
    k_gpd['center_point'] = k_gpd['center_point'].to_crs(epsg=4326)

    k_gpd['lat'] = k_gpd['center_point'].map(lambda x: x.xy[1][0])   # 위도(latitude)
    k_gpd['lon'] = k_gpd['center_point'].map(lambda x: x.xy[0][0])   # 경도(longitude)

    k_gpd = k_gpd.rename(columns={"CTP_KOR_NM":"측정소명"})           # 컬럼명 변경
    
    apData_area = apData['측정소명'].tolist()
    k_gpd['측정소명'] = k_gpd['측정소명'].apply(lambda x: replace_name(x, apData_area))
#######################################################################################
    c_gpd = gpd.read_file('data/map/china_administrative_division_all.gpkg') # 한국 지도 데이터 불러오기
    
    c_gpd=c_gpd.set_crs(epsg='5178', allow_override=True)            # set_crs(): 좌표계 정의 (kad.crs 확인 결과 초기엔 좌표계가 정의되어있지 않음 )
    
    c_gpd['center_point'] = c_gpd['geometry'].geometry.centroid      # 좌표의 중심점 컬럼 추가
    

    c_gpd['geometry'] = c_gpd['geometry'].to_crs(epsg=4326)          # to_crs(): 좌표계 변환기능 (epsg: 5178-> epsg: 4326)
    c_gpd['center_point'] = c_gpd['center_point'].to_crs(epsg=4326)

    c_gpd['lat'] = c_gpd['center_point'].map(lambda x: x.xy[1][0])   # 위도(latitude)
    c_gpd['lon'] = c_gpd['center_point'].map(lambda x: x.xy[0][0])   # 경도(longitude)

    c_gpd = c_gpd.rename(columns={"VARNAME_1":"측정소명"})           # 컬럼명 변경

    
    replace_dict = {'Běijīng': '베이징', 'Shànghǎi': '상하이', 'Shāndōng':'산둥성'} # 변경할 값들 dict으로 정의
    c_gpd['측정소명'] = c_gpd['측정소명'].replace(replace_dict)                     # 한국어로 변경
    apData['year'] = apData['측정일시'].dt.year
    apData['month'] = apData['측정일시'].dt.month

    c_apData['year'] = c_apData['측정일시'].dt.year
    c_apData['month'] = c_apData['측정일시'].dt.month
    
    with col1:
        selected_year = st.selectbox('연도 선택', apData['year'].unique())

    with col2:
        selected_month =  st.selectbox('월 선택', apData['month'].unique())
    
    with col3:
        selected_ap = st.selectbox('대기 오염 물질', sorted(apData.columns.to_list()[2:8]))
    
    c_filtered_data = c_apData[(c_apData['year']==selected_year)&(c_apData['month']==selected_month)]
    filtered_data = apData[(apData['year']==selected_year)&(apData['month']==selected_month)]

    summary_df = filtered_data.groupby(['측정소명', 'month'])[selected_ap].agg(['mean', 'std', 'size'])
    c_summary_df = c_filtered_data.groupby(['측정소명', 'month'])[selected_ap].agg(['mean', 'std', 'size'])

    merge_df = k_gpd.merge(summary_df, on='측정소명')
    c_merge_df = c_gpd.merge(c_summary_df, on='측정소명')
    
    ##################################################################################################

    
    geoMatplotlib(merge_df, selected_year, selected_month, selected_ap, k_gpd, c_gpd, c_merge_df)

    # st.write(merge_df[['측정소명', 'geometry', 'mean']])

def geoMatplotlib(merge_df, selected_year, selected_month, selected_ap, k_gpd, c_gpd, c_merge_df):
    # 한글 폰트 설정
    path = "font/H2HDRM.TTF"
    fontprop = fm.FontProperties(fname=path, size=12)

    selected_view = st.radio('보기 방식', ['전체 지도 표시', '선택된 지도만 표시', '자세히 보기'])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))  # 1행 2열의 서브플롯 생성

    
    for ax, g, m, country in zip(axes, [k_gpd, c_gpd], [merge_df, c_merge_df], ['한국', '중국']):
        if selected_view=='자세히 보기':

            fig, ax = plt.subplots(figsize=(10, 5))

            g.plot(ax=ax, color='white', edgecolor='black')

            m.plot(ax=ax, column='mean', cmap='Greens', legend=False, alpha=0.9, edgecolor='gray')

            path_col = ax.collections[0]
            cb = fig.colorbar(path_col, ax=ax, shrink=0.5)

            ap_unit = selected_ap[selected_ap.find('(')+1:-1]

            for i, row in m.iterrows():
                ax.annotate(row['측정소명'], xy=(row['lon'], row['lat']), xytext=(-7, 2),
                            textcoords='offset points', fontsize=5, color='black', fontproperties=fontprop)
                ax.annotate(f'{round(row["mean"],2)}{ap_unit}', xy=(row['lon'], row['lat']), xytext=(-7, -7),
                            textcoords='offset points', fontsize=3, color='black', fontproperties=fontprop)

            ax.set_title(f'{country} {selected_year}년 {selected_month}월 평균 {selected_ap}', fontproperties=fontprop)    
            
            st.pyplot(fig)
        
    

        if selected_view=='전체 지도 표시':
            g.plot(ax=ax, color='white', edgecolor='black')
        
        m.plot(ax=ax, column='mean', cmap='Greens', legend=False, alpha=0.9, edgecolor='gray')

        path_col = ax.collections[0]
        cb = fig.colorbar(path_col, ax=ax, shrink=0.5)

        ap_unit = selected_ap[selected_ap.find('(')+1:-1]

        for i, row in m.iterrows():
            ax.annotate(row['측정소명'], xy=(row['lon'], row['lat']), xytext=(-7, 2),
                        textcoords='offset points', fontsize=8, color='black', fontproperties=fontprop)
            ax.annotate(f'{round(row["mean"],2)}{ap_unit}', xy=(row['lon'], row['lat']), xytext=(-7, -7),
                        textcoords='offset points', fontsize=6, color='black', fontproperties=fontprop)
        ax.set_title(f'{country} {selected_year}년 {selected_month}월 평균 {selected_ap}', fontproperties=fontprop)
        
    
    plt.tight_layout()  # 서브플롯 간 간격 조절
    st.pyplot(fig)
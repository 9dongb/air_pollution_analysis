import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from viz import showViz
from statistic import showStat
from map import showMap

def run_eda_home(k_apData, c_apData):
    st.markdown("## 탐색적 자료 분석\n")
    
    # 2. horizontal menu
    selected = option_menu(None, ["홈", "데이터 탐색", "통계", '지도'], 
        icons=['house', 'bi bi-bar-chart-line', "bi bi-graph-up", 'map'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    
    if selected == '홈':
        st.markdown("### 데이터 탐색\n"
                    "- 서울 대기오염 농도\n"
                    "   - 조건별 평균 대기오염 농도 (표, 그래프)\n"
                    "   - 자치구별 대기오염 농도 (표, 그래프)\n"
                    "- 중국 대기오염 농도\n"
                    "   - 조건별 평균 대기오염 농도 (표, 그래프)\n"
                    "   - 자치구별 대기오염 농도 (표, 그래프)\n"
                    "### 통계\n"
                    "- 한국과 중국의 대기오염 물질 차이 검정\n"
                    "- 한국과 중국의 대기오염 물질 상관 분석\n"
                    "### 지도\n"
                    "- 서울의 대기오염 농도 지도 시각화\n"
                    "- 중국의 대기오염 농도 지도 시각화\n"
                    )
        
    elif selected == '데이터 탐색':
        showViz(k_apData, c_apData)
    elif selected == '통계':
        showStat(k_apData, c_apData)
    elif selected == '지도':
        showMap(k_apData, c_apData)
    else:
        pass
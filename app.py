import streamlit as st
from streamlit_option_menu import option_menu
from home import run_home
from eda import run_eda_home
from utils import load_ap_data
from ml import run_ml
from info import air_pollution
from result import run_result
def main():
    
    k_apData = load_ap_data()
    c_apData = load_ap_data('china')
    

    with st.sidebar:
        st.write('# 대기오염 데이터 분석')
        selected = option_menu("목록", ["메인", "대기 오염 물질 정보", "탐색적 자료분석", "미래 데이터 예측", "보고서"],
                            icons=['house', 'bi bi-radioactive', 'file-bar-graph', 'bi bi-robot', 'bi bi-body-text'],
                            menu_icon="app-indicator", default_index=0,
                            styles={
            "container": {"padding": "4!important", "background-color": "#fafafa"},
            "icon": {"color": "black", "font-size": "25px"},
            "nav-link": {"font-size": "12x", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
            "nav-link-selected": {"background-color": "#08c7b4"},
        })
    
    if selected =='메인':
        run_home()
    elif selected == '탐색적 자료분석':
        run_eda_home(k_apData, c_apData)
    elif selected == '미래 데이터 예측':
        run_ml(k_apData, c_apData)
    elif selected == '대기 오염 물질 정보':
        air_pollution()
    elif selected == '보고서':
        run_result(k_apData, c_apData)
    else:
        st.warning('Error 발생')


# 직접 실행할 때 main함수 실행
if __name__ == '__main__':
    main()
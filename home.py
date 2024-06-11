import streamlit as st


def run_home():
    # st.title('환경성 질환 경보 시스템 혹은')
    st.title('한국과 중국의 대기오염 데이터 분석')
    st.image('img/air_pollution_1.png')
    st.markdown('''
                ### 개요
                - 최근 몇 년간 한국과 중국에서는 미세먼지가 **심각한 환경 문제**로 대두됨
                - 미세먼지는 건강에 해로울 뿐만 아니라 일상 생활에도 큰 영향을 미침
                - 본 분석에서는 **한국과 중국의 대기오염 데이터를 비교**하여 그 **관계성을 분석**하고,
                - 중국의 대기오염 물질들이 **실제로 한국의 대기오염 물질에 영향**을 얼마나 미치는지 파악하기 위함''')
    
    st.markdown('''
                ### 분석 목적
                - 한국과 중국의 대기오염물질들의 농도 비교
                - 한국과 중국의 대기오염물질들의 예측
                - 한국과 중국의 (초)미세먼지 농도 관계성 분석                
                ''')
    
    st.markdown('### 활용 데이터\n')

    with st.expander('자세히 보기') :
        st.markdown(
                    '- 공공데이터 포털(https://www.data.go.kr)\n'
                    '   - 국민건강보험공단_환경성질환(비염) 의료이용정보 2006~2022\n'
                    '   - 국민건강보험공단_환경성질환(아토피) 의료이용정보 2006~2022\n'
                    '   - 국민건강보험공단_환경성질환(천식) 의료이용정보 2006~2022\n'

                    '- 서울 열린 데이터 광장(https://data.seoul.go.kr)\n'
                    '   - 서울시 일별 평균 대기오염도 정보 2014 ~ 2023\n'
                    '- 대기질 이력 데이터 플랫폼(https://aqicn.org)\n'
                    '   - 한국 서울 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 인천 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 대전 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 대구 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 부산 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 울산 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 광주 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 경기(수원) 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 경북(포항) 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 경남(김해) 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 충북(청주) 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 충남(천안) 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 전북(전주) 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 전남(여수) 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 강원(원주-명륜동) 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 한국 제주(제주-이도동) 대기오염농도 데이터 201401 ~ 202405\n'
                                  
                    '   - 중국 베이징 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 중국 지난시 대기오염농도 데이터 201401 ~ 202405\n'
                    '   - 중국 상하이 대기오염농도 데이터 201401 ~ 202405\n'
                    '- GEOSERVICE (https://www.geoservice.co.kr)\n'
                    '   - 한국 행정구역 shp 파일\n'
                    '- GADM maps and data (https://gadm.org)\n'
                    '   - 중국 행정구역 shp 파일\n')


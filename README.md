# 🌫️한국과 중국의 미세먼지 데이터 분석

### 📅 개발기간
- 2024.05.28 ~ 2024.06.11

## 1. 개요
- 최근 몇 년간 한국과 중국에서는 미세먼지가 심각한 환경 문제로 대두됨
- 미세먼지는 건강에 해로울 뿐만 아니라 일상 생활에도 큰 영향을 미침
- 본 서비스는 사용자에게 한국과 중국의 대기오염 데이터를 비교하여 그 관계성을 분석할 수 있는 기능을 제공
- 중국의 대기오염 물질들이 실제로 한국의 대기오염 물질에 영향을 얼마나 미치는지 분석하기 위함
- **사용자**가 직접 **원하는 데이터**와 기간을 **선택해 분석**할 수 있는 것이 특징
  
## 2. 주요 기능
- pandas를 통해 **데이터 전처리** 및 **자동화**
- 한국과 중국의 대기오염물질의 농도를 **직접 분석**해볼 수 있는 기능 제공
- prophet을 이용해 한국과 중국의 대기오염물질의 **예측** 기능 제공
- **t검정**으로 한국과 중국의 (초)미세먼지 농도를 통계적으로 검정
- **피어슨 상관분석**으로 한국과 중국의 (초)미세먼지 농도 관계성 분석

## 3. 동작 화면
■ 메인 페이지
- 개요와 목적, 활용 데이터 확인 가능 
<img src="https://github.com/user-attachments/assets/d33f0d52-f415-4223-b2f1-76f7e0008a6e" width="50%" height="50%"/>

■ 탐색적 자료분석 - 홈

<img src="https://github.com/user-attachments/assets/2e9c2953-2023-4e6f-92b6-5de1a0ab9940" width="50%" height="50%"/>

- 각 탭에 대한 설명 확인 가능

■ 탐색적 자료분석 - 데이터 탐색

<img src="https://github.com/user-attachments/assets/381e190e-0f55-4f65-b43d-a2a42e0014d4" width="50%" height="50%"/>
<img src="https://github.com/user-attachments/assets/153004ed-f531-490a-bfef-3f36d3d9a061" width="50%" height="50%"/>

- 분석에 사용한 한국과 중국 대기오염 데이터를 행정구역, 연도, 월 별로 조회할 수 있는 기능 구현
- 조회된 데이터를 다운로드할 수 있는 기능 구현

■ 탐색적 자료분석 - 통계

<img src="https://github.com/user-attachments/assets/52bc531c-783f-41fb-90fb-4410882ee866" width="50%" height="50%"/>

- 두 국가의 행정구역별 데이터를 t-test 검정과 상관 분석을 해볼 수 있는 기능 구현

■ 탐색적 자료분석 - 지도

<img src="https://github.com/user-attachments/assets/c34a6f0d-9968-4ef8-9186-d8e4bb2ef68d" width="50%" height="50%"/>

- QGIS를 이용해 지도와 대기오염 수치를 비교해 확인할 수 있는 기능 구현 

■ 미래 데이터 예측

<img src="https://github.com/user-attachments/assets/e5779c32-c2c3-48dc-80cb-9fc5d706fca8" width="50%" height="50%"/>

- Prophet을 이용해 미래 데이터를 원하는 일 수 만큼 예측할 수 있는 기능 구현


■ 보고서

<img src="https://github.com/user-attachments/assets/1f0221a8-c4f7-4aab-871b-7fb79a29c622" width="50%" height="50%"/>
<img src="https://github.com/user-attachments/assets/83e020e2-7037-45ba-9d16-b30c69f9edbd" width="50%" height="50%"/>

- 전반적으로 서쪽 지역의 (초)미세먼지 농도가 높다는 것을 알 수 있음
- 서쪽 지역은 다른 지역보다 중국과 비교적 가까운 곳에 위치해 있기에 중국 미세먼지의 영향을 받을 것으로 추측. 
- 하지만 경북(포항), 부산, 울산은 서쪽 지역 못지않게 (초)미세먼지 농도가 높음
- 이는 중국의 미세먼지 영향을 받기 보다는 다른 이유가 있을 것으로 추측



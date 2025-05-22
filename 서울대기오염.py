import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error


import matplotlib.pyplot as plt  # EDA 시각화용
import seaborn as sns            # EDA 시각화용

# 1. 📂 데이터 불러오기 및 전처리
# [1-1] 데이터프레임으로 불러오기
df = pd.read_csv('서울대기오염_2019.csv')
df.head()
# [1-2] 분석변수만 추출 및 컬럼명 변경: date, district, pm10, pm25
selected_columns = ['날짜','측정소명','미세먼지','초미세먼지']
df_selected = df[selected_columns]
df_selected = df_selected.rename(columns={
    '날짜': 'date',
    '측정소명': 'district',
    '미세먼지': 'pm10',
    '초미세먼지': 'pm25'
})
# 데이터 확인
# 1) 데이터 정보 확인
df_selected.info()
#  칼럼 형태 확인
df_selected['date'].unique() #'전체', '2019-12-31', '2019-12-30'...
df_selected['district'].unique() #'평균', '강남구', '강동구', '강북구'...

# [1-4] 자료형 변환: 문자형 → 날짜형, 실수형 등
le = LabelEncoder()
df_selected['district_code'] = le.fit_transform(df_selected['district']).astype(float)
print(df_selected.head())

#서울시평균값과 각 disctrict 데이터 분리
df_mean = df_selected[df_selected['district']=='평균'].reset_index(drop=True)
df_loc  = df_selected[df_selected['district']!='평균'].reset_index(drop=True)


# [1-3] 결측치 확인 및 제거
# 1.결측치 개수 확인
# 1) 칼럼별 결측치 개수
df_loc.isnull().sum()

# 2) 칼럼별 결측치 비율
missing_pct = df_selected.isna().mean() * 100
missing_pct[missing_pct > 0].sort_values(ascending=False).to_frame('Missing(%)')


# 2.결측치 전략 수립
# 1) season-지역에 따른 pm10, pm25 패턴이 있을거라 판단됨
# 날짜를 datetime 으로
df_loc['date'] = pd.to_datetime(df_loc['date'])

# month, season 파생
df_loc['month']= df_loc['date'].dt.month
df_loc['day'] = df_loc['date'].dt.day
print(df_loc)

season_map = {
    12:'winter', 1:'winter', 2:'winter',
     3:'spring',   4:'spring',   5:'spring',
     6:'summer', 7:'summer', 8:'summer',
     9:'autumn',10:'autumn',11:'autumn'
}
df_loc['season'] = df_loc['month'].map(season_map)

# season-지역에 따른 pm10 패턴을 시각화로 확인해보기
# 측정소·월별 중앙값 테이블
pivot_pm10 = df_loc.pivot_table(
    values='pm10',               # 또는 'pm25'
    index='district', columns='month',
    aggfunc='median'
)
print(pivot_pm10.head())   # 행=disctrict, 열=1~12월

plt.figure(figsize=(14,6))
sns.heatmap(pivot_pm10, cmap='YlOrRd', linewidths=.5)
plt.rc('font', family='NanumBarunGothic')
plt.title('측정소별 월간 pm10 중앙값(㎍/㎥)')
plt.xlabel('month'); plt.ylabel('disctrict')
plt.show()
# ==> season-측정소에 따라 pm10 농도 차이가 있음을 확인

# 결측치 처리
# 날짜를 datetime 으로
df_loc['date'] = pd.to_datetime(df_loc['date'])

# 1) 미세먼지·초미세먼지: disctrict+month 그룹 중앙값 ->농도 분포가 비슷함
for col in ['pm10','pm25']:
    df_loc[col] = df_loc.groupby(['district', df_loc['date'].dt.month])[col].transform(lambda x: x.fillna(x.median()))

# 4. 결측치 완료 여부 확인->안된 행들이 있음.
df_loc.isnull().sum()
#결측치 처리가 안된 행들 확인
rows_nan_pm = df_loc[df_loc[['pm10','pm25']].isnull().any(axis=1)]
print(rows_nan_pm[['date','district','pm10','pm25']])
#==>9월 한달간 관악구의 경우 측정이 안됨.

# 4-1 관악구의 9월 관측값은 서울시 전체 평균으로 대체
for col in ['pm10','pm25']:
    df_loc[col] = df_loc[col].fillna(df_loc[col].mean())
# [3-1] 최종 분석 대상 데이터 확인
print(df_loc.head())

# [3-2] '201906_output.csv'로 저장 (GitHub에 업로드 or 구글 드라이브 공유)
#df_loc.to_csv('201906_output.csv', index=False)

# [4-1] 전체 데이터 기준 PM10 평균
year_mean = df_loc['pm10'].mean()
print(year_mean)
# 분석결과 작성
# 전체 데이터 기준 PM10 평균은 41.4로 나쁨 수준을 보임

# [5-1] PM10 최댓값이 발생한 날짜, 구 출력
pm10_max = df_loc['pm10'].max()
max_rows = df_loc[df_loc['pm10'] == pm10_max][['date', 'district', 'pm10']]
print(max_rows)
# 분석결과 작성
# 2019-03-05, 강북구, 228

# [6-1] 각 구별 pm10 평균 계산
# 구(측정소)별 PM10(미세먼지) 평균 계산
pm10_by_district = (
    df_loc.groupby('district', as_index=False)['pm10']
          .mean()
          .rename(columns={'pm10': 'avg_pm10'})
          .sort_values('avg_pm10', ascending=False)   # 높은 곳부터 보고 싶다면
)
# [6-2] 상위 5개 구만 출력 (컬럼: district, avg_pm10)
print(pm10_by_district.head())     

# 분석결과 작성
#  district   avg_pm10
#18      양천구  47.657534
#4       관악구  47.136045
#3       강서구  46.517808
#2       강북구  44.950685
#15      성동구  44.838356

# [7-1] 계절별 평균 pm10, pm25 동시 출력
season_avg = (
    df_loc.groupby('season', as_index=False)[['pm10', 'pm25']]
          .mean()
          .rename(columns={'pm10':'avg_pm10', 'pm25':'avg_pm25'})
)
# [7-2] 평균값 기준 오름차순 정렬 (컬럼: season, avg_pm10, avg_pm25)
season_avg = season_avg.sort_values('avg_pm10', ascending=True)
# 분석결과 작성
#   season   avg_pm10   avg_pm25
#2  summer  26.286304  18.132391
#0  autumn  30.861387  15.654524
#1  spring  54.087826  31.559565
#3  winter  54.624222  33.620889
#==>봄, 겨울에 미세먼지 농도가 높다는 것을 알 수 있다.

# [8-1] pm10 값을 기준으로 등급 분류 (good/normal/bad/worse)
# 1) 구간·라벨 정의 
pm10_bins    = [-np.inf, 30, 80, 150, np.inf]            # 좋음·보통·나쁨·매우나쁨
grade_labels = ['good', 'normal', 'bad', 'worse']

df_loc['pm_grade'] = pd.cut(
    df_loc['pm10'],
    bins=pm10_bins,
    labels=grade_labels
)
# [8-2] 전체 데이터 기준 등급별 빈도, 비율 계산 (컬럼: pm_grade, n, pct)
grade_counts=df_loc['pm_grade'].value_counts().reset_index()
grade_counts.columns=['pm_grade','n']
grade_counts['pct']=grade_counts['n'] / grade_counts['n'].sum() * 100

print(grade_counts)
# 분석결과 작성
# 보통과 좋음이 가장 많으나, 유독 안좋은 날들이 있는것으로 예측됨

# [9-1] 구별 등급 분포 중 'good' 빈도와 전체 대비 비율 계산
total_per_dist = (
    df_loc.groupby('district')
          .size()
          .rename('total_n')
)

good_per_dist = (
    df_loc[df_loc['pm_grade'] == 'good']
      .groupby('district')
      .size()
      .rename('n_good')
)

good_ratio = (
    pd.concat([total_per_dist, good_per_dist], axis=1)
      .assign(
          n_good=lambda d: d['n_good'].astype(int),
          pct_good=lambda d: (d['n_good'] / d['total_n'] * 100).round(2)
      )
      .reset_index()
      .sort_values('pct_good', ascending=False)
)
# [9-2] 비율(pct) 기준 내림차순 정렬 후 상위 5개 구만 출력 (컬럼: district, n, pct)
print(good_ratio.head())
# 분석결과 작성
#   district  total_n  n_good  pct_good
#20      용산구      365     198     54.25
#24      중랑구      365     185     50.68
#10     동대문구      365     180     49.32
#23       중구      365     169     46.30
#22      종로구      365     163     44.66
#==>용산의 경우 남산 등 녹지 면적이 넓고 공단이 없어서 미제먼지가 좋은 것으로 판단됨. 녹지 면적이 넓거나, 강변 주변 도시들의 공기가 좋음

# [10-1] x축: date, y축: pm10 (선그래프)
# 1) 날짜 컬럼을 datetime, 2019년 자료만 필터
daily_2019 = (
    df_loc.groupby('date')['pm10']
    .mean()
    .reset_index()
)

# 2) 선그래프
plt.figure(figsize=(12,4))
sns.lineplot(data=daily_2019, x='date', y='pm10', linewidth=1.2)
plt.title('Daily Trend of PM10 in Seoul, 2019')
plt.xlabel('Date')
plt.ylabel('PM10 (µg/m³)')
plt.tight_layout()
plt.show()
# [10-2] 제목: 'Daily Trend of PM10 in Seoul, 2019'
# 분석결과 작성
#2019년 1월과 3월에 특히나 미세먼지가 매우 높았다는 것을 알수 있음


# [11-1] x축: season, y축: pct, fill: pm_grade (막대그래프 - seaborn barplot)
print(df_loc.head())

# 1) 계절·등급별 빈도 → 비율 계산
season_grade = (
    df_loc.groupby(['season', 'pm_grade'])
          .size()
          .reset_index(name='n')
)
season_grade['pct'] = season_grade.groupby('season')['n'].transform(lambda x: x/x.sum()*100)
print(season_grade)

# 2) 시각화 
season_order = ['봄', '여름', '가을', '겨울']
grade_order  = ['good', 'normal', 'bad', 'worse']
palette = {'good':'#4CAF50','normal':'#FFC107','bad':'#F44336','worse':'#9C27B0'}

plt.figure(figsize=(8,5))
sns.barplot(
    data=season_grade,
    x='season', y='pct', hue='pm_grade',
    order=season_order, hue_order=grade_order,
    palette=palette
)
plt.title('Seasonal Distribution of PM10 Grades in Seoul, 2019')
plt.xlabel('Season'); plt.ylabel('Percentage (%)')
plt.legend(title='PM10 grade', loc='upper right')
plt.tight_layout()
plt.show()
# [11-2] 범례: good, normal, bad, worse
# [11-3] 제목: 'Seasonal Distribution of PM10 Grades in Seoul, 2019'
# 분석 결과 작성

# 전처리 완료 파일 csv 파일로 저장
df_loc.to_csv('201906_output.csv', index=False)
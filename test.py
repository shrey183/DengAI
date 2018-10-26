# Empty File to be filled later. 
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

train = data1.loc[(data1.city == 'sj') & (data1.year == 1990),['precipitation_amt_mm','reanalysis_relative_humidity_percent'
                                                            ,'station_diur_temp_rng_c','station_precip_mm','total_cases']]

compare = data1.loc[(data1.city == 'sj') & (data1.year == 1991),['precipitation_amt_mm','reanalysis_relative_humidity_percent'
                                                            ,'station_diur_temp_rng_c','station_precip_mm','total_cases']]

y = train.total_cases
#here are the few features that can be interesting
visual_features = ['precipitation_amt_mm','reanalysis_relative_humidity_percent',
                   'station_diur_temp_rng_c','station_precip_mm','total_cases'] 
X = train[visual_features]
X.describe()
Xbis = compare[visual_features]

#now onto prediction with a given model
dengue_model = DecisionTreeRegressor(random_state=1)
dengue_model.fit(X,y)
#print(dengue_model.predict(Xbis))
#print(compare.total_cases)

train1 = data1.loc[(data1.city == 'sj') & (data1.year < 1993),['precipitation_amt_mm','reanalysis_relative_humidity_percent'
                                                            ,'station_diur_temp_rng_c','station_precip_mm','total_cases']]

train1 = train1.loc[train1.precipitation_amt_mm.notnull()]
y1 = train1.total_cases
X1 = train1[visual_features]
compare1 = data1.loc[(data1.city == 'sj') & (data1.year == 1993),['precipitation_amt_mm','reanalysis_relative_humidity_percent'
                                                            ,'station_diur_temp_rng_c','station_precip_mm','total_cases']]
compare1 = compare1.loc[compare1.precipitation_amt_mm.notnull()]
X1bis = compare1[visual_features]
#print(compare1)
dengue_model1 = DecisionTreeRegressor(random_state=1)
dengue_model1.fit(X1,y1)
#print(dengue_model1.predict(X1bis))
#print(compare1.total_cases)
train2 = data1.loc[(data1.city == 'sj') & (data1.year < 2008),['precipitation_amt_mm','reanalysis_relative_humidity_percent'
                                                            ,'station_diur_temp_rng_c','station_precip_mm','total_cases']]

train2 = train2.loc[train2.precipitation_amt_mm.notnull()]
y2 = train2.total_cases
X2 = train2[visual_features]
compare2 = data1.loc[(data1.city == 'sj') & (data1.year == 2008),['precipitation_amt_mm','reanalysis_relative_humidity_percent'
                                                            ,'station_diur_temp_rng_c','station_precip_mm','total_cases']]
compare2 = compare2.loc[compare2.precipitation_amt_mm.notnull()]
X2bis = compare1[visual_features]

dengue_model2 = DecisionTreeRegressor(random_state=1)
dengue_model2.fit(X2,y2)
print(dengue_model2.predict(Xbis))
print(compare.total_cases)

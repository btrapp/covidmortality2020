# Source

This project is shared on gitub at [https://github.com/btrapp/covidmortality2020](https://github.com/btrapp/covidmortality2020) 
please feel free to submit pull requests to improve it or clone it to continue your own work.
---


```python
%matplotlib inline
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import xgboost as xg 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.linear_model import LinearRegression
# disable chained assignments
pd.options.mode.chained_assignment = None 
```

---
# Data Sources 

For the source of data, lets use the "Weekly Counts of Deaths by State and Select Causes" data from the CDC.

2019-2020: [https://data.cdc.gov/NCHS/Weekly-Counts-of-Deaths-by-State-and-Select-Causes/muzy-jte6]

2014-2018: [https://data.cdc.gov/NCHS/Weekly-Counts-of-Deaths-by-State-and-Select-Causes/3yf8-kanr]

Download these two datasets as CSVs and name them death_2020.csv and death_2018.csv.  A copy from when I first ran this on Nov 28 2020 is provided in the github repo.


In order to reduce ambiguity, lets ONLY use the "All Cause" column from this dataset, which tracks deaths per
 week from all causes, so we don't have to worry about if the underlying cause of death was attributed correctly
 or not.  Let's just see if we can compare total death rates in 2020 to the historical averages of 2014-2019 for the same periods.
 
Also, fun fact, the dataset contains deaths by jurisdiction for each state, plus summary rows at the end of the dataset for the entire US.  So don't just sum up all the data by year/week or you'll double count.  Google says 2.8 million people died in 2018 so there's a quick sanity checkpoint.





```python
#Read the 2019-2020 file
df2020all = pd.read_csv('death_2020.csv'); 
df2020all.head()

#Drop all columns except Juristiction, Year, Week, and All Cause
df2020=df2020all.iloc[:, [0, 1, 2, 4]] 
print(df2020[:5])
print(df2020[-5:])
```

      Jurisdiction of Occurrence  MMWR Year  MMWR Week  All Cause
    0                    Alabama       2019          1     1077.0
    1                    Alabama       2019          2     1090.0
    2                    Alabama       2019          3     1114.0
    3                    Alabama       2019          4     1063.0
    4                    Alabama       2019          5     1095.0
         Jurisdiction of Occurrence  MMWR Year  MMWR Week  All Cause
    5286              United States       2020         42    56233.0
    5287              United States       2020         43    56674.0
    5288              United States       2020         44    55477.0
    5289              United States       2020         45    53367.0
    5290              United States       2020         46    42566.0



```python
#Read the 2014-2018 file
df2018all = pd.read_csv('death_2018.csv'); 
df2018all.head()

#Drop all columns except Juristiction, Year, Week, and All Cause
df2018=df2018all.iloc[:, [0, 1, 2, 4]] 
print(df2018[:5])
print(df2018[-5:])

```

      Jurisdiction of Occurrence  MMWR Year  MMWR Week  All  Cause
    0                    Alabama       2014          1         355
    1                    Alabama       2014          2         872
    2                    Alabama       2014          3        1044
    3                    Alabama       2014          4        1022
    4                    Alabama       2014          5        1040
          Jurisdiction of Occurrence  MMWR Year  MMWR Week  All  Cause
    14089              United States       2018         48       55210
    14090              United States       2018         49       56095
    14091              United States       2018         50       56530
    14092              United States       2018         51       56689
    14093              United States       2018         52       56163


---
# Data Cleanup

- One annoying problem is that the "All Cause" column is "All Cause" in one dataset, and "All  Cause" (two spaces) in the other.
- Also, the dataset contains rows for state-level data plus a US-level summary.  Keep only the US level summary rows.
- The first and last points in the dataset (for just US-level) are outliers and should be removed.



```python
#In order to be able to merge them cleanly, lets set the columns to a more helpful set of consistent names.
df2020.columns=['Jurisdiction','Year','Week', 'Deaths']
df2018.columns=['Jurisdiction','Year','Week', 'Deaths']

#Merge the dataframes together
dfAllRaw = pd.concat([df2018,df2020])
# More cleanup - We don't need the data broken out by each state, there's a summary row for ALL states
#  called just 'United States'.  Select just those rows.
dfUsRaw = dfAllRaw[dfAllRaw['Jurisdiction'] == 'United States'] 

#The first and last records in the final dataset are suspiciously low, and look outlier-y.  Lets remove them
dfUs = dfUsRaw.iloc[1:].iloc[:-1]

#Check our work
print(dfUs[:5])
print(dfUs[-5:])
```

            Jurisdiction  Year  Week   Deaths
    13834  United States  2014     2  55715.0
    13835  United States  2014     3  54681.0
    13836  United States  2014     4  54175.0
    13837  United States  2014     5  54049.0
    13838  United States  2014     6  53492.0
           Jurisdiction  Year  Week   Deaths
    5285  United States  2020    41  57997.0
    5286  United States  2020    42  56233.0
    5287  United States  2020    43  56674.0
    5288  United States  2020    44  55477.0
    5289  United States  2020    45  53367.0



```python
#Feature engineering:
#Add a helpful column that's the year and week joined together with short weeks like 4 padded to '04'
yr = dfUs['Year'].astype(str)+'-'+dfUs['Week'].astype(str).str.pad(2,fillchar='0')
dfUs['YYYY-WW'] = yr

#We also need a time based component that's numeric for use in a regression and to make plotting easier.
#Encode Year-Week as a fraction of a year:
dfUs['YearValue']=dfUs['Year']+ ((dfUs['Week']-1)/52.0)

print(dfUs[:5])
print(dfUs[-5:])

```

            Jurisdiction  Year  Week   Deaths  YYYY-WW    YearValue
    13834  United States  2014     2  55715.0  2014-02  2014.019231
    13835  United States  2014     3  54681.0  2014-03  2014.038462
    13836  United States  2014     4  54175.0  2014-04  2014.057692
    13837  United States  2014     5  54049.0  2014-05  2014.076923
    13838  United States  2014     6  53492.0  2014-06  2014.096154
           Jurisdiction  Year  Week   Deaths  YYYY-WW    YearValue
    5285  United States  2020    41  57997.0  2020-41  2020.769231
    5286  United States  2020    42  56233.0  2020-42  2020.788462
    5287  United States  2020    43  56674.0  2020-43  2020.807692
    5288  United States  2020    44  55477.0  2020-44  2020.826923
    5289  United States  2020    45  53367.0  2020-45  2020.846154



```python
#Just a sanity check - should be about 2.8 million 
totalDeathsIn2018 = dfUs.loc[dfUs['Year']==2018]['Deaths'].sum();
print(totalDeathsIn2018)

#And visualise the data:
dfUs.plot(title='US Mortality by Week',x='YearValue',y=['Deaths'],figsize=(12,8),ylim=(40000,80000))

```

    2839076.0





    <AxesSubplot:title={'center':'US Mortality by Week'}, xlabel='YearValue'>




    
![png](output_8_2.png)
    


---
# What's a "normal" number of people that die by week?

Well - there is a *strong* cyclical component there, right?  Any analysis we do had better take that into account. 

Lets take all the data EXCEPT for 2020, and figure out what the number of people that die each week
by averaging the mortality by week for 2014-2019:


```python
#Lets figure out what the *normal* number of deaths per week is.  
#Create a dataframe of everything *except* 2020's deaths:
dfOld = dfUs[dfUs['Year'] < 2020]
#dfOld.tail()

#Lets try a simple mean/median by week as two simple predictors
avgDeathsDf = dfOld.groupby('Week').agg({'Deaths':[np.median,np.mean]}).reset_index()
print(avgDeathsDf[:5])
print(avgDeathsDf[-5:])
vals = avgDeathsDf.values;
#print(vals)
dictMedian = {a : b for a,b,c in vals}
dictMean = {a : c for a,b,c in vals}
print(dictMedian)


#And, because machine learning is the new hotness, lets try using XGBoost to build a 
# regression model for a more sophisticated predictor:
#Big thanks for this guy: https://www.geeksforgeeks.org/xgboost-for-regression/
xgBoostDfXold = dfOld[['Year', 'Week']].copy()
xgBoostDfYold = dfOld['Deaths'].copy()
train_X, test_X, train_y, test_y = train_test_split(xgBoostDfXold, xgBoostDfYold, test_size = 0.3, random_state=42) 
 
xgb_r = xg.XGBRegressor(objective ='reg:squarederror', n_estimators = 10, seed = 123) 
xgb_r.fit(train_X, train_y) 

# Use the model to predict our test data
xgbPredTest = xgb_r.predict(test_X) 

#Now use it to predict ALL the values in the dataset
xgBoostDfXall = dfUs[['Year', 'Week']].copy()
xgbPredAll = xgb_r.predict(xgBoostDfXall) 


```

      Week   Deaths              
             median          mean
    0    1  59898.0  60498.800000
    1    2  59824.5  59992.500000
    2    3  58548.5  58554.666667
    3    4  57710.0  57739.666667
    4    5  57838.5  57398.166667
       Week   Deaths              
              median          mean
    48   49  55501.0  55165.500000
    49   50  56320.0  55595.333333
    50   51  56754.0  56398.333333
    51   52  56967.0  57098.166667
    52   53  59481.0  59481.000000
    {1.0: 59898.0, 2.0: 59824.5, 3.0: 58548.5, 4.0: 57710.0, 5.0: 57838.5, 6.0: 57587.0, 7.0: 57294.0, 8.0: 56872.5, 9.0: 56419.0, 10.0: 56729.0, 11.0: 55508.5, 12.0: 55352.5, 13.0: 54515.5, 14.0: 54727.0, 15.0: 54286.0, 16.0: 53217.5, 17.0: 52494.5, 18.0: 52251.0, 19.0: 51600.0, 20.0: 50656.5, 21.0: 50823.5, 22.0: 50492.5, 23.0: 50778.0, 24.0: 50467.5, 25.0: 50479.5, 26.0: 50302.5, 27.0: 50443.0, 28.0: 49934.0, 29.0: 49811.0, 30.0: 49648.0, 31.0: 49936.0, 32.0: 50100.5, 33.0: 50109.0, 34.0: 49638.0, 35.0: 50218.0, 36.0: 50321.5, 37.0: 50626.5, 38.0: 50674.0, 39.0: 50735.5, 40.0: 51886.5, 41.0: 51837.5, 42.0: 51966.0, 43.0: 52321.5, 44.0: 52819.5, 45.0: 53012.0, 46.0: 53731.0, 47.0: 53730.5, 48.0: 54774.5, 49.0: 55501.0, 50.0: 56320.0, 51.0: 56754.0, 52.0: 56967.0, 53.0: 59481.0}



```python
#Plot overall mortality by week of year.  
avgDeathsDf.plot.line(x='Week', y='Deaths',title='Average US Mortality by Week',linewidth=3)
plt.scatter(x=dfOld['Week'],y=dfOld['Deaths'],facecolors='none', edgecolors='g')
#Wow the ends of the year are dangerous times to be alive
```




    <matplotlib.collections.PathCollection at 0x7f14ea2bbb50>




    
![png](output_11_1.png)
    



```python
#Now put the expected # of deaths by week back into the dataframe using a map of the dicts we created earlier
dfUs['ExpectedDeathsMedian'] = dfUs['Week'].map(dictMedian)
dfUs['ExpectedDeathsMean'] = dfUs['Week'].map(dictMean)
#Or by using the XGBoost predicted series directly.
dfUs['ExpectedDeathsXgb'] = xgbPredAll
dfUs.head()


#What's the R2 value for our predicted vs actual for 2014-2019?
rmseMed = np.sqrt(MSE(dfUs[dfUs['Year'] < 2020]['Deaths'], dfUs[dfUs['Year'] < 2020]['ExpectedDeathsMedian'])) 
rmseMea = np.sqrt(MSE(dfUs[dfUs['Year'] < 2020]['Deaths'], dfUs[dfUs['Year'] < 2020]['ExpectedDeathsMean'])) 
rmseXgb = np.sqrt(MSE(dfUs[dfUs['Year'] < 2020]['Deaths'], dfUs[dfUs['Year'] < 2020]['ExpectedDeathsXgb'])) 
print("RMSE Median:",rmseMed,"RMSE Mean:",rmseMea,"RMSE XGBoost:",rmseXgb)

#Of the 3 techniques, Mean has the lowest RMSE
```

    RMSE Median: 1962.5359228298055 RMSE Mean: 1932.2438512364388 RMSE XGBoost: 2127.6515559635823


---
# Improving the Model

Can we do a little bit better?  What if we first used a simple linear regression by time to account for the fact that the population is usually continuously increasing?  Then take the residuals of those predictions and put *those* into the average-by-week logic.  



```python
#Guide -https://realpython.com/linear-regression-in-python/
x = dfOld['YearValue'].to_numpy().reshape((-1, 1))
y = dfOld['Deaths'].to_numpy()

model = LinearRegression().fit(x,y)
#print(f)
y_pred = model.predict(x)

#Now use a delta y of the number of dealth minus the number of deaths predicted by the year linear model
yPredResiduals = dfOld['Deaths']-y_pred
week = dfOld['Week']
#print(dy)

dfYearAdjusted = pd.DataFrame()
dfYearAdjusted['YearValue'] = dfOld['YearValue']
dfYearAdjusted['Year'] = dfOld['Year']
dfYearAdjusted['Week'] = dfOld['Week']
dfYearAdjusted['PredDeathResidual'] = yPredResiduals
#print(dfYearAdjusted)

#Lets try a simple mean/median by week as two simple predictors
avgDeathsDfAdj = dfYearAdjusted.groupby('Week').agg({'PredDeathResidual':[np.median,np.mean]}).reset_index()
print(avgDeathsDfAdj[:5])
print(avgDeathsDfAdj[-5:])
valsAdj = avgDeathsDfAdj.values;
#print(vals)
dictMedianAdj = {a : b for a,b,c in valsAdj}
dictMeandj = {a : c for a,b,c in valsAdj}
print(dictMedianAdj)
```

      Week PredDeathResidual             
                      median         mean
    0    1       6850.165874  7450.965874
    1    2       6526.108647  7319.484956
    2    3       5154.115884  5866.658859
    3    4       4252.123120  5036.666096
    4    5       4520.071130  4680.173333
       Week PredDeathResidual             
                       median         mean
    48   49       1768.201391  1787.825081
    49   50       2322.332318  2202.665652
    50   51       2759.339555  2990.672888
    51   52       3952.846792  3675.513458
    52   53       7992.413256  7992.413256
    {1.0: 6850.16587377619, 2.0: 6526.1086468881695, 3.0: 5154.1158836010145, 4.0: 4252.123120313743, 5.0: 4520.071129694814, 6.0: 4464.202057340415, 7.0: 4212.644830452395, 8.0: 3726.280994366738, 9.0: 3483.9119220123393, 10.0: 3794.166540590697, 11.0: 2610.5500863710186, 12.0: 2139.7454776172526, 13.0: 1498.188250729232, 14.0: 1560.7599510429427, 15.0: 1186.7027241549222, 16.0: -45.60188459861092, 17.0: -645.1538752176566, 18.0: -638.8347930384334, 19.0: -1254.8275563255884, 20.0: -2164.7610922808526, 21.0: -2111.8775465009967, 22.0: -2711.2466188551625, 23.0: -1998.8630730750738, 24.0: -2607.7321454294724, 25.0: -2785.72490871686, 26.0: -2730.3413629367715, 27.0: -2672.1459716903046, 28.0: -2931.450580443605, 29.0: -3274.695961865247, 30.0: -3678.3124160853913, 31.0: -3492.552561238059, 32.0: -3188.045324525214, 33.0: -3641.790705946856, 34.0: -3810.9071601667674, 35.0: -3381.776232521166, 36.0: -3139.580841274699, 37.0: -3120.761759095476, 38.0: -2766.1308314501075, 39.0: -2623.811749270768, 40.0: -1648.8637398896972, 41.0: -1965.6683486432303, 42.0: -1318.537420997629, 43.0: -1041.653875217773, 44.0: -945.5229475721717, 45.0: -318.0801744600758, 46.0: 252.17968038737308, 47.0: 475.6869170999853, 48.0: 567.6296902119648, 49.0: 1768.2013905256754, 50.0: 2322.3323181712767, 51.0: 2759.3395548841218, 52.0: 3952.846791596967, 53.0: 7992.413255641703}



```python
#Plot overall mortality by week of year.  
avgDeathsDfAdj.plot.line(x='Week', y='PredDeathResidual',title='Average US Mortality by Week',linewidth=3)
plt.scatter(x=dfYearAdjusted['Week'],y=dfYearAdjusted['PredDeathResidual'],facecolors='none', edgecolors='g')
#Wow the ends of the year are dangerous times to be alive
```




    <matplotlib.collections.PathCollection at 0x7f14ea69ce50>




    
![png](output_15_1.png)
    



```python
#Now run the yearly predictor for all the data:
yPredYear = model.predict(dfUs['YearValue'].to_numpy().reshape((-1, 1)))
dfUs['PredJustYear']=yPredYear #The year component
dfUs['PredJustMonth']=dfUs['Week'].map(dictMedianAdj) #The week component
dfUs['ExpectedDeathsAdv']=dfUs['PredJustYear']+dfUs['PredJustMonth']
dfUs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Jurisdiction</th>
      <th>Year</th>
      <th>Week</th>
      <th>Deaths</th>
      <th>YYYY-WW</th>
      <th>YearValue</th>
      <th>ExpectedDeathsMedian</th>
      <th>ExpectedDeathsMean</th>
      <th>ExpectedDeathsXgb</th>
      <th>PredJustYear</th>
      <th>PredJustMonth</th>
      <th>ExpectedDeathsAdv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13834</th>
      <td>United States</td>
      <td>2014</td>
      <td>2</td>
      <td>55715.0</td>
      <td>2014-02</td>
      <td>2014.019231</td>
      <td>59824.5</td>
      <td>59992.500000</td>
      <td>53278.285156</td>
      <td>50723.955817</td>
      <td>6526.108647</td>
      <td>57250.064464</td>
    </tr>
    <tr>
      <th>13835</th>
      <td>United States</td>
      <td>2014</td>
      <td>3</td>
      <td>54681.0</td>
      <td>2014-03</td>
      <td>2014.038462</td>
      <td>58548.5</td>
      <td>58554.666667</td>
      <td>53278.285156</td>
      <td>50738.948580</td>
      <td>5154.115884</td>
      <td>55893.064464</td>
    </tr>
    <tr>
      <th>13836</th>
      <td>United States</td>
      <td>2014</td>
      <td>4</td>
      <td>54175.0</td>
      <td>2014-04</td>
      <td>2014.057692</td>
      <td>57710.0</td>
      <td>57739.666667</td>
      <td>52629.488281</td>
      <td>50753.941343</td>
      <td>4252.123120</td>
      <td>55006.064464</td>
    </tr>
    <tr>
      <th>13837</th>
      <td>United States</td>
      <td>2014</td>
      <td>5</td>
      <td>54049.0</td>
      <td>2014-05</td>
      <td>2014.076923</td>
      <td>57838.5</td>
      <td>57398.166667</td>
      <td>52629.488281</td>
      <td>50768.934107</td>
      <td>4520.071130</td>
      <td>55289.005236</td>
    </tr>
    <tr>
      <th>13838</th>
      <td>United States</td>
      <td>2014</td>
      <td>6</td>
      <td>53492.0</td>
      <td>2014-06</td>
      <td>2014.096154</td>
      <td>57587.0</td>
      <td>57383.000000</td>
      <td>52629.488281</td>
      <td>50783.926870</td>
      <td>4464.202057</td>
      <td>55248.128927</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Is our advanced "year + week" model actually better (a lower RMSE?)

rmseMea = np.sqrt(MSE(dfUs[dfUs['Year'] < 2020]['Deaths'], dfUs[dfUs['Year'] < 2020]['ExpectedDeathsMean'])) 
rmseMeaAdv = np.sqrt(MSE(dfUs[dfUs['Year'] < 2020]['Deaths'], dfUs[dfUs['Year'] < 2020]['ExpectedDeathsAdv'])) 
print("RMSE Mean (Week only model):",rmseMea,"RMSE Mean (Year+Week model):",rmseMeaAdv)

#We dropped RMSE from 1932 to 1165!
```

    RMSE Mean (Week only model): 1932.2438512364388 RMSE Mean (Year+Week model): 1165.5279074762302



```python
#Use the expected death mean values to calculate the amount of deaths that are above or below what we'd expect 
# from our historical model
dfUs['ExtraDeathsMean'] = dfUs['Deaths']-dfUs['ExpectedDeathsAdv']

```

# Results

Plot the actual number of deaths by week versus the "expected" number of deaths for that week from our previously calculated averages.

Things to look for:
 - Do deaths & expected deaths track well in our historical data?
 - Sections where the blue line is above the orange line means more people are dying in that week than the historical averages would suggest


```python
y_min = dfUs.Deaths.min()
y_max = dfUs.Deaths.max()

dfUs.plot(title='US Mortality by Week',x='YearValue',y=['Deaths','ExpectedDeathsAdv'],figsize=(12,8),ylim=(40000,80000))
#Why can't i draw vertical year lines?
#plt.vlines(x=['2015-01','2016-01'],ymin=y_min, ymax=y_max)
#Draw a line at March 2020
plt.vlines(2020+(2/12.0),dfUs.Deaths.min(),dfUs.Deaths.max(),color='red')
```




    <matplotlib.collections.LineCollection at 0x7f14ea49ab90>




    
![png](output_20_1.png)
    



```python
dfUs.plot(title='US EXTRA Mortality by Week',x='YearValue',y=['ExtraDeathsMean'],figsize=(12,8))
#Draw a line at y=0 (where expected deaths=actual deaths)
plt.hlines(0,min(dfUs['YearValue']), max(dfUs['YearValue']),color='black')
#Draw a line at March 2020
plt.vlines(2020+(2/12.0),dfUs.ExtraDeathsMean.min(),dfUs.ExtraDeathsMean.max(),color='red')
```




    <matplotlib.collections.LineCollection at 0x7f14ea3d9950>




    
![png](output_21_1.png)
    


# Intepreting the results

- There's a pretty clear deviation from 'normal' staring in early 2020 and remaining strong through the rest of the year.  Independent of *why* its pretty obvious that a lot more people are dying now than compared to the trend suggested by historical data.



# Suggestions for future work:
 - ~~Try using a simple ML model like XGBoost for a more sophisticated expected death rate predictor~~ (turns out mean was still a better predictor!)
 - ~~The "EXTRA Mortality" shows I still have a time based bias.  I bet correcting historical averages for US population by year would help account for that.~~ Splitting the model into a contiuous linear fit plus a by-week mean of the rediduals helped reduce the RMSE from 1932 to 1165
 - ~~Why can't i get nice x-axis tickmarks showing the start of each year?  Darn you matplotlib!~~ Encoding YYYY-WW into a year fraction (2020-05 -> 2020+(5-1/52)) makes for much nicer plots


```python

```


```python

```

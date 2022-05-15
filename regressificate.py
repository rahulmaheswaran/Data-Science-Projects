import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from parsedata import *
from sklearn.linear_model import Ridge, LinearRegression

bikedf = parse('NYC_Bicycle_Counts_2016_Corrected.csv')
bikedf = cleanData(bikedf)


#sns.lmplot(x="Precipitation", y="Total", data=bikedf, x_estimator=np.mean, x_jitter=.05).set(xlim=(0,1),ylim=(0,30000))
#plt.show()

#sns.scatterplot(data=bikedf, x="Precipitation", y="Total", hue="High Temp (°F)", size="High Temp (°F)", sizes=(20,200), legend="full").set(xlim=(0,1),ylim=(0,30000))
#plt.show()

def predictRain(bicyclists, bikedf): #train the data, find MSE
    x = np.array(bikedf['Total']).reshape((-1, 1))
    y = np.array(bikedf['Precipitation'])
    model = Ridge().fit(x,y)
    r_sq = model.score(x, y)
    print(f"The Ridge Regression Model is y = {model.coef_[0]}x + {model.intercept_} with a score of {r_sq}")
    print(f"The predicted chance of when rain is {bicyclists} is {model.intercept_ + model.coef_[0]*bicyclists}\n")

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print(f"The Linear Regression Model is y = {model.coef_[0]}x + {model.intercept_} with a score of {r_sq}")
    print(f"The predict chance of rain when is {bicyclists} is {model.intercept_ + model.coef_[0] * bicyclists}\n")

    sns.regplot(x="Precipitation", y="Total", data=bikedf)
    plt.show()

predictRain(15000, bikedf)

def predictBicyclists(bikedf):
    x = np.array([bikedf['Precipitation'], bikedf['Low Temp (°F)'], bikedf['High Temp (°F)']]).reshape(-1, 3)
    y = np.array(bikedf['Total'])
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print(f"The Linear Regression Model is y = {model.coef_[0]}x^2 + {model.coef_[1]}x + {model.coef_[2]} + {model.intercept_} with a score of {r_sq}\n")

    x = np.array([bikedf['Precipitation'], bikedf['Low Temp (°F)'], bikedf['High Temp (°F)']]).reshape(-1, 3)
    y = np.array(bikedf['Total'])
    model = Ridge().fit(x, y)
    r_sq = model.score(x, y)
    print(f"The Ridge Regression Model is y = {model.coef_[0]}x^2 + {model.coef_[1]}x + {model.coef_[2]} + {model.intercept_} with a score of {r_sq}\n")

predictBicyclists(bikedf)
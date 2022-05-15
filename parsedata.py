import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import re
from plots import *


def parse(filepath):
    data = pd.read_csv(filepath)
    return data
def cleanData(bikedf):
    #Add 2016 to be able to use pandas to_datetime
    bikedf['Date'] = bikedf['Date'].astype(str) + '-2016'
    bikedf['Date'] = pd.to_datetime(bikedf['Date'])
    bikedf['Precipitation'].replace( { r"[T]+" : '0' }, inplace= True, regex = True)
    bikedf['Precipitation'].replace( { r"[(S)]+" : '' }, inplace= True, regex = True)
    bikedf['Precipitation'] = bikedf['Precipitation'].str.replace(" ","")
    #bikedf['Precipitation'] = bikedf['Date'].astype(str) + '-2016'
    #Convert All the numbers to floats
    #Total, and all the bridge values
    bikedf['Total'] = bikedf['Total'].str.replace(',', '').astype(int)
    bikedf['Brooklyn Bridge'] = bikedf['Brooklyn Bridge'].str.replace(',', '').astype(int)
    bikedf['Manhattan Bridge'] = bikedf['Manhattan Bridge'].str.replace(',', '').astype(int)
    bikedf['Williamsburg Bridge'] = bikedf['Williamsburg Bridge'].str.replace(',', '').astype(int)
    bikedf['Queensboro Bridge'] = bikedf['Queensboro Bridge'].str.replace(',', '').astype(int)
    bikedf['Precipitation'] = bikedf['Precipitation'].astype(float)

    return bikedf



if __name__ == '__main__' :
#Let's first clean the data and convert them to the right data types
    bikedf = parse('NYC_Bicycle_Counts_2016_Corrected.csv')
    bikedf = cleanData(bikedf)
   # print(bikedf.loc[1])
   # print(bikedf.dtypes)
    plotBridge(bikedf)


#Let's visualize some data
   # plotNumber(bikedf['Date'],bikedf['Brooklyn Bridge'], bikedf, "Cyclists at Brooklyn")
    #plotMulti(bikedf['Date'], bikedf, "Cyclist traffic at Bridges in NYC")







   # print(names)



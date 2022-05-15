import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix
from parsedata import *
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression

bikedf = parse('NYC_Bicycle_Counts_2016_Corrected.csv')
bikedf = cleanData(bikedf)


sns.lmplot(x="Precipitation", y="Total", data=bikedf).set(xlim=(0,1),ylim=(0,30000))
plt.title("Precipitation v. Bicyclists")
plt.show()

#sns.scatterplot(data=bikedf, x="Precipitation", y="Total", size="High Temp (°F)", sizes=(20,200), legend="full").set(xlim=(0,1),ylim=(0,30000))
#plt.show()



def predictRain(bicyclists, bikedf):
    x = np.array(bikedf['Total']).reshape((-1, 1))
    y = []
    #y = np.array(bikedf['Precipitation'])
    for val in bikedf['Precipitation']: #light rain is .1, moderate rain is considered .3
        if(val == 0):
            y.append(0)
        else:
            y.append(1)
    y = np.array(y)

    bikedf["Precipitation"] = y
    # Step 3: Create a model and train it
    #model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
    #model.fit(x, y)
    #plt.scatter(x,y)
    plt.title("Rain prediction")
    sns.regplot(x="Total", y="Precipitation", data=bikedf, logistic=True)
    plt.xlabel('Total Bikers')
    plt.ylabel('Status (1:Raining, 0:Not Raining)')

    plt.show()

    X=x
    #print(f"The Ridge Regression Model is y = {model.coef_[0]}x + {model.intercept_} with a score of {r_sq}")
    #print(f"The predicted chance of when rain is {bicyclists} is {model.intercept_ + model.coef_[0]*bicyclists}\n")
    # training split vs testing split
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=42)
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    # Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    # Define the range of lambda to test
    lmbda = np.logspace(-1.00, 1, num=101)
    #lmbda = [0,2]

    MODEL = []
    MSE = []
    for l in lmbda:
        # Train the regression model using a regularization parameter of l
        model = train_model_log(X_train, y_train, l)

        # Evaluate the MSE on the test set
        mse = error(X_test, y_test, model)

        # Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    ind = MSE.index(min(MSE))
    [lmda_best, MSE_best, model_best] = [lmbda[ind], MSE[ind], MODEL[ind]]

    print('Best lambda tested is ' + str(lmda_best) + ', which yields an MSE of ' + str(MSE_best))
    # Plot the MSE as a function of lmbda
    plt  # fill in
    plt.plot(lmbda, MSE, color='black', label='MSE vs Lmbda')
    plt.plot(lmda_best, MSE_best, 'r*', label='Best lambda')
    plt.grid(alpha=.4, linestyle='--')
    plt.xlabel('Lambda λ')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE Model for best fitting Lambda')
    plt.legend()
    plt.show()

    print(model.intercept_)
    print(model.coef_)

#predictRain(15000, bikedf)

def predictBicyclists(bikedf):

    X = bikedf[['Precipitation','Low Temp (°F)','High Temp (°F)']]
    y = bikedf[['Total']]

    X = X.to_numpy()
    y = y.to_numpy()

    #training split vs testing split
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=42)
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    # Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    # Define the range of lambda to test
    lmbda = np.logspace(-1.00, 2, num=101)
    # lmbda = [0,2]

    MODEL = []
    MSE = []
    for l in lmbda:
        # Train the regression model using a regularization parameter of l
        model = train_model(X_train, y_train, l)

        # Evaluate the MSE on the test set
        mse = error(X_test, y_test, model)

        # Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    ind = MSE.index(min(MSE))
    [lmda_best, MSE_best, model_best] = [lmbda[ind], MSE[ind], MODEL[ind]]

    print('Best lambda tested is ' + str(lmda_best) + ', which yields an MSE of ' + str(MSE_best))
    # Plot the MSE as a function of lmbda
    plt  # fill in
    plt.plot(lmbda, MSE, color='black', label='MSE vs Lmbda')
    plt.plot(lmda_best, MSE_best, 'r*', label='Best lambda')
    plt.grid(alpha=.4, linestyle='--')
    plt.xlabel('Lambda λ')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE Model for best fitting Lambda')
    plt.legend()
    plt.show()


    #LOOK AT ME

    print(trn_mean)
    print(trn_std)

    print(model_best.intercept_)
    print(model_best.coef_)
    print(model_best.score(X_test, y_test))

    testie = np.array([.01, 29, 49])
    num = (testie - trn_mean)/trn_std

    def train_model(X, y, l):

        regr = linear_model.Ridge(alpha=l, fit_intercept=True)  # Define
        regr.fit(X, y)  # Fit model to training set
        return regr

    # Function that calculates the mean squared error of the model on the input dataset.
    # Input: Feature matrix X, target variable vector y, numpy model object
    # Output: mse, the mean squared error
    # pdf on website for SKlearn or discord "forth page"
    def error(X, y, model):
        y = np.array(y)
        y_pred_test = model.predict(X)
        squareError = (y - y_pred_test) ** 2
        mse = np.mean(squareError)

        return mse

    if __name__ == '__main__':
        model_best = main()
        # We use the following functions to obtain the model parameters instead of model_best.get_params()
        print(model_best.coef_)
        print(model_best.intercept_)

    print(num[0]*model_best.coef_[0][0]**3 + num[1]*model_best.coef_[0][1]**2 + num[2]*model_best.coef_[0][2] + model_best.intercept_[0])

    print(model_best.coef_[0][0]*(.01) + model_best.coef_[0][1]*(29) + model_best.coef_[0][2]*(49) + model_best.intercept_[0])

    #LOOK AT ME

    return




    '''

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print(f"The Linear Regression Model is y = {model.coef_[0]}x^2 + {model.coef_[1]}x + {model.coef_[2]} + {model.intercept_} with a score of {r_sq}\n")

    x = np.array([bikedf['Precipitation'], bikedf['Low Temp (°F)'], bikedf['High Temp (°F)']]).reshape(-1, 3)
    y = np.array(bikedf['Total'])
    model = Ridge().fit(x, y)
    r_sq = model.score(x, y)
    print(f"The Ridge Regression Model is y = {model.coef_[0]}x^2 + {model.coef_[1]}x + {model.coef_[2]} + {model.intercept_} with a score of {r_sq}\n")

    '''

def error(X,y,model):

    ypred = model.predict(X)
    mse = mean_squared_error(y, ypred)
    print("R score:", r2_score(y, ypred))

    return mse

def train_model(X,y,l):

    #fill in
    model = Ridge(alpha=l, fit_intercept=True)
    model.fit(X,y)


    return model

def train_model_log(X,y,l):

    #fill in
    #model = Ridge(alpha=l, fit_intercept=True)
    #model.fit(X,y)

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X,y)


    return model


def normalize_test(X_test, trn_mean, trn_std):

    num_col = len(X_test[0])
    X = X_test

    for i in range(num_col):
        X[:,i] = ((X_test[:,i] - trn_mean[i])/trn_std[i])


    return X


def normalize_train(X_train):

    #fill in
    num_col = len(X_train[0])
    mean = []
    std = []
    X = X_train
    for i in range(num_col):
        mean.append(np.mean(X_train[:,i], axis=0))
        std.append(np.std(X_train[:,i], axis=0))
        X[:,i] = ((X_train[:,i] - mean[i])/std[i])


    return X, mean, std

#predictRain(12500, bikedf)
predictBicyclists(bikedf)
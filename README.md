# Ex.No : 04   FIT ARMA MODEL FOR TIME SERIES
## Date : 08/04/2023

## AIM :
To implement ARMA model in python.

## ALGORITHM :
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

## PROGRAM :
#### Import necessary Modules and Functions
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```
#### Load the dataset
```
data = pd.read_csv('Gold Price Prediction.csv')
```
#### Detect the first numeric column to use
```
numeric_cols = data.select_dtypes(include=np.number).columns
if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found in the dataset.")
else:
    column_to_use = numeric_cols[0]
    print(f"Using column: {column_to_use}")

X = data[column_to_use].dropna()
```
#### Set figure size
```
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]
```
#### Plot original data
```
plt.plot(X)
plt.title(f'Original Data - {column_to_use}')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
```
#### Plot ACF and PACF of original data with reduced lags
```
plt.subplot(2, 1, 1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('ACF of Original Data (Lags=40)')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=40, ax=plt.gca())
plt.title('PACF of Original Data (Lags=40)')

plt.tight_layout()
plt.show()
```
#### Fit ARMA(1,1)
```
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']
```

#### Simulate ARMA(1,1)
```
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1, lags=40)
plt.title('ACF of Simulated ARMA(1,1) (Lags=40)')
plt.show()

plot_pacf(ARMA_1, lags=40)
plt.title('PACF of Simulated ARMA(1,1) (Lags=40)')
plt.show()
```
#### Fit ARMA(2,2)
```
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
```
#### Simulate ARMA(2,2)
```
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2, lags=40)
plt.title('ACF of Simulated ARMA(2,2) (Lags=40)')
plt.show()

plot_pacf(ARMA_2, lags=40)
plt.title('PACF of Simulated ARMA(2,2) (Lags=40)')
plt.show()
```

## OUTPUT :
#### ORIGINAL DATA :
![image](https://github.com/user-attachments/assets/a336fe6c-95bc-4c32-82be-91bb78e5f67f)

#### i. Partial Autocorrelation
![image](https://github.com/user-attachments/assets/3a04c88a-8ae6-42aa-93cd-b088b417755c)

#### ii. Autocorrelation
![image](https://github.com/user-attachments/assets/ff6bdb8a-7328-426f-bf71-19e67c75dea2)

#### SIMULATED ARMA(1,1) PROCESS :
![image](https://github.com/user-attachments/assets/3b0be9d8-9adf-473e-9f8e-7aaf036e9741)

#### i. Partial Autocorrelation
![image](https://github.com/user-attachments/assets/315972e3-4359-422d-a548-e0887f8cb845)

#### ii. Autocorrelation
![image](https://github.com/user-attachments/assets/2b1caca9-6cbc-4be9-ae26-d2e2e5679e2d)

#### SIMULATED ARMA(2,2) PROCESS :
![image](https://github.com/user-attachments/assets/79e643cb-83a3-44be-a39d-22af2dc66133)

#### i. Partial Autocorrelation
![image](https://github.com/user-attachments/assets/b408973f-0f61-459f-a6b5-52b99c5b5b6e)

#### ii. Autocorrelation
![image](https://github.com/user-attachments/assets/c9bd2e23-0fa4-42b8-9229-a7f6ce007d8c)

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.

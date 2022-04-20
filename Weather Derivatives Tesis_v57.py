
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tools as smt
import scipy.stats as scs
#import statsmodels.tsa.stattools as smtt
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
##acf(residuals_AEP, nlags=3650, alpha=.05, unbiased=True, qstat=True, fft=True)
##pacf(residuals_AEP, nlags=100, method='ols', alpha=.05)
#from statsmodels.graphics.tsaplots import plot_pacf
##plot_acf(residuals_AEP, lags=3650)
##plot_pacf(residuals_AEP, lags=100)
#from scipy import stats
import numpy as np
from scipy.signal import periodogram
#from statsmodels.stats.diagnostic import acorr_ljungbox
#Import the packages that we will be using


data = pd.read_excel("C:/Users/turko/Google Drive/Academia/Maestría en Finanzas UDESA/Tesis/Weather Derivatives/Data/Thesis_data_python.xlsx",index_col=0)
#Import the data as a DataFrame

data.describe()

#We drop the leap year days from the sample
data = data[~((data.index.month == 2) & (data.index.day == 29))]

#Now we create the vector with reference temperature for Heating Degree Day Calculation
data['Reference temp'] = 18


#SLice the dataframe 
Aeroparquepd = pd.DataFrame(data['AEROPARQUE'], index=data.index)
Aeroparques = data['AEROPARQUE']

Ezeizapd = pd.DataFrame(data['EZEIZA'], index=data.index)
Ezeizas = data['EZEIZA']

Bairespd = pd.DataFrame(data['BUENOS AIRES'], index=data.index)
Bairess = data['BUENOS AIRES']

#Define the daily returns function
def daily_return(variable):
    return variable.shift(-1) - variable
 
daily_returns_AEP = daily_return(Aeroparques)
daily_returns_AEP = daily_returns_AEP.dropna(axis=0) #Drop NaN values
daily_returns_EZE = daily_return(Ezeizas)
daily_returns_EZE = daily_returns_EZE.dropna(axis=0)
daily_returns_BA = daily_return(Bairess)
daily_returns_BA = daily_returns_BA.dropna(axis=0)

#We plot the qqplot for daily temperatures returns and confirm non normality
sm.qqplot(daily_returns_AEP, line='s')
sm.qqplot(daily_returns_EZE, line='s')
sm.qqplot(daily_returns_BA, line='s')

def print_statistics(array):
    ''' Prints selected statistics.
    Parameters
    ==========
    array: ndarray
    object to generate statistics on
    '''
    sta = scs.describe(array)
    print("%14s %15s" % ('statistic', 'value'))
    print(30 * "-")
    print("%14s %15.5f" % ('size', sta[0]))
    print("%14s %15.5f" % ('min', sta[1][0]))
    print("%14s %15.5f" % ('max', sta[1][1]))
    print("%14s %15.5f" % ('mean', sta[2]))
    print("%14s %15.5f" % ('std', np.sqrt(sta[3])))
    print("%14s %15.5f" % ('skew', sta[4]))
    print("%14s %15.5f" % ('kurtosis', sta[5]))

data2 = data.as_matrix() #A little transformation to numpy
Ezeiza = data2[ : , :1]
Aeroparque = data2[ : , 1:2]
Buenos_Aires = data2[ : , 2:3]
Reference_Temp = data2[ : , 4:5]

#Here we create the time vector
t = np.arange(1., len(data) + 1, 1)

#Period of oscillations
w = 2*np.pi/365

#We will construct the regression terms here for parameter estimation
sinwt = np.sin(w*t) 
coswt = np.cos(w*t)

#We reshape the vectors (8030, 1) so we can operate with them later 
#sinwt = sinwt.reshape((len(data), 1))
#coswt = coswt.reshape((len(data), 1))

#We use the periodogram method to detect cyclic components other than the yearly one
#First, we create the frequencies vector
f = np.arange(2, 268, .1)

#Second, we calculate the w corresponding to each frequency
wi = f* 2 * np.pi / len(data)

#Third, we calculate the periodogram
#####Aeroparque
periodogram_Aero = np.ones(wi.shape)
for i in wi:
    p = np.sqrt(sum(Aeroparque * np.sin(i * np.reshape(t, Aeroparque.shape))) ** 2 + sum(Aeroparque * np.cos(i * np.reshape(t, Aeroparque.shape))) ** 2) / len(Aeroparque)
    periodogram_Aero[np.where(wi == i)] = p
    
period_graph_Aero = pd.DataFrame(periodogram_Aero, f)
plt.plot(period_graph_Aero)

periodogram_AEP_2 = periodogram(np.reshape(Aeroparque, (8030, )))
periodogram_AEP_2 = pd.DataFrame(periodogram_AEP_2[1], index=periodogram_AEP_2[0])


#####Ezeiza
periodogram_Eze = np.ones(wi.shape)
for i in wi:
    p = np.sqrt(sum(Ezeiza * np.sin(i * np.reshape(t, Ezeiza.shape))) ** 2 + sum(Ezeiza * np.cos(i * np.reshape(t, Ezeiza.shape))) ** 2) / len(Ezeiza)
    periodogram_Eze[np.where(wi == i)] = p
    
period_graph_Eze = pd.DataFrame(periodogram_Eze, f)
plt.plot(period_graph_Eze)

periodogram_EZE_2 = periodogram(np.reshape(Ezeiza, (8030, )))
periodogram_EZE_2 = pd.DataFrame(periodogram_EZE_2[1], index=periodogram_EZE_2[0])

#####Buenos Aires
periodogram_Baires = np.ones(wi.shape)
for i in wi:
    p = np.sqrt(sum(Buenos_Aires * np.sin(i * np.reshape(t, Buenos_Aires.shape))) ** 2 + sum(Buenos_Aires * np.cos(i * np.reshape(t, Buenos_Aires.shape))) ** 2) / len(Buenos_Aires)
    periodogram_Baires[np.where(wi == i)] = p
    
period_graph_Baires = pd.DataFrame(periodogram_Baires, f)
plt.plot(period_graph_Baires)

periodogram_BA_2 = periodogram(np.reshape(Buenos_Aires, (8030, )))
periodogram_BA_2 = pd.DataFrame(periodogram_BA_2[1], index=periodogram_BA_2[0])

#We create the vector of exogenous variables and convert it to pd.DataFrame
exog_variables = np.append([t], [sinwt, coswt], axis=0)
exog_variables = exog_variables.transpose()
exog_variables2 = pd.DataFrame(exog_variables)
exog_variables2 = exog_variables2.rename(index=str, columns={0:"t", 1:"sinwt", 2:"coswt"})

#We add the constant term to the vector of exogenous variables
exog_variables2 = smt.add_constant(exog_variables2, prepend=True)

#Now that we have our exogenous variables ready, we perform the linear regression model
#####Aeroparque
mod_AEP = smf.OLS(Aeroparque, exog_variables2)
res_AEP = mod_AEP.fit()
print(res_AEP.summary())
#####Ezeiza
mod_EZE = smf.OLS(Ezeiza, exog_variables2)
res_EZE = mod_EZE.fit()
print(res_EZE.summary())
#####Buenos Aires
mod_BA = smf.OLS(Buenos_Aires, exog_variables2)
res_BA = mod_BA.fit()
print(res_BA.summary())

#We create a new vector with the regression parameters
params_AEP = res_AEP.params
params_EZE = res_EZE.params
params_BA = res_BA.params

#We calculate the parameters of our temperature model
#####Aeroparque
A_AEP = params_AEP['const']
B_AEP = params_AEP['t']
C_AEP = np.sqrt(params_AEP['sinwt']**2 + params_AEP['coswt']**2)
phi_AEP = np.arctan(params_AEP['coswt']/params_AEP['sinwt']) 
#####Ezeiza
A_EZE = params_EZE['const']
B_EZE = params_EZE['t']
C_EZE = np.sqrt(params_EZE['sinwt']**2 + params_EZE['coswt']**2)
phi_EZE = np.arctan(params_EZE['coswt']/params_EZE['sinwt']) 
#####Buenos Aires
A_BA = params_BA['const']
B_BA = params_BA['t']
C_BA = np.sqrt(params_BA['sinwt']**2 + params_BA['coswt']**2)
phi_BA = np.arctan(params_BA['coswt']/params_BA['sinwt']) 

#Now we calculate the first, second and third terms of our deterministic model
#####Aeroparque
First_term_AEP = A_AEP * np.ones(t.shape)
Second_term_AEP = B_AEP * t
Third_term_AEP = C_AEP*np.sin(w*t + phi_AEP)

T_est_AEP = First_term_AEP + Second_term_AEP + Third_term_AEP

#####Ezeiza
First_term_EZE = A_EZE * np.ones(t.shape)
Second_term_EZE = B_EZE * t
Third_term_EZE = C_EZE*np.sin(w*t + phi_EZE)

T_est_EZE = First_term_EZE + Second_term_EZE + Third_term_EZE

#####Buenos Aires
First_term_BA = A_BA * np.ones(t.shape)
Second_term_BA = B_BA * t
Third_term_BA = C_BA*np.sin(w*t + phi_BA)

T_est_BA = First_term_BA + Second_term_BA + Third_term_BA

#Each station is plotted
plt.figure(figsize=(12, 14))
plt.subplot(311)
plt.plot(Aeroparquepd,'b', lw=1.0)
T_estpd_AEP = pd.DataFrame(T_est_AEP, index=data.index)
T_estpd_AEP.columns = ['Deterministic Temperature']
plt.plot(T_estpd_AEP,'r')
plt.ylabel('Temperature in °C')
plt.title('Aeroparque')
plt.grid(True, alpha=0.3, linestyle='dashed')


plt.subplot(312)
plt.plot(Ezeizapd,'b', lw=1.0)
T_estpd_EZE = pd.DataFrame(T_est_EZE, index=data.index)
T_estpd_EZE.columns = ['Deterministic Temperature']
plt.plot(T_estpd_EZE,'r')
plt.ylabel('Temperature in °C')
plt.title('Ezeiza')
plt.grid(True, alpha=0.3, linestyle='dashed')


plt.subplot(313)
plt.plot(Bairespd,'b', lw=1.0)
T_estpd_BA = pd.DataFrame(T_est_BA, index=data.index)
T_estpd_BA.columns = ['Deterministic Temperature']
plt.plot(T_estpd_BA,'r')
plt.ylabel('Temperature in °C')
plt.title('Buenos Aires')
plt.grid(True, alpha=0.3, linestyle='dashed')



#We calculate the error (daily temperature differences between actual temperature and predicted temperature) 
#####Aeroparque
residuals_AEP = Aeroparques - T_est_AEP
residuals_AEP.columns = ['AEROPARQUE']
residualspd_AEP = pd.DataFrame(residuals_AEP, index=Aeroparquepd.index)

periodogram_resid_AEP = periodogram(residuals_AEP.as_matrix())
periodogram_resid_AEP = pd.DataFrame(periodogram_resid_AEP[1], index=periodogram_resid_AEP[0])
#####Ezeiza
residuals_EZE = Ezeizas - T_est_EZE
residuals_EZE.columns = ['EZEIZA']
residualspd_EZE = pd.DataFrame(residuals_EZE, index=Ezeizapd.index)

periodogram_resid_EZE = periodogram(residuals_EZE.as_matrix())
periodogram_resid_EZE = pd.DataFrame(periodogram_resid_EZE[1], index=periodogram_resid_EZE[0])
######Buenos Aires
residuals_BA = Bairess - T_est_BA
residuals_BA.columns = ['BUENOS AIRES']
residualspd_BA = pd.DataFrame(residuals_BA, index=Bairespd.index)

periodogram_resid_BA = periodogram(residuals_BA.as_matrix())
periodogram_resid_BA = pd.DataFrame(periodogram_resid_BA[1], index=periodogram_resid_BA[0])
#Now we calculate the return of the residuals for the qqplot
#####Aeroparque
daily_returns_residuals_AEP = daily_return(residuals_AEP)
daily_returns_residuals_AEP = daily_returns_residuals_AEP.dropna(axis=0) #Drop NaN values
daily_returns_residualspd_AEP = pd.DataFrame(daily_returns_residuals_AEP, index=Aeroparquepd.index).dropna()
#####Ezeiza
daily_returns_residuals_EZE = daily_return(residuals_EZE)
daily_returns_residuals_EZE = daily_returns_residuals_EZE.dropna(axis=0) #Drop NaN values
daily_returns_residualspd_EZE = pd.DataFrame(daily_returns_residuals_EZE, index=Ezeizapd.index).dropna()
#####Buenos Aires
daily_returns_residuals_BA = daily_return(residuals_BA)
daily_returns_residuals_BA = daily_returns_residuals_BA.dropna(axis=0) #Drop NaN values
daily_returns_residualspd_BA = pd.DataFrame(daily_returns_residuals_BA, index=Bairespd.index).dropna()

#We plot the qqplot for residuals returns to test normality
QQplt_resid_AEP = sm.qqplot(daily_returns_residuals_AEP, line='s')
QQplt_resid_EZE = sm.qqplot(daily_returns_residuals_EZE, line='s')
QQplt_resid_BA = sm.qqplot(daily_returns_residuals_BA, line='s')

#We test the normality of returns below

def normality_tests(arr):
    ''' Tests for normality distribution of given data set.
    Parameters
    ==========
    array: ndarray
    object to generate statistics on
    '''
    print("Skew of data set %14.3f" % scs.skew(arr))
    print("Skew test p-value %14.3f" % scs.skewtest(arr)[1])
    print("Kurt of data set %14.3f" % scs.kurtosis(arr))
    print("Kurt test p-value %14.3f" % scs.kurtosistest(arr)[1])
    print("Norm test p-value %14.3f" % scs.normaltest(arr)[1])
    
normality_tests(daily_returns_residuals_AEP)
normality_tests(daily_returns_residuals_EZE)
normality_tests(daily_returns_residuals_BA)

#We filter outliers
#####Aeroparque
filtered_drresid_AEP = daily_returns_residualspd_AEP[~(daily_returns_residualspd_AEP < -np.std(daily_returns_residualspd_AEP) * 2.1).any(1)]
filtered_drresid2_AEP = filtered_drresid_AEP[~(np.abs(filtered_drresid_AEP) > np.std(filtered_drresid_AEP) * 3).any(1)]
#filtered_drresid3_AEP = filtered_drresid2_AEP[~(np.abs(filtered_drresid2_AEP) > np.std(filtered_drresid2_AEP) * 3).any(1)]
#####Ezeiza
filtered_drresid_EZE = daily_returns_residualspd_EZE[~(daily_returns_residualspd_EZE < -np.std(daily_returns_residualspd_EZE) * 2.8).any(1)]
filtered_drresid2_EZE = filtered_drresid_EZE[~(np.abs(filtered_drresid_EZE) > np.std(filtered_drresid_EZE) * 3).any(1)]
#filtered_drresid3_EZE = filtered_drresid2_EZE[~(np.abs(filtered_drresid2_EZE) > np.std(filtered_drresid2_EZE) * 3).any(1)]
#####Buenos Aires
filtered_drresid_BA = daily_returns_residualspd_BA[~(np.abs(daily_returns_residualspd_BA) > np.std(daily_returns_residualspd_BA) * 3).any(1)]
filtered_drresid2_BA = filtered_drresid_BA[~(np.abs(filtered_drresid_BA) > np.std(filtered_drresid_BA) * 3).any(1)]
#filtered_drresid3_BA = filtered_drresid2_BA[~(np.abs(filtered_drresid2_BA) > np.std(filtered_drresid2_BA) * 3).any(1)]

#Plot histograms
plt.figure(figsize=(8, 16))
#########Aeroparque
plt.subplot(311)
plt.hist(daily_returns_residuals_AEP, bins=70, histtype='step', color='b', label='frequency', density='True')
plt.grid(True, alpha=0.3, linestyle='dashed')
plt.xlabel('daily returns of the residuals')
plt.ylabel('frequency')
plt.title('Aeroparque')
x = pd.DataFrame(np.linspace(plt.axis()[0], plt.axis()[1]))
plt.plot(x, scs.norm.pdf(x, loc=pd.DataFrame.mean(daily_returns_residuals_AEP), scale=pd.DataFrame.std(daily_returns_residuals_AEP)),
         'r', lw=1.0, label='pdf')
plt.legend()
#########Ezeiza
plt.subplot(312)
plt.hist(daily_returns_residuals_EZE, bins=70, histtype='step', color='b', label='frequency', density='True')
plt.grid(True, alpha=0.3, linestyle='dashed')
plt.xlabel('daily returns of the residuals')
plt.ylabel('frequency')
plt.title('Ezeiza')
x = pd.DataFrame(np.linspace(plt.axis()[0], plt.axis()[1]))
plt.plot(x, scs.norm.pdf(x, loc=pd.DataFrame.mean(daily_returns_residuals_EZE), scale=pd.DataFrame.std(daily_returns_residuals_EZE)),
         'r', lw=1.0, label='pdf')
plt.legend()
#########Buenos Aires
plt.subplot(313)
plt.hist(daily_returns_residuals_BA, bins=70, histtype='step', color='b', label='frequency', density='True')
plt.grid(True, alpha=0.3, linestyle='dashed')
plt.xlabel('daily returns of the residuals')
plt.ylabel('frequency')
plt.title('Buenos Aires')
x = pd.DataFrame(np.linspace(plt.axis()[0], plt.axis()[1]))
plt.plot(x, scs.norm.pdf(x, loc=pd.DataFrame.mean(daily_returns_residuals_BA), scale=pd.DataFrame.std(daily_returns_residuals_BA)),
         'r', lw=1.0, label='pdf')
plt.legend()

###We compare results before and after filtering for outliers to see if the normality test improve
#########Aeroparque
plt.figure(figsize=(7,10))
plt.subplot(211)
plt.hist(daily_returns_residuals_AEP, bins=70, histtype='step', color='b', label='frequency', density='True')
plt.grid(True, alpha=0.3, linestyle='dashed')
plt.xlabel('daily returns of the residuals')
plt.ylabel('frequency')
plt.title('Aeroparque')
x = pd.DataFrame(np.linspace(plt.axis()[0], plt.axis()[1]))
plt.plot(x, scs.norm.pdf(x, loc=pd.DataFrame.mean(daily_returns_residuals_AEP), scale=pd.DataFrame.std(daily_returns_residuals_AEP)),
         'r', lw=1.0, label='pdf')
plt.legend(loc='upper right')
plt.subplot(212, sharex=plt.subplot(211))
plt.hist(filtered_drresid2_AEP.as_matrix(), bins=70, histtype='step', color='b', label='frequency', density='True')
plt.grid(True, alpha=0.3, linestyle='dashed')
plt.xlabel('daily returns of filtered residuals')
plt.ylabel('frequency')
plt.title('Aeroparque')
x = pd.DataFrame(np.linspace(plt.axis()[0], plt.axis()[1]))
plt.plot(x, scs.norm.pdf(x, loc=pd.DataFrame.mean(filtered_drresid2_AEP), scale=pd.DataFrame.std(filtered_drresid2_AEP)),
         'r', lw=1.0, label='pdf')
plt.legend(loc='upper right')
#########Ezeiza
plt.figure(figsize=(7,10))
plt.subplot(211)
plt.hist(daily_returns_residuals_EZE, bins=70, histtype='step', color='b', label='frequency', density='True')
plt.grid(True, alpha=0.3, linestyle='dashed')
plt.xlabel('daily returns of the residuals')
plt.ylabel('frequency')
plt.title('Ezeiza')
x = pd.DataFrame(np.linspace(plt.axis()[0], plt.axis()[1]))
plt.plot(x, scs.norm.pdf(x, loc=pd.DataFrame.mean(daily_returns_residuals_EZE), scale=pd.DataFrame.std(daily_returns_residuals_EZE)),
         'r', lw=1.0, label='pdf')
plt.legend(loc='upper right')
plt.subplot(212, sharex=plt.subplot(211))
plt.hist(filtered_drresid2_EZE.as_matrix(), bins=70, histtype='step', color='b', label='frequency', density='True')
plt.grid(True, alpha=0.3, linestyle='dashed')
plt.xlabel('daily returns of filtered residuals')
plt.ylabel('frequency')
plt.title('Ezeiza')
x = pd.DataFrame(np.linspace(plt.axis()[0], plt.axis()[1]))
plt.plot(x, scs.norm.pdf(x, loc=pd.DataFrame.mean(filtered_drresid2_EZE), scale=pd.DataFrame.std(filtered_drresid2_EZE)),
         'r', lw=1.0, label='pdf')
plt.legend(loc='upper right')
#########Buenos Aires
plt.figure(figsize=(7,10))
plt.subplot(211)
plt.hist(daily_returns_residuals_BA, bins=70, histtype='step', color='b', label='frequency', density='True')
plt.grid(True, alpha=0.3, linestyle='dashed')
plt.xlabel('daily returns of the residuals')
plt.ylabel('frequency')
plt.title('Buenos Aires')
x = pd.DataFrame(np.linspace(plt.axis()[0], plt.axis()[1]))
plt.plot(x, scs.norm.pdf(x, loc=pd.DataFrame.mean(daily_returns_residuals_BA), scale=pd.DataFrame.std(daily_returns_residuals_BA)),
         'r', lw=1.0, label='pdf')
plt.legend(loc='upper right')
plt.subplot(212, sharex=plt.subplot(211))
plt.hist(filtered_drresid2_BA.as_matrix(), bins=70, histtype='step', color='b', label='frequency', density='True')
plt.grid(True, alpha=0.3, linestyle='dashed')
plt.xlabel('daily returns of filtered residuals')
plt.ylabel('frequency')
plt.title('Buenos Aires')
x = pd.DataFrame(np.linspace(plt.axis()[0], plt.axis()[1]))
plt.plot(x, scs.norm.pdf(x, loc=pd.DataFrame.mean(filtered_drresid2_BA), scale=pd.DataFrame.std(filtered_drresid2_BA)),
         'r', lw=1.0, label='pdf')
plt.legend(loc='upper right')

#####Now we incorporate all peaks of the periodogram to the OLS model and we will try to see if the normality tests improve

sin2wt = np.sin(2*w*t) 
cos2wt = np.cos(2*w*t)
sin73o4wt = np.sin((73/4)*w*t) #this is the periodic component for 20 days
cos73o4wt = np.cos((73/4)*w*t)
#sin365o34wt = np.sin((365/34)*w*t) #this is the periodic component for 34 days present in EZE and BA only
#cos365o34wt = np.cos((365/34)*w*t)
#sin365o141wt = np.sin((365/141)*w*t) #this is the periodic component for 141 days present in AEP only
#cos365o141wt = np.cos((365/141)*w*t)
#sin4wt = np.sin(4*w*t) 
#cos4wt = np.cos(4*w*t)
#sin1o3wt = np.sin((1/3)*w*t) 
#cos1o3wt = np.cos((1/3)*w*t)
#sin4o7wt = np.sin((4/7)*w*t) 
#cos4o7wt = np.cos((4/7)*w*t)
#sin6o7wt = np.sin((6/7)*w*t) 
#cos6o7wt = np.cos((6/7)*w*t)



exog_variables_periodogram = np.append([t], [sinwt, coswt, sin2wt, cos2wt, sin73o4wt, cos73o4wt], axis=0)
exog_variables_periodogram = exog_variables_periodogram.transpose()
exog_variables_periodogram2 = pd.DataFrame(exog_variables_periodogram)
exog_variables_periodogram2 = exog_variables_periodogram2.rename(index=str, columns={0:"t", 1:"sinwt", 2:"coswt", 3:"sin2wt", 4:"cos2wt", 5:"sin73/4wt", 6:"cos73/4wt"})
exog_variables_periodogram2 = smt.add_constant(exog_variables_periodogram2, prepend=True)
######Aeroparque
mod_period_AEP = smf.OLS(Aeroparque, exog_variables_periodogram2)
res_period_AEP = mod_period_AEP.fit()
print(res_period_AEP.summary())
######Ezeiza
mod_period_EZE = smf.OLS(Ezeiza, exog_variables_periodogram2)
res_period_EZE = mod_period_EZE.fit()
print(res_period_EZE.summary())
######Buenos Aires
mod_period_BA = smf.OLS(Buenos_Aires, exog_variables_periodogram2)
res_period_BA = mod_period_BA.fit()
print(res_period_BA.summary())

######Aeroparque
params_period_AEP = res_period_AEP.params
######Ezeiza
params_period_EZE = res_period_EZE.params
######Buenos Aires
params_period_BA = res_period_BA.params

######Aeroparque####################################################################
A1_AEP = params_period_AEP['const']
A2_AEP = params_period_AEP['t']
A3_AEP = np.sqrt(params_period_AEP['sinwt']**2 + params_period_AEP['coswt']**2)
phi1_AEP = np.arctan(params_period_AEP['coswt']/params_period_AEP['sinwt'])
A4_AEP = np.sqrt(params_period_AEP['sin2wt']**2 + params_period_AEP['cos2wt']**2)
phi2_AEP = np.arctan(params_period_AEP['cos2wt']/params_period_AEP['sin2wt'])
A5_AEP = np.sqrt(params_period_AEP['sin73/4wt']**2 + params_period_AEP['cos73/4wt']**2)
phi3_AEP = np.arctan(params_period_AEP['cos73/4wt']/params_period_AEP['sin73/4wt'])

######Ezeiza####################################################################
A1_EZE = params_period_EZE['const']
A2_EZE = params_period_EZE['t']
A3_EZE = np.sqrt(params_period_EZE['sinwt']**2 + params_period_EZE['coswt']**2)
phi1_EZE = np.arctan(params_period_EZE['coswt']/params_period_EZE['sinwt'])
A4_EZE = np.sqrt(params_period_EZE['sin2wt']**2 + params_period_EZE['cos2wt']**2)
phi2_EZE = np.arctan(params_period_EZE['cos2wt']/params_period_EZE['sin2wt'])
A5_EZE = np.sqrt(params_period_EZE['sin73/4wt']**2 + params_period_EZE['cos73/4wt']**2)
phi3_EZE = np.arctan(params_period_EZE['cos73/4wt']/params_period_EZE['sin73/4wt'])

######Buenos Aires####################################################################
A1_BA = params_period_BA['const']
A2_BA = params_period_BA['t']
A3_BA = np.sqrt(params_period_BA['sinwt']**2 + params_period_BA['coswt']**2)
phi1_BA = np.arctan(params_period_BA['coswt']/params_period_BA['sinwt'])
A4_BA = np.sqrt(params_period_BA['sin2wt']**2 + params_period_BA['cos2wt']**2)
phi2_BA = np.arctan(params_period_BA['cos2wt']/params_period_BA['sin2wt'])
A5_BA = np.sqrt(params_period_BA['sin73/4wt']**2 + params_period_BA['cos73/4wt']**2)
phi3_BA = np.arctan(params_period_BA['cos73/4wt']/params_period_BA['sin73/4wt'])

######Aeroparque####################################################################
First_term_period_AEP = A1_AEP * np.ones(t.shape)
Second_term_period_AEP = A2_AEP * t
Third_term_period_AEP = A3_AEP*np.sin(w*t + phi1_AEP)
Fourth_term_period_AEP = A4_AEP*np.sin(2*w*t + phi2_AEP)
Fifth_term_period_AEP = A5_AEP*np.sin((73/4)*w*t + phi3_AEP)

######Ezeiza####################################################################
First_term_period_EZE = A1_EZE * np.ones(t.shape)
Second_term_period_EZE = A2_EZE * t
Third_term_period_EZE = A3_EZE*np.sin(w*t + phi1_EZE)
Fourth_term_period_EZE = A4_EZE*np.sin(2*w*t + phi2_EZE)
Fifth_term_period_EZE = A5_EZE*np.sin((73/4)*w*t + phi3_EZE)

######Buenos Aires####################################################################
First_term_period_BA = A1_BA * np.ones(t.shape)
Second_term_period_BA = A2_BA * t
Third_term_period_BA = A3_BA*np.sin(w*t + phi1_BA)
Fourth_term_period_BA = A4_BA*np.sin(2*w*t + phi2_BA)
Fifth_term_period_BA = A5_BA*np.sin((73/4)*w*t + phi3_BA)

######Aeroparque####################################################################
T_est_period_AEP = First_term_period_AEP + Second_term_period_AEP + Third_term_period_AEP + Fourth_term_period_AEP + Fifth_term_period_AEP
T_est_period2_AEP = params_period_AEP['const'] * np.ones(t.shape) + params_period_AEP['t'] * t + params_period_AEP['sinwt'] * np.sin(w * t) + params_period_AEP['coswt'] * np.cos(w * t) + params_period_AEP['sin2wt'] * np.sin(2 * w * t) + params_period_AEP['cos2wt'] * np.cos(2 * w * t) + params_period_AEP['sin73/4wt'] * np.sin((73/4) * w * t) + params_period_AEP['cos73/4wt'] * np.cos((73/4) * w * t) 
######Ezeiza####################################################################
T_est_period_EZE = First_term_period_EZE + Second_term_period_EZE + Third_term_period_EZE + Fourth_term_period_EZE + Fifth_term_period_EZE 
T_est_period2_EZE = params_period_EZE['const'] * np.ones(t.shape) + params_period_EZE['t'] * t + params_period_EZE['sinwt'] * np.sin(w * t) + params_period_EZE['coswt'] * np.cos(w * t) + params_period_EZE['sin2wt'] * np.sin(2 * w * t) + params_period_EZE['cos2wt'] * np.cos(2 * w * t) + params_period_EZE['sin73/4wt'] * np.sin((73/4) * w * t) + params_period_EZE['cos73/4wt'] * np.cos((73/4) * w * t)
######Buenos Aires####################################################################
T_est_period_BA = First_term_period_BA + Second_term_period_BA + Third_term_period_BA + Fourth_term_period_BA + Fifth_term_period_BA
T_est_period2_BA = params_period_BA['const'] * np.ones(t.shape) + params_period_BA['t'] * t + params_period_BA['sinwt'] * np.sin(w * t) + params_period_BA['coswt'] * np.cos(w * t) + params_period_BA['sin2wt'] * np.sin(2 * w * t) + params_period_BA['cos2wt'] * np.cos(2 * w * t) + params_period_BA['sin73/4wt'] * np.sin((73/4) * w * t) + params_period_BA['cos73/4wt'] * np.cos((73/4) * w * t)

######Aeroparque####################################################################
#residuals_period_AEP = res_period_AEP.resid
residuals_period_AEP = Aeroparquepd - pd.DataFrame(T_est_period_AEP, index=Aeroparquepd.index, columns=['AEROPARQUE'])
#residuals_period_AEP = pd.DataFrame(np.reshape(residuals_period_AEP, Aeroparquepd.shape),index=Aeroparquepd.index ,columns=['AEROPARQUE'])
daily_returns_resid_period_AEP = daily_return(residuals_period_AEP)
daily_returns_resid_period_AEP = daily_returns_resid_period_AEP.dropna()

filtered_drresid_period_AEP = daily_returns_resid_period_AEP[~(np.abs(daily_returns_resid_period_AEP) > np.std(daily_returns_resid_period_AEP) * 3).any(1)]
filtered_drresid_period2_AEP = filtered_drresid_period_AEP[~(np.abs(filtered_drresid_period_AEP) > np.std(filtered_drresid_period_AEP) * 3).any(1)]
filtered_drresid_period3_AEP = filtered_drresid_period2_AEP[~(np.abs(filtered_drresid_period2_AEP) > np.std(filtered_drresid_period2_AEP) * 3).any(1)]
######Ezeiza####################################################################
#residuals_period_EZE = res_period_EZE.resid
residuals_period_EZE = Ezeizapd - pd.DataFrame(T_est_period_EZE, index=Ezeizapd.index, columns=['EZEIZA'])
#residuals_period_EZE = pd.DataFrame(residuals_period_EZE)
daily_returns_resid_period_EZE = daily_return(residuals_period_EZE)
daily_returns_resid_period_EZE = daily_returns_resid_period_EZE.dropna(axis=0)

filtered_drresid_period_EZE = daily_returns_resid_period_EZE[~(np.abs(daily_returns_resid_period_EZE) > np.std(daily_returns_resid_period_EZE) * 3).any(1)]
filtered_drresid_period2_EZE = filtered_drresid_period_EZE[~(np.abs(filtered_drresid_period_EZE) > np.std(filtered_drresid_period_EZE) * 3).any(1)]
filtered_drresid_period3_EZE = filtered_drresid_period2_EZE[~(np.abs(filtered_drresid_period2_EZE) > np.std(filtered_drresid_period2_EZE) * 3).any(1)]
######Buenos Aires####################################################################
#residuals_period_BA = res_period_BA.resid
residuals_period_BA = Bairespd - pd.DataFrame(T_est_period_BA, index=Bairespd.index, columns=['BUENOS AIRES'])
#residuals_period_BA = pd.DataFrame(residuals_period_BA)
daily_returns_resid_period_BA = daily_return(residuals_period_BA)
daily_returns_resid_period_BA = daily_returns_resid_period_BA.dropna(axis=0)

filtered_drresid_period_BA = daily_returns_resid_period_BA[~(np.abs(daily_returns_resid_period_BA) > np.std(daily_returns_resid_period_BA) * 3).any(1)]
filtered_drresid_period2_BA = filtered_drresid_period_BA[~(np.abs(filtered_drresid_period_BA) > np.std(filtered_drresid_period_BA) * 3).any(1)]
filtered_drresid_period3_BA = filtered_drresid_period2_BA[~(np.abs(filtered_drresid_period2_BA) > np.std(filtered_drresid_period2_BA) * 3).any(1)]

#Plot raw data with fitted OLS estimation after adding additional terms

plt.figure(figsize=(12, 14))
plt.subplot(311)
plt.plot(Aeroparquepd,'b', lw=1.0)
T_estpd_period_AEP = pd.DataFrame(T_est_period_AEP, index=data.index)
T_estpd_period_AEP.columns = ['Deterministic Temperature']
plt.plot(T_estpd_period_AEP,'r')
plt.ylabel('Temperature in °C')
plt.title('Aeroparque')
plt.grid(True, alpha=0.3, linestyle='dashed')


plt.subplot(312)
plt.plot(Ezeizapd,'b', lw=1.0)
T_estpd_period_EZE = pd.DataFrame(T_est_period_EZE, index=data.index)
T_estpd_period_EZE.columns = ['Deterministic Temperature']
plt.plot(T_estpd_period_EZE,'r')
plt.ylabel('Temperature in °C')
plt.title('Ezeiza')
plt.grid(True, alpha=0.3, linestyle='dashed')


plt.subplot(313)
plt.plot(Bairespd,'b', lw=1.0)
T_estpd_period_BA = pd.DataFrame(T_est_period_BA, index=data.index)
T_estpd_period_BA.columns = ['Deterministic Temperature']
plt.plot(T_estpd_period_BA,'r')
plt.ylabel('Temperature in °C')
plt.title('Buenos Aires')
plt.grid(True, alpha=0.3, linestyle='dashed')

#Plot histograms
plt.figure(figsize=(8, 16))
#########Aeroparque
plt.subplot(311)
plt.hist(daily_returns_resid_period_AEP.as_matrix(), bins=70, histtype='step', color='b', label='frequency', density='True')
plt.grid(True, alpha=0.3, linestyle='dashed')
plt.xlabel('daily returns of the residuals')
plt.ylabel('frequency')
plt.title('Aeroparque')
x = pd.DataFrame(np.linspace(plt.axis()[0], plt.axis()[1]))
plt.plot(x, scs.norm.pdf(x, loc=pd.DataFrame.mean(daily_returns_resid_period_AEP), scale=pd.DataFrame.std(daily_returns_resid_period_AEP)),
         'r', lw=1.0, label='pdf')
plt.legend()
#########Ezeiza
plt.subplot(312)
plt.hist(daily_returns_resid_period_EZE.as_matrix(), bins=70, histtype='step', color='b', label='frequency', density='True')
plt.grid(True, alpha=0.3, linestyle='dashed')
plt.xlabel('daily returns of the residuals')
plt.ylabel('frequency')
plt.title('Ezeiza')
x = pd.DataFrame(np.linspace(plt.axis()[0], plt.axis()[1]))
plt.plot(x, scs.norm.pdf(x, loc=pd.DataFrame.mean(daily_returns_resid_period_EZE), scale=pd.DataFrame.std(daily_returns_resid_period_EZE)),
         'r', lw=1.0, label='pdf')
plt.legend()
#########Buenos Aires
plt.subplot(313)
plt.hist(daily_returns_resid_period_BA.as_matrix(), bins=70, histtype='step', color='b', label='frequency', density='True')
plt.grid(True, alpha=0.3, linestyle='dashed')
plt.xlabel('daily returns of the residuals')
plt.ylabel('frequency')
plt.title('Buenos Aires')
x = pd.DataFrame(np.linspace(plt.axis()[0], plt.axis()[1]))
plt.plot(x, scs.norm.pdf(x, loc=pd.DataFrame.mean(daily_returns_resid_period_BA), scale=pd.DataFrame.std(daily_returns_resid_period_BA)),
        'r', lw=1.0, label='pdf')
plt.legend()

######qqplots
sm.qqplot(np.reshape(daily_returns_resid_period_AEP.as_matrix(), (len(daily_returns_resid_period_AEP), )), line='s')
sm.qqplot(np.reshape(daily_returns_resid_period_EZE.as_matrix(), (len(daily_returns_resid_period_EZE), )), line='s')
sm.qqplot(np.reshape(daily_returns_resid_period_BA.as_matrix(), (len(daily_returns_resid_period_BA), )), line='s')

normality_tests(daily_returns_resid_period_AEP)
normality_tests(daily_returns_resid_period_EZE)
normality_tests(daily_returns_resid_period_BA)

######################################################################################################################
######################################################################################################################

#We start calculating the monthly variance based in the quadratic variation of Tt (See Basawa & Prasaka Rao)
#####Aeroparque
monthly_sigma_AEP = daily_return(Aeroparquepd)**2
monthly_sigma_AEP = monthly_sigma_AEP.dropna(axis=0)
monthly_sigma_AEP = monthly_sigma_AEP.resample('M').mean()
#####Ezeiza
monthly_sigma_EZE = daily_return(Ezeizapd)**2
monthly_sigma_EZE = monthly_sigma_EZE.dropna(axis=0)
monthly_sigma_EZE = monthly_sigma_EZE.resample('M').mean()
#####Buenos Aires
monthly_sigma_BA = daily_return(Bairespd)**2
monthly_sigma_BA = monthly_sigma_BA.dropna(axis=0)
monthly_sigma_BA = monthly_sigma_BA.resample('M').mean()

#Here we have to deal with leap years
#####Aeroparque
leap_years_monthly_sigma_AEP = monthly_sigma_AEP[((monthly_sigma_AEP.index.month == 2) & (monthly_sigma_AEP.index.day == 29))]
leap_years_monthly_sigma_AEP = leap_years_monthly_sigma_AEP.shift(-1, freq='D')
monthly_sigma_AEP = monthly_sigma_AEP.combine_first(leap_years_monthly_sigma_AEP)
monthly_sigma_AEP = monthly_sigma_AEP[~((monthly_sigma_AEP.index.month == 2) & (monthly_sigma_AEP.index.day == 29))]

#####Ezeiza
leap_years_monthly_sigma_EZE = monthly_sigma_EZE[((monthly_sigma_EZE.index.month == 2) & (monthly_sigma_EZE.index.day == 29))]
leap_years_monthly_sigma_EZE = leap_years_monthly_sigma_EZE.shift(-1, freq='D')
monthly_sigma_EZE = monthly_sigma_EZE.combine_first(leap_years_monthly_sigma_EZE)
monthly_sigma_EZE = monthly_sigma_EZE[~((monthly_sigma_EZE.index.month == 2) & (monthly_sigma_EZE.index.day == 29))]

#####Buenos Aires
leap_years_monthly_sigma_BA = monthly_sigma_BA[((monthly_sigma_BA.index.month == 2) & (monthly_sigma_BA.index.day == 29))]
leap_years_monthly_sigma_BA = leap_years_monthly_sigma_BA.shift(-1, freq='D')
monthly_sigma_BA = monthly_sigma_BA.combine_first(leap_years_monthly_sigma_BA)
monthly_sigma_BA = monthly_sigma_BA[~((monthly_sigma_BA.index.month == 2) & (monthly_sigma_BA.index.day == 29))]

#we incorporate the monthly sigma to the data matrix
#####Aeroparque
data['Monthly Sigma Aeroparque'] = monthly_sigma_AEP
daily_sigma_AEP = data['Monthly Sigma Aeroparque']
daily_sigma_AEP = daily_sigma_AEP.fillna(method='bfill')
#####Ezeiza
data['Monthly Sigma Ezeiza'] = monthly_sigma_EZE
daily_sigma_EZE = data['Monthly Sigma Ezeiza']
daily_sigma_EZE = daily_sigma_EZE.fillna(method='bfill')
#####Buenos Aires
data['Monthly Sigma Buenos Aires'] = monthly_sigma_BA
daily_sigma_BA = data['Monthly Sigma Buenos Aires']
daily_sigma_BA = daily_sigma_BA.fillna(method='bfill')

#We calculate the monthly standard deviation 
monthly_var_AEP = Aeroparquepd.resample('M').std()
monthly_var_EZE = Ezeizapd.resample('M').std()
monthly_var_BA = Bairespd.resample('M').std()

#daily_quadratic_variation = pd.read_excel("C:/Users/turko/Google Drive/Academia/Maestría en Finanzas UDESA/Tesis/Weather Derivatives/Data/Daily_Quadratic_Variation.xlsx",index_col=0)
daily_quadratic_variation_AEP = pd.DataFrame(daily_sigma_AEP, index=daily_sigma_AEP.index)
daily_quadratic_variation_EZE = pd.DataFrame(daily_sigma_EZE, index=daily_sigma_EZE.index)
daily_quadratic_variation_BA = pd.DataFrame(daily_sigma_BA, index=daily_sigma_AEP.index)

#Here we estimate the coefficient for the reversion speed 
numerator_AEP = [(T_estpd_AEP.shift(1).as_matrix() - Aeroparquepd.shift(1).as_matrix())* (Aeroparque - np.reshape(T_est_AEP, Aeroparque.shape))]/daily_quadratic_variation_AEP.shift(1).as_matrix()
numerator_AEP = numerator_AEP[~np.isnan(numerator_AEP)]

numerator_EZE = [(T_estpd_EZE.shift(1).as_matrix() - Ezeizapd.shift(1).as_matrix())* (Ezeiza - np.reshape(T_est_EZE, Ezeiza.shape))]/daily_quadratic_variation_EZE.shift(1).as_matrix()
numerator_EZE = numerator_EZE[~np.isnan(numerator_EZE)]

numerator_BA = [(T_estpd_BA.shift(1).as_matrix() - Bairespd.shift(1).as_matrix())* (Buenos_Aires - np.reshape(T_est_BA, Buenos_Aires.shape))]/daily_quadratic_variation_BA.shift(1).as_matrix()
numerator_BA = numerator_BA[~np.isnan(numerator_BA)]

denominator_AEP = (T_estpd_AEP.shift(1).as_matrix() - Aeroparquepd.shift(1).as_matrix())* (Aeroparquepd.shift(1).as_matrix() - T_estpd_AEP.shift(1).as_matrix())/ daily_quadratic_variation_AEP.shift(1).as_matrix() 
denominator_AEP = denominator_AEP[~np.isnan(denominator_AEP)]

denominator_EZE = (T_estpd_EZE.shift(1).as_matrix() - Ezeizapd.shift(1).as_matrix())* (Ezeizapd.shift(1).as_matrix() - T_estpd_EZE.shift(1).as_matrix())/ daily_quadratic_variation_EZE.shift(1).as_matrix() 
denominator_EZE = denominator_EZE[~np.isnan(denominator_EZE)]

denominator_BA = (T_estpd_BA.shift(1).as_matrix() - Bairespd.shift(1).as_matrix())* (Bairespd.shift(1).as_matrix() - T_estpd_BA.shift(1).as_matrix())/ daily_quadratic_variation_BA.shift(1).as_matrix() 
denominator_BA = denominator_BA[~np.isnan(denominator_BA)]

reversion_speed_AEP = -np.log(sum(numerator_AEP) / sum(denominator_AEP))
reversion_speed_EZE = -np.log(sum(numerator_EZE) / sum(denominator_EZE))
reversion_speed_BA = -np.log(sum(numerator_BA) / sum(denominator_BA))

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

#Monte Carlo Simulation and Option Valuation

##Calculation of the daily quadratic variation monthly average
avg_quadratic_variation_AEP = daily_quadratic_variation_AEP.resample('M', convention= 'e', kind='period').mean()
avg_quadratic_variation_EZE = daily_quadratic_variation_EZE.resample('M', convention= 'e', kind='period').mean()
avg_quadratic_variation_BA = daily_quadratic_variation_BA.resample('M', convention= 'e', kind='period').mean()

months = np.arange(1, 13)
monthly_quadratic_variation_AEP = []
for month in months:
    df1 = pd.DataFrame(avg_quadratic_variation_AEP[pd.PeriodIndex(avg_quadratic_variation_AEP.index).month == month].mean())
    monthly_quadratic_variation_AEP.append(df1)
monthly_quadratic_variation_AEP = pd.concat(monthly_quadratic_variation_AEP, axis=0)
monthly_quadratic_variation_AEP.columns = ['Monthly Sigma Aeroparque']
monthly_quadratic_variation_AEP.index = months
monthly_quadratic_variation_AEP = np.sqrt(monthly_quadratic_variation_AEP)

##simulation of the paths following the SDE dT(t) = \kappa(g(t) - T(t))dt + \sigma(t)dW(t) with g(t) \equiv \frac{1}{\kappa}\frac{\partial f(t)}{\partial t}+f(t)
paths = 50000
valuation_period = pd.date_range(start='2018-05-01', end='2018-09-30', freq='D') 
rnd_matrix = pd.DataFrame(np.random.randn(len(valuation_period), paths), index=valuation_period)
t_0 = 8156 #number of days since 1996-01-01 to 2018-04-30
t_val = np.arange(t_0 + 1, t_0 + len(valuation_period) + 1)
t_val = np.reshape(t_val, (len(t_val), 1))
T_0 = params_period_AEP['const'] + params_period_AEP['t'] * t_0 + params_period_AEP['sinwt'] * np.sin(w * t_0) + params_period_AEP['coswt'] * np.cos(w * t_0) + params_period_AEP['sin2wt'] * np.sin(2 * w * t_0) + params_period_AEP['cos2wt'] * np.cos(2 * w * t_0) + params_period_AEP['sin73/4wt'] * np.sin((73/4) * w * t_0) + params_period_AEP['cos73/4wt'] * np.cos((73/4) * w * t_0)
T_det_AEP = params_period_AEP['const'] + params_period_AEP['t'] * t_val + params_period_AEP['sinwt'] * np.sin(w * t_val) + params_period_AEP['coswt'] * np.cos(w * t_val) + params_period_AEP['sin2wt'] * np.sin(2 * w * t_val) + params_period_AEP['cos2wt'] * np.cos(2 * w * t_val) + params_period_AEP['sin73/4wt'] * np.sin((73/4) * w * t_val) + params_period_AEP['cos73/4wt'] * np.cos((73/4) * w * t_val)
T_det_AEP = np.reshape(T_det_AEP, (len(T_det_AEP), 1))
g_0 = np.array(1 / reversion_speed_AEP) * [params_period_AEP['t'] + params_period_AEP['sinwt'] * w * np.cos(w * t_0) - params_period_AEP['coswt'] * w * np.sin(w * t_0) + params_period_AEP['sin2wt'] * 2 * w * np.cos(2 * w * t_0) - params_period_AEP['cos2wt'] * 2 * w * np.sin(2 * w * t_0) + params_period_AEP['sin73/4wt'] * (73/4) * w * np.cos((73/4) * w * t_0) - params_period_AEP['cos73/4wt'] * (73/4) * w * np.sin((73/4) * w * t_0)] + T_0
g_t = np.array(1 / reversion_speed_AEP) * (params_period_AEP['t'] + params_period_AEP['sinwt'] * w * np.cos(w * t_val) - params_period_AEP['coswt'] * w * np.sin(w * t_val) + params_period_AEP['sin2wt'] * 2 * w * np.cos(2 * w * t_val) - params_period_AEP['cos2wt'] * 2 * w * np.sin(2 * w * t_val) + params_period_AEP['sin73/4wt'] * (73/4) * w * np.cos((73/4) * w * t_val) - params_period_AEP['cos73/4wt'] * (73/4) * w * np.sin((73/4) * w * t_val)) + T_det_AEP
daily_sigma_for_sim_AEP = pd.DataFrame(index=valuation_period, columns=['sigma_t'])

for month in months: #this for loop generates the vector with the average sigma for every month displayed daily for simulation purposes
    daily_sigma_for_sim_AEP[pd.DatetimeIndex(daily_sigma_for_sim_AEP.index).month == month] = monthly_quadratic_variation_AEP.loc[month, 'Monthly Sigma Aeroparque']

T_AEP = np.zeros(shape=(len(valuation_period),paths))

for x in range(paths):
    T_1 = T_0 + reversion_speed_AEP * (g_0 - T_0) + daily_sigma_for_sim_AEP.as_matrix()[0]* rnd_matrix.as_matrix()[0, x]
    T_AEP[0, x] = T_1
    for i in range(1, len(valuation_period)):
         T_i =  T_det_AEP[i - 1] + reversion_speed_AEP * (g_t[i - 1] - T_det_AEP[i - 1]) + daily_sigma_for_sim_AEP.as_matrix()[i] * rnd_matrix.as_matrix()[i, x]
         T_AEP[i, x] = T_i

###HDD function
def HDD(variable):
    return np.maximum(18 - variable, 0)

##observed HDDs
HDD_AEP = HDD(Aeroparquepd)
HDD_EZE = HDD(Ezeizapd)
HDD_BA = HDD(Bairespd)

###CDD function
def CDD(variable):
    return np.maximum(variable - 18, 0)

##observed CDDs
CDD_AEP = CDD(Aeroparquepd)
CDD_EZE = CDD(Ezeizapd)
CDD_BA = CDD(Bairespd)

CDDs = CDD_AEP.join(CDD_EZE)
CDDs = CDDs.join(CDD_BA)
CDDs['Month'] = pd.DatetimeIndex(CDDs.index).month

cooling_seasons = CDDs[(CDDs['Month'] < 4) | (CDDs['Month'] > 10)]

HDDs = HDD_AEP.join(HDD_EZE)
HDDs = HDDs.join(HDD_BA)
HDDs['Month'] = pd.DatetimeIndex(HDDs.index).month

heating_seasons = HDDs[(HDDs['Month'] >= 5) & (HDDs['Month'] <= 9)]


#Below we plot the data for all stations
#plt.figure(figsize=(14, 10))
#plt.subplot(221)
#plt.plot(data['BARILOCHE'])
#plt.grid(True)
#plt.axis('tight')
#plt.ylabel('Temperature in °C')
#plt.title('Bariloche')
#plt.subplot(222)
#plt.plot(data['BUENOS AIRES'])
#plt.grid(True)
#plt.axis('tight')
#plt.title('Buenos Aires')
#plt.ylabel('Temperature in °C')
#plt.subplot(223)
#plt.plot(data['EZEIZA'])
#plt.grid(True)
#plt.axis('tight')
#plt.title('Ezeiza')
#plt.subplot(224)
#plt.plot(data['AEROPARQUE'])
#plt.grid(True)
#plt.axis('tight')
#plt.title('Aeroparque')


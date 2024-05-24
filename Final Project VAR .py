####
####
import pandas as pd
import csv
import time
import numpy as np
import yahooquery as yq
from datetime import timedelta, date
from arch import arch_model
import scipy.stats as stats
import requests
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import RuleInduction
import DataProcess
import KNN
import random

class datatable:
    # Corresponds to ticker.financial_data
    financial_keys = [
        'currentPrice', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice',
        'targetMedianPrice', 'recommendationMean', 'recommendationKey', 
        'totalCash', 'totalCashPerShare', 'ebitda',
        'totalDebt', 'quickRatio', 'currentRatio', 'totalRevenue',
        'debtToEquity', 'revenuePerShare', 'returnOnAssets', 'returnOnEquity',
        'grossProfits', 'freeCashflow', 'operatingCashflow', 'earningsGrowth',
        'revenueGrowth', 'grossMargins', 'ebitdaMargins', 'operatingMargins',
        'financialCurrency'
    ]
    # Corresponds to ticker.key_stats
    key_stats = [
        'priceHint', 'enterpriseValue', 'forwardPE', 'profitMargins',
        'floatShares', 'sharesOutstanding', 'sharesShort',
        'sharesShortPriorMonth', 'sharesShortPreviousMonthDate',
        'dateShortInterest', 'sharesPercentSharesOut', 'heldPercentInsiders',
        'heldPercentInstitutions', 'shortRatio', 'shortPercentOfFloat', 'beta',
        'impliedSharesOutstanding', 'category', 'bookValue', 'priceToBook',
        'fundFamily', 'legalType', 'lastFiscalYearEnd', 'nextFiscalYearEnd',
        'mostRecentQuarter', 'earningsQuarterlyGrowth', 'netIncomeToCommon',
        'trailingEps', 'forwardEps', 'pegRatio', 'lastSplitFactor',
        'lastSplitDate', 'enterpriseToRevenue', 'enterpriseToEbitda',
        '52WeekChange', 'SandP52WeekChange', 'lastDividendValue',
        'lastDividendDate'
    ]

    ## companySnapshot_sectorInfo is text
    # corresponds to ticker.p_company_360
    p_360_keys = [
        'innovations_score', 'innovations_sectorAvg', 'sustainability_totalScore',
        'sustainability_totalScorePercentile', 'sustainability_environmentScore',
        'sustainability_environmentScorePercentile', 'sustainability_socialScore',
        'sustainability_socialScorePercentile', 'sustainability_governanceScore',
        'sustainability_governanceScorePercentile', 'sustainability_controversyLevel',
        'companySnapshot_sectorInfo', 'companySnapshot_company_innovativeness',
        'companySnapshot_company_hiring', 'companySnapshot_company_sustainability',
        'companySnapshot_company_insiderSentiments', 'companySnapshot_company_earningsReports',
        'companySnapshot_company_dividends', 'companySnapshot_sector_innovativeness',
        'companySnapshot_sector_hiring', 'companySnapshot_sector_sustainability',
        'companySnapshot_sector_insiderSentiments', 'companySnapshot_sector_earningsReports',
        'companySnapshot_sector_dividends', 'dividend_amount', 'dividend_date', 'dividend_yield',
        'dividend_sectorMedian', 'dividend_marketMedian'
    ]
    calc_keys = [
        'relVaR', 'absVaR', 'relVaREWMA', 'absVaREWMA', 'relVaRGARCH', 'absVaRGARCH',
        'CAGR', 'volatility', 'sharpe', 'calmar']

    sdetail_keys = [
        'marketCap', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'fiftyDayAverage', 'twoHundredDayAverage'
    ]

    def __init__(self, companies=None):
        if companies == None:
            self.Data = pd.read_csv("DATASET.csv", index_col = 0)
            self.NoData = []
            for index, row in self.Data.iterrows():
                num_nans = row.isna().sum()
            
                if num_nans > len(self.financial_keys):
                    self.NoData.append(index)
                
            #print(self.Data.iloc[:, : 50])
            return
        
        starttime=time.time()
        self.NoData = []
        self.Data = pd.DataFrame(index=companies, columns=datatable.financial_keys
                                 + datatable.key_stats + datatable.p_360_keys
                                 + datatable.sdetail_keys + datatable.calc_keys)

        # KEEP THIS
        # Very important - when using premium features, this maintains an active session so we can change tickers
        # without having to log in again and again (which will result in an error quickly)
        r = yq.Research(username="placeholder",                        password="placeholder")
        self.Data['payoutRatio']=np.nan
        self.Data['annualizedDividend']=np.nan
        self.Data['dividends_all']=np.nan
        for company in companies:
            print("Adding: ", company)
            tmp = yq.Ticker(company, session=r.session)
            financials = tmp.financial_data[company]
            keystats = tmp.key_stats[company]
            p_360 = flatten(tmp.p_company_360[company])
            summarydetail = tmp.summary_detail[company]
            today = date.today()
            startdate = today - timedelta(days=3652)
            df = tmp.history(period="max", start=startdate, end=today)
            if(np.isnan(df['adjclose'][0]) == True or df['adjclose'][0] <= 0.0):
                print("Error: adjclose, stock:" + company)
                self.NoData.append(company)
                continue
                
            count = 0
            for key in datatable.financial_keys:
                try:
                    self.Data.at[company, key] = financials[key]
                except Exception:
                    count += 1
                    self.Data.at[company, key] = np.nan
                    pass

            for key in datatable.key_stats:
                try:
                    self.Data.at[company, key] = keystats[key]
                except Exception:
                    self.Data.at[company, key] = np.nan
                    count += 1
                    pass

            for key in datatable.p_360_keys:
                try:
                    self.Data.at[company, key] = p_360[key]
                except Exception:
                    self.Data.at[company, key] = np.nan
                    count += 1
                    pass

            for key in datatable.sdetail_keys:
                try:
                    self.Data.at[company, key] = summarydetail[key]
                except Exception:
                    self.Data.at[company, key] = np.nan
                    count += 1
                    pass

            relVaR, absVaR = value_at_risk(df)
            relVaREWMA, absVaREWMA = value_at_risk_ewma(df)
            relVaRGARCH, absVaRGARCH = value_at_risk_GARCH(df)
            stock_metrics = {
                "relVaR": relVaR,
                "absVaR": absVaR,
                "relVaREWMA": relVaREWMA,
                "absVaREWMA": absVaREWMA,
                "relVaRGARCH": relVaRGARCH,
                "absVaRGARCH": absVaRGARCH,
                "CAGR": CAGR(df),
                "volatility": volatility(df),
                "sharpe": sharpe_ratio(df),
                "calmar": calmar_ratio(df)}
            
            div, pr, ad= NasdaqScrape(company)
            try:
                if(np.isnan(div)):
                    self.Data.at[company, 'IsDivStock']=0
                    self.Data.at[company, 'dividends_all']=div
                else:
                    self.Data.at[company, 'IsDivStock']=1
                    pass
            except:
                self.Data.at[company, 'dividends_all']=[div]
                self.Data.at[company, 'IsDivStock']=1
           
            self.Data.at[company, 'payoutRatio']=pr
            self.Data.at[company, 'annualizedDividend']=ad
            
            for key in datatable.calc_keys:
                try:
                    self.Data.at[company, key] = stock_metrics[key]
                except Exception:
                    self.Data.at[company, key] = np.nan
                    count += 1
                    pass

            if count >= len(datatable.financial_keys): #If count is above a certain threshold, we remove the company
                self.NoData.append(company)
        end=time.time()
        
        print("time Taken: ", end-starttime)
        # Check if there are overlapping metrics
        # Filter out irrelevant

    # Append ys

# risk-free rate, based on 1-yr T-Bill interest rate
current_rf = 0.048

## function to suppress stdout/stderr via internet
## (used to avoid arch_model errors)
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Note: this is not a $ value, rather, the % value of the stock
# One-day value at risk
def value_at_risk(df):
    alpha = 0.05
    current_share_price = df['adjclose'][-1]

    Z_value = stats.norm.ppf(abs(alpha))
    mean_return_rate = df['adjclose'].pct_change().mean()
    std_return_rate = df['adjclose'].pct_change().std()

    rel_VaR = -1 * current_share_price * Z_value * std_return_rate
    abs_VaR = -1 * current_share_price * (Z_value * std_return_rate +
                                          mean_return_rate)

    return rel_VaR / current_share_price, abs_VaR / current_share_price


# EWMA (Exponential Weighted Moving Average) VaR
# Weighs
def value_at_risk_ewma(df):
    alpha = 0.05
    current_share_price = df['adjclose'][-1]

    Z_value = stats.norm.ppf(abs(alpha))

    # Estimate volatility using EWMA
    daily_drop_off = 0.94
    returns = df['adjclose'].pct_change()
    ewma_vol = np.sqrt(returns.ewm(alpha=daily_drop_off).var().iloc[-1])

    rel_VaR = -1 * current_share_price * Z_value * ewma_vol
    abs_VaR = -1 * current_share_price * (Z_value * ewma_vol + returns.mean())

    return rel_VaR / current_share_price, abs_VaR / current_share_price


# Adapted from Jon Danielsson's "Financial Risk Forecasting" (2011)
def value_at_risk_GARCH(data):
    try:
        df =data.copy()
        df['returns'] = np.log(data['adjclose']).diff()
        current_share_price = df['adjclose'][-1]
    
        alpha = 0.05
        Z_value = stats.norm.ppf(abs(alpha))
        mean_return_rate = df['adjclose'].pct_change().mean()
    
      # Fit GARCH(1,1) model
      # arch_model throws many errors due to missing optimisations on the Zoo
      # this is a hacky workaround.
        with suppress_stdout_stderr():
            print(df['returns'])
            am = arch_model(df['returns'][1:],
                            mean='Zero',
                            vol='Garch',
                            p=1,
                            o=0,
                            q=1,
                            dist='Normal',
                            rescale=False)
            res = am.fit(update_freq=5, disp='off')
        # Get omega, alpha, beta etc.
        omega = res.params.loc['omega']
        alpha = res.params.loc['alpha[1]']
        beta = res.params.loc['beta[1]']
        sigma2 = omega + alpha * df['returns'][len(df)-1]**2 + beta * res.conditional_volatility[-1]**2
    
        rel_VaR = -1 * current_share_price * np.sqrt(sigma2) * Z_value
        abs_VaR = -1 * current_share_price *  (np.sqrt(sigma2) * Z_value + mean_return_rate)
        return rel_VaR / current_share_price, abs_VaR / current_share_price
    except: 
        return np.nan, np.nan
    


# Cumulative Annual Growth Rate
def CAGR(data):
    # so we don't modify the base dataframe
    df = data.copy()
    df['daily_returns'] = df['adjclose'].pct_change()
    df['cumulative_returns'] = (1 + df['daily_returns']).cumprod()

    if len(df) == 2517:
        n = 10
    else:
        n = len(df) / 252

    cagr = (df['cumulative_returns'][-1])**(1 / n) - 1
    return cagr


def volatility(data):
    # so we don't modify the base dataframe
    df = data.copy()

    df['daily_returns'] = df['adjclose'].pct_change()
    vol = df['daily_returns'].std() * np.sqrt(252)
    return vol


def sharpe_ratio(df):
    sharpe = (CAGR(df) - current_rf) / volatility(df)
    return sharpe


def calmar_ratio(data):
    df = data.copy()

    df['daily_returns'] = df['adjclose'].pct_change()
    df['cumulative_returns'] = (1 + df['daily_returns']).cumprod()
    df['cumulative_max'] = df['cumulative_returns'].cummax()
    df['drawdown'] = df['cumulative_max'] - df['cumulative_returns']
    df['drawdown_pct'] = df['drawdown'] / df['cumulative_max']
    max_dd = df['drawdown_pct'].max()

    calmar = (CAGR(df) - current_rf) / max_dd
    return calmar


def NasdaqScrape(ticker):
    url = f'https://api.nasdaq.com/api/quote/{ticker}/dividends?assetclass=stocks'
    headers = {
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'User-Agent': 'Java-http-client/'
    }

    response = requests.get(url, headers=headers)

    try:
        df = pd.json_normalize(response.json()['data']['dividends']['rows'])
        df=df.values.tolist()
    except:
        df=np.nan
    try:
        payoutratio= pd.json_normalize(response.json()['data'])['payoutRatio']
        payoutratio=float(payoutratio[0])
    except:
        payoutratio=np.nan
    try:
        annualizedDividend= pd.json_normalize(response.json()['data'])['annualizedDividend']
        annualizedDividend=float(annualizedDividend[0])
    except:
        annualizedDividend=np.nan
    return df, payoutratio, annualizedDividend



# Figure out which metrics we need time series data for

# Calculate Predictors, append to self.Data

with open('Russell3000.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    russell3000_tickers = []
    next(reader)
    for row in reader:
        #russell3000_tickers.append(row[0]) #for test data
        russell3000_tickers.append(row[0][0:-1]) 

#To scrape, run:
#x = datatable(russell3000)
#To extract from pre-scraped and provided DATASET.CSV
x = datatable()

#%%
#DataPreProcessing

#Drop marked companies
def PruneFeatures(x):
    try:
        x.Data=x.Data.drop(x.NoData, axis='index')
    except:
        pass
    
    data=x.Data.drop('dividends_all', axis=1)
    print("Removing irrelevant columns such as dividend dates, hiring and so forth: ")
    tmp=['dividend_date', 'lastDividendDate', 'lastSplitDate', 'dateShortInterest', 'sharesShortPreviousMonthDate', 'companySnapshot_company_hiring', 'companySnapshot_company_insiderSentiments']
    for j in tmp:
        try:
            data=data.drop(j, axis=1)
            print(f'Dropped feature: {j}')
        except:
            pass

    ycol=['relVaR', 'absVaR', 'relVaREWMA', 'absVaREWMA', 'relVaRGARCH', 'absVaRGARCH', 'CAGR', 'volatility', 'sharpe', 'calmar']
    xcol=[i for i in list(data.columns) if i not in ycol]
    #Combine features which have correlation>0.95 AND average of abs(corrdiff) with other features <0.1
    data=DataProcess.ReduceFeatures(data, xcol=xcol)
    xcol=[i for i in list(data.columns) if i not in ycol]
    corrmatrix, sortedcorr=DataProcess.crosscorr(table=data, xcol=xcol, ycol=ycol, method='spearman')
    print("Correlation Matrix: ")
    print(corrmatrix)
    #drop out features which are horizontally NANS in the X-Y corr matrix
    drop=[]
    features=corrmatrix.index.values
    for feature in features:
        if corrmatrix.loc[feature].isna().all()==True:
            drop.append(feature)
    data=data.drop(drop, axis=1)
    xcol=[i for i in list(data.columns) if i not in ycol]
    #Calculate percentage of rows for each features which are NAN, ignoring dividend stocks
    nan_percentages = data.isna().mean() * 100
    print("Percentage of NaNs for features with NaNs: \n")
    for i in range(len(nan_percentages)):
        if nan_percentages.iloc[i]!=0:
            print(list(nan_percentages.keys())[i],': ', round(nan_percentages.iloc[i], 2), '%')
    print(data.head())
    return data

#Generate Local Ruleset

def Predict(x, data, ticker, output=False, intervals=5, K=[40, 40, 5]):
    if ticker in data.index.values:
        pass
    else:
        print(f'Insufficent data found from the YahooQuery API, so {ticker} was pruned')
        return False, False, False, False
    if output:
        ycol=output
    else:
        ycol=['relVaR', 'absVaR', 'relVaREWMA', 'absVaREWMA', 'relVaRGARCH', 'absVaRGARCH', 'CAGR', 'volatility', 'sharpe', 'calmar']
    
    try:
        data=data.drop('IsDivStock', axis=1)
    except:
        pass   
    corrmatrix, sortedcorr=DataProcess.crosscorr(table=data, xcol=xcol, ycol=ycol, method='spearman')
    corrmatrix1, sortedcorr1=DataProcess.crosscorr(table=data, xcol=xcol, ycol=ycol, method='pearson')
    # print("Number of Dividend Stocks is:", round(np.sum(data['IsDivStock'])), 'out of',len(data))
    #Feature Selection based on the correlation=importance assumption
    top_metrics=DataProcess.pickfeatures(sortedcorr, num=12, threshold=0.5)
    features=list(set(RuleInduction.flatten([list(top_metrics[j].keys()) for j in ycol])))
    print("Final features: ", features)
    data['companySnapshot_sectorInfo']=x.Data['companySnapshot_sectorInfo']
    try:
        ticker_sector = data["companySnapshot_sectorInfo"][ticker]
    except:
        ticker_sector = None

    try:
        ticker_mcap = data["marketCap"][ticker]
    except:
        ticker_mcap = None 
    #Filter by marketcap
    if ticker_sector and ticker_mcap:
       data_norm=DataProcess.normalisedata(data[(data['companySnapshot_sectorInfo'] == ticker_sector) | ((ticker_mcap*2 > data['marketCap']) & (data['marketCap'] > ticker_mcap/2))], 'zscore')
    elif ticker_sector:
       data_norm=DataProcess.normalisedata(data[(data['companySnapshot_sectorInfo'] == ticker_sector)], 'zscore')
    elif ticker_mcap:
       data_norm=DataProcess.normalisedata(data[ (ticker_mcap*2 > data['marketCap']) & (data['marketCap'] > ticker_mcap/2)], 'zscore')
    else:
       data_norm=DataProcess.normalisedata(data, 'zscore')
       
    data_norm=data_norm[features+ycol]
    #First, find the nearest neighbors:
    print("Calculating distance matrices:")
    rtn_euclidean=KNN.kNearest(table=data_norm, xcol=features, k=K[0], dist_metric='euclidean', p=2 )
    rtn_minkowski=KNN.kNearest(table=data_norm, xcol=features, k=K[1], dist_metric='minkowski', p=4 )
    rtn_cosine=KNN.kNearest(table=data_norm, xcol=features, k=K[2], dist_metric='cosine' )
    
    Examples=list(set(RuleInduction.flatten([i[0] for i in rtn_euclidean[ticker]])).union(set(RuleInduction.flatten([i[0] for i in rtn_cosine[ticker]]))).union(set(RuleInduction.flatten([i[0] for i in rtn_minkowski[ticker]]))))
    #INDUCE RULES from these nearest neighbors
    Neighbors=data_norm[features+ycol].loc[Examples]
    Neighbors=Neighbors.dropna(axis=1, how='all')
    rows=Neighbors.index.values
    rows=[row for row in rows if Neighbors.loc[row].isna().mean()==0]
    cleaned=Neighbors.loc[rows]
    discretized=DataProcess.discretisation(cleaned, num=intervals)
    attributes=[j for j in discretized.columns if j not in ycol]
    check=len(attributes)
    if(check<=1):
        print("Rule Induction not possible due to sparseness of data from YahooQuery Data ")
        return False, False, False, True
    cover_min, Rules=RuleInduction.main(discretized, attributes=attributes, decisions=ycol)
    if cover_min==False or Rules==False:
        print("Rule Induction not possible")
        efficiency=False
    else:
        print(f'\nSuccess: Local-Rule estimation of stock {ticker}')
        efficiency=1-len(cover_min)/len(discretized.columns)
        print('Nearest Neighbors are: ', list(discretized.index.values))
    return cover_min, Rules, efficiency, flag



#Run main
ycol=['relVaR', 'absVaR', 'relVaREWMA', 'absVaREWMA', 'relVaRGARCH', 'absVaRGARCH', 'CAGR', 'volatility', 'sharpe', 'calmar']
data=PruneFeatures(x)
xcol=[i for i in list(data.columns) if i not in ycol]
companies=data.index.values
ticker='IBM'

efficiency=False
timeout = time.time() + 60*5 #5mins from now
flag=True
while(efficiency==False and flag==True):
    intervals, ks=random.randint(4, 9), [random.randint(30, 45), random.randint(30, 45), random.randint(5, 10) ]
    intervals=8
    ks=[35,30,5]
    cover_min, Rules, efficiency, flag=Predict(x, data, ticker, output=['absVaR', 'sharpe'],  intervals=8, K=[35,30,5])
    if time.time() > timeout:
        break

if Rules:
    print("A locally minimal covering set of features with efficiency", efficiency, " is: ", cover_min)
    print("A local covering set of Rules is: ")
    for key, value in Rules.items():
        print(key, "---->", value)
    print(f'Parameters used are: {intervals} intervals for data discretisation and K={ks}')
    

#%%


#testing rule induction:
# features=list(sortedcorr['absVaRGARCH'].keys())[-10:]
# tickers=['GOOG', 'MSFT', 'TSLA', 'WMT', 'META', 'V', 'PG', 'VZ', 'MA', 'NVDA', 'UNH', 'DIS']
# features=['currentRatio','earningsGrowth','sharesPercentSharesOut','beta','earningsQuarterlyGrowth']
# ys=['relVaR']

#NEED TO REMOVE NANS
#Calculate Non-Nan Rows
# rows=discretized.index.values
# rows=[row for row in rows if discretized.loc[row].isna().mean()==0]
# cleaned_discretized=discretized.loc[rows]
# B=RuleInduction.elementaryset(cleaned_discretized, attributes=features)
# D=RuleInduction.elementaryset(cleaned_discretized, attributes=ys)
# if RuleInduction.consistent(B=B, D=D):
#     print('Dataset is consistent')
# else:
#     print("Dataset is not consistent")
# cover_min=RuleInduction.smallestcover(cleaned_discretized, D, features)
#Ways to deal with inconsistency: 1. delete rows, 2. change features

# RuleInduction.inferrule(cleaned_discretized, subcover=cover_min, decisions=ys)





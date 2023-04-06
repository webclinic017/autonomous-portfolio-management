import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def load_and_preprocess_data():

    np.random.seed(42)
    idx = pd.IndexSlice

    meta = pd.read_hdf('/home/groovyjac/projects/autonomous-portfolio-management/main_data_store_JDKv1.h5',
                        'stocks/base_fundamentals')

    selected_features = ['MarketCapitalization', 'PERatio', 'Beta', 'EBITDA', 'RevenueTTM',
             'GrossProfitTTM', 'OperatingMarginTTM', 'ReturnOnAssetsTTM', 'ReturnOnEquityTTM', 'PriceSalesTTM',
       'PriceBookMRQ', 'EnterpriseValue', 'EnterpriseValueRevenue',
       'EnterpriseValueEbitda', 'PercentInstitutions','PayoutRatio', '52WeekHigh', '52WeekLow', '50DayMA', '200DayMA',
       'ForwardPE', 'SharesFloat', 'PercentInsiders', 'ShortPercent',
       'ForwardAnnualDividendRate', 'ForwardAnnualDividendYield', 'TrailingPE',
       'DividendShare',]
   

    df = pd.DataFrame(meta.loc[idx[:, :, selected_features]])
    df = df.droplevel(1)[~df.droplevel(1).index.duplicated(keep='first')]
    df = df.unstack(level=1)
    df.columns = df.columns.droplevel(0)
    #df = df.drop(['Sector', 'GicSector', 'Industry', 'GicGroup', 'GicIndustry', 'GicSubIndustry'], axis=1)
    df = df.astype('float64')
    
    # Check for missing values
    print(df.isna().sum())
    
    # Fill in missing values with KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = imputer.fit_transform(df)
    df = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
    
    # EDA plots
    # sns.set(style='whitegrid', palette='muted')
    # df.hist(figsize=(20, 20), layout=(11, 4))
    # plt.tight_layout()
    # plt.show()
    
    # sns.pairplot(df.sample(1000, replace=True))
    # plt.show()
    
    # plt.figure(figsize=(20, 20))
    # sns.boxplot(data=df)
    # plt.xticks(rotation=90)
    # plt.show()
    
    # corr = df.corr()
    # plt.figure(figsize=(20, 20))
    # sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    # plt.show()
    
    # Feature selection for DEC - Use all features for now, as feature selection for DEC is problem-specific
    selected_features_final = df.columns

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[selected_features_final])

    return scaled_data
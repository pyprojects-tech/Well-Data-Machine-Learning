import streamlit as st

#import regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

#import regression analysis models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.io as pio
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt

pio.templates.default = "ggplot2"
plt.style.use('dark_background')

st.set_page_config(layout="wide")

#Start of App Including Type and Raw Data
st.title('Maine Well Data Machine Learning App')

@st.cache(allow_output_mutation=True)
def data_import():
    data =  pd.read_csv('data3.csv',low_memory=False)
    
    data0= data[['LATITUDE','LONGITUDE','WELL_DEPTH_FT','CASING_LENGTH_FT','WELL_YIELD_GPM','UNIT_AGE','ROCKTYPE1','ROCKTYPE2']].copy()
    data0['x'] = np.cos(data0['LATITUDE'])*np.cos(data0['LONGITUDE'])
    data0['y'] = np.cos(data0['LATITUDE'])*np.sin(data0['LONGITUDE'])
    data0['z'] = np.sin(data0['LATITUDE'])
    
    data['x'] = np.cos(data0['LATITUDE'])*np.cos(data0['LONGITUDE'])
    data['y'] = np.cos(data0['LATITUDE'])*np.sin(data0['LONGITUDE'])
    data['z'] = np.sin(data0['LATITUDE'])
    
    return data0,data

data0 =  data_import()[0]
data =  data_import()[1]

col001, col002, col003  = st.beta_columns([1,1,1])
with col001:
    x_params_num = st.multiselect('Numerical Input Values', data.columns)
with col002:
    x_params_cat = st.multiselect('Categorical Input Values', data.columns)
with col003:
        y_params = st.selectbox('Model Parameter', data.columns)
        


df = pd.DataFrame()
df[x_params_num] = data[x_params_num]
df[x_params_cat] = data[x_params_cat]
df[y_params] = data[y_params]

col11, col12, col13= st.beta_columns([1,1,1])

min_val = float(np.min(df[y_params]))
max_val = float(np.max(df[y_params]))

with col11:
    min_filt = st.slider('Minimum Filter Value for Model Parameter',
                         min_value = min_val,
                         max_value = max_val,
                         value = min_val,
                         step = 1.0,
                         )
with col12:
   max_filt = st.slider('Maximum Filter Value for Model Parameter',
                         min_value = min_val,
                         max_value = max_val,
                         value = max_val,
                         step = 1.0,
                         )
df = df[(df[y_params]>min_filt) & (df[y_params]<max_filt)]

categorical_cols = x_params_cat
data_g0 = pd.get_dummies(df, columns=categorical_cols)

#Build x and y data sets
data_drop = data_g0.dropna()
datax = data_drop.drop(columns=y_params,axis=1)
datay= data_drop[y_params]


#datay_drop = datay.drop(labels=['LATITUDE','LONGITUDE','WELL_DEPTH_FT','CASING_LENGTH_FT','WELL_YIELD_GPM'],axis=1)

#Create a pipeline to call all of the transformations, when the pipeline is called it executes the steps sequentially

t_size = st.sidebar.slider('Select Training Set Size %',0,100,50,key='t_size')

@st.cache(allow_output_mutation=True)
def pipeline():
    num_pipeline_minmax = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('std_scaler',MinMaxScaler())
    ])

    
    train_setx,test_setx,train_sety,test_sety = train_test_split(
        num_pipeline_minmax.fit_transform(datax),
        datay,
        test_size=t_size/100,random_state=42
        )
    
    return train_setx,test_setx,train_sety,test_sety
    
train_setx,test_setx,train_sety,test_sety = pipeline()

data_expand = st.beta_expander("Show Data Table",expanded=True)

with data_expand:
    #Import the data table for viewing that will be modeled
    st.subheader('Data Table - '+
                 'Columns: '+str(len(df.columns))+
                 ', Rows: '+str(len(df[y_params]))
                 )
    df

#Build Regression Selection to select regression type

reg_list = ['LinearRegression','KNeighborsRegressor','DecisionTreeRegressor','RandomForestRegressor']
fit_type = st.sidebar.selectbox('Select Regression Type',reg_list)


#Build the regression model and call the regression
reg0 = globals()[fit_type]()
params = reg0.get_params()

d={}
d2={}

for key,value in params.items():
   
    d[key] = st.sidebar.text_input(key, value)
    
    if params[key] == None:
        d2[key] = None
    else: 
        d2[key] = type(params[key])(d[key])
    

    def regression(**kwargs):
        reg = globals()[fit_type](**kwargs)
        reg.fit(train_setx,train_sety)
        return reg

reg = regression(**d2)

def error_calcs():
    well_predictions = reg.predict(test_setx)
    lin_mse = mean_squared_error(test_sety, well_predictions)
    lin_rmse = np.sqrt(lin_mse)

    return lin_rmse


#Build Columns for regression fit data and plotting 
col01, col02, col03 = st.beta_columns([1,1,1])

with col01:    
    n = st.slider('Number of Data Points to Plot',10,1000)
   
with col02:
    st.write('Regression Type: '+fit_type)
   
with col03:
    st.write("RMSE Score: "+str(np.round(error_calcs(),2)))
    
col1, col2  = st.beta_columns([1,1])
   
ydat = reg.predict(test_setx)[:n]
xdat = test_sety[:n]  

df1 = pd.DataFrame()
df2 = pd.DataFrame()
df_hist  = pd.DataFrame()

df1['data']=xdat
df1['label']='actual'

df2['data']=ydat
df2['label']='predicted'

df_plot = pd.concat([df1,df2])

df_hist['actual']=xdat
df_hist['predicted']=ydat

#Scatterplot for fit performance
with col1:
    
    plot = sns.regplot(x=df_plot['data'][df_plot['label'] =='actual'],
                  y=df_plot['data'][df_plot['label'] =='predicted'],
                  )
    plot.set_xlim(left=min_filt, right=max_filt)
    plot.set_ylim(bottom=min_filt, top=max_filt)
    plot.set(xlabel='Actual Value', ylabel='Predicted Value',title='Regression Fit Scatterplot')
    fig = plot.get_figure()
    
    st.pyplot(fig,clear_figure=True)
        

#Histogram for fit performance
with col2:
    
    plot2 = sns.histplot(df_hist,kde=True)
    plot2.set(xlabel=str(y_params), title='Regression Fit Histogram')
    fig2 = plot2.get_figure()
    
    st.pyplot(fig2,clear_figure=True)
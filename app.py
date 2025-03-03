import streamlit as st

#import regression models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import SGDRegressor
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.neural_network import MLPRegressor

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
from sklearn.model_selection import GridSearchCV


from itertools import product

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm 
import matplotlib


import seaborn as sns

import ast 

st.set_page_config(layout="wide")
plt.style.use('dark_background')
#matplotlib.pyplot.ion

reg_list = ['LinearRegression','KNeighborsRegressor','DecisionTreeRegressor','RandomForestRegressor',
            'RANSACRegressor','TheilSenRegressor','HuberRegressor',
            'LinearSVR', 'SVR','NuSVR', 'GaussianProcessRegressor', 'RadiusNeighborsRegressor',
            'MLPRegressor','Ridge','KernelRidge','ElasticNet','SGDRegressor', 'PassiveAggressiveRegressor'
            ]

re_list = reg_list.sort()

#Start of App Including Type and Raw Data
st.title('Machine Learning Data Testing App')
st.set_option('deprecation.showPyplotGlobalUse', False)

file =  st.file_uploader('Upload your data file (must be CSV)')

if file is not None:
    data = pd.read_csv(file)
    
    params_num_col = data.select_dtypes(np.number)
    params_cat_col = data.select_dtypes(object)
    
    col001, col002, col003,col004  = st.beta_columns([2,2,1,1])
    with col001:
        x_params_num = st.multiselect('Numerical Input Values', params_num_col.columns)
    with col002:
        x_params_cat = st.multiselect('Categorical Input Values', params_cat_col.columns)
    with col003:
        y_params = st.selectbox('Model Parameter', params_num_col.columns)
    with col004:
       st_dev_filt = st.number_input('Std. Deviation Filter to Remove Values > x St. Dev',
                            min_value = 0, 
                            max_value = 100,
                            value = 3,
                            step=1)
       
      
    df = pd.DataFrame()
    
    if x_params_num is not None:
        df[x_params_num] = data[x_params_num]
    if x_params_cat is not None:
        df[x_params_cat] = data[x_params_cat]
    if y_params is not None:
        df[y_params] = data[y_params]
    
    df = df[(np.abs(df-df.mean()) <= (st_dev_filt*df.std()))].dropna()
    
       # if y_params is not None: 
       #     min_val = float(np.min(df[y_params]))
       #     max_val = float(np.max(df[y_params]))
           
    # with col12:
    #     min_filt = st.number_input('Minimum Filter Value for Model Parameter, Data Min: '+str(np.round(min_val,2)),
    #                          min_value = min_val,
    #                          max_value = max_val,
    #                          value = min_val,
    #                          step = 1.0,
    #                          )
    # with col13:
    #    max_filt = st.number_input('Maximum Filter Value for Model Parameter, Data Min: '+str(np.round(max_val,2)),
    #                          min_value = min_val,
    #                          max_value = max_val,
    #                          value = max_val,
    #                          step = 1.0,
    #                          )
    
    #df = df[(df[y_params]>min_filt) & (df[y_params]<max_filt)]
    categorical_cols = x_params_cat
    data_g0 = pd.get_dummies(df, columns=categorical_cols)
     
    data_expand = st.beta_expander("Show Data Table",expanded=True)
        
    with data_expand:
        #Import the data table for viewing that will be modeled
        st.subheader('Data Table - '+
                     'Columns: '+str(len(df.columns))+
                     ', Rows: '+str(len(df[y_params]))
                     )
        df
        
    #Build x and y data sets
    data_drop = data_g0.dropna()
    datax = data_drop.drop(columns=y_params,axis=1)
    datay= data_drop[y_params]

    #Create a pipeline to call all of the transformations, when the pipeline is called it executes the steps sequentially
    
    t_size = st.sidebar.slider('Select Training Set Size %',0,100,50,key='t_size')
    
    #@st.cache(allow_output_mutation=True)
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
    
    if x_params_num or x_params_cat and y_params is not None: 
        train_setx,test_setx,train_sety,test_sety = pipeline()
       
        
        #Build Regression Selection to select regression type
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
            
        regplots_expand = st.beta_expander("Show Regression Plots",expanded=False)
        #Build Columns for regression fit data and plotting 
    
        n=10
        with regplots_expand:
            col01, col02, col03 = st.beta_columns([1,1,1])
        
            with col01:    
                n = st.number_input('Number of Data Points to Plot',10,10000,10)
               
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
                
               # plot.set_xlim(left=min_filt, right=max_filt)
               # plot.set_ylim(bottom=min_filt, top=max_filt)
                plot.set(xlabel='Actual Value', ylabel='Predicted Value',title='Regression Fit Scatterplot')
                fig = plot.get_figure()
                
                st.pyplot(fig,clear_figure=True)
                    
            
            #Histogram for fit performance
            with col2:
                
                plot2 = sns.histplot(df_hist,kde=True)
                plot2.set(xlabel=str(y_params), title='Regression Fit Histogram')
                fig2 = plot2.get_figure()
                st.pyplot(fig2,clear_figure=True)
        
        
        grid_expand = st.beta_expander("Show Grid Search Optimization",expanded=False)
        with grid_expand:
            grid_params = st.multiselect('Select GridSearch Params', d2.keys())
            grid_dict = {}
            grid_but = {}
            state=[]
            
            for keys in grid_params:
                i=0
                col = []
                
                for i in range(len(grid_params)):
                    col.append(str('col')+str(i+1))
                col  = st.beta_columns(i+1)
                
            j=0
            for keysj in grid_params:
                if params[keysj] == None:      
                    with col[j]:
                        grid_but[keysj] = st.text_input(keysj, [0,0],key=str('A')+str(j))
                        grid_dict[keysj] = ast.literal_eval(grid_but[keysj])
                        j+=1 
                else:
                    with col[j]:
                        grid_but[keysj] = st.text_input(keysj, [0,0],key=str('A')+str(j))
                        grid_dict[keysj] = ast.literal_eval(grid_but[keysj])
                        j+=1
            
            grid_df=[]
            for keysi in grid_params:
                grid_df.append('param_'+keysi)
            grid_df.append('rank_test_score')
            
            
            grid_button = st.button('Randomized Grid Search Optimization')
            @st.cache(suppress_st_warning=True)
            def grdsearch():
                if grid_button:
                    gsearch = GridSearchCV(reg,grid_dict)
                    gsearch.fit(train_setx,train_sety)
                    gresults = pd.DataFrame(gsearch.cv_results_)
                    st.write(gresults[grid_df])
                    
            grdsearch()
            
            
            #Build a new test and training set for graphing/manipulation purposes
            df_par= df.dropna()
            
            train_setx0,test_setx0,train_sety0,test_sety0 = train_test_split(
                df_par,
                df_par[y_params],
                test_size=t_size/100,random_state=42
                )
            
            def reg_par(xpar,ypar,dat):
                 regpar = globals()[fit_type](**d2)
                 regpar.fit(xpar,ypar)          
                 return regpar.predict(dat)
         
                
        xyzplot_expand = st.beta_expander("Show Parameter 3D Plotting",expanded=False)
        with xyzplot_expand:
            col001a, col002a, col003a = st.beta_columns([1,1,1])
            
            with col001a:
                n2 = st.number_input('Number^2 of Data Points to Plot',5,100)
            with col002a:
                bar_factor = st.slider('Select Bar Size Factor',1,20,5,key='bar_size')
            with col003a:
                transparency = st.slider('Select Bar Transparency',0.0,1.0,.5,key='alpha')
     
            col001, col002  = st.beta_columns([1,1])
            with col001:
                model_par_1 = st.multiselect('Input Parameters for Model 1', x_params_num,key='par1')
                chart_angle =st.slider('Select Chart Rotation Angle',0,360,30,key='angle')
                
                if len(model_par_1)==2:
                    x = np.linspace(df[model_par_1[0]].min(),df[model_par_1[0]].max(),n2)
                    y = np.linspace(df[model_par_1[1]].min(),df[model_par_1[1]].max(),n2)
                    
                    df_par = pd.DataFrame()
                    df_par['x'] = x
                    df_par['y'] = y
        
                    prod = product(df_par['x'].unique(), df_par['y'].unique())
                    z_cols = [x for x in df_par.columns if x not in ('x', 'y')]
                    z = df_par[z_cols].drop_duplicates().values.tolist()
                    
                    df_par_full = pd.DataFrame([s + list(p) for p in prod for s in z],
                                       columns=list(z_cols+['x', 'y'])).sort_values(list(z_cols+['x', 'y'])).drop_duplicates()
                    
                    
                    df_par_full['z'] = reg_par(test_setx0[model_par_1],test_sety0,df_par_full)
                    df_final = df_par_full.pivot_table(values='z', index='x', columns='y')
                    
                    X,Y = np.meshgrid(df_final.index,df_final.columns)
                    Z = np.array(df_final.values)
                    
                    Xi = X.flatten()
                    Yi = Y.flatten()
                    Zi = np.ones(Z.size)*np.min(Z)
             
                    dx = (np.max(X)-np.min(X))/(len(Xi))*bar_factor*np.ones(Z.size)
                    dy = (np.max(Y)-np.min(Y))/(len(Yi))*bar_factor*np.ones(Z.size)
                    dz = Z.flatten()-np.ones(Z.size)*np.min(Z)
                    
                    fig3 = plt.figure(figsize=(12,6))
                    ax3 = Axes3D(fig3)
                    
                    cmap = cm.get_cmap('jet') # Get desired colormap
                    max_height = np.max(dz)   # get range of colorbars
                    min_height = np.min(Zi)
    
                    # scale each z to [0,1], and get their rgb values
                    
                    rgba = [cmap((k-min_height)/max_height) for k in dz] 
                    
                    surf3 = ax3.bar3d(Xi,Yi,Zi,dx,dy,dz,color=rgba,alpha = transparency)
                    
                    #fig3.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
                    norm = matplotlib.colors.Normalize(vmin=min_height, vmax=max_height, clip=False)
                    fig3.colorbar(cm.ScalarMappable(norm = norm ,cmap=cmap), ax=ax3, shrink=0.75, aspect=10)
                    ax3.set_xlabel(model_par_1[0])
                    ax3.set_ylabel(model_par_1[1])
                    ax3.set_zlabel(y_params)
                    ax3.view_init(azim=chart_angle)
                    st.pyplot(fig3,clear_figure=True)
                                        
     
                 
                    # surf3 = ax3.plot_surface(X,Y,df_final.values,cmap='viridis')
                    # fig3.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
                    # ax3.set_xlabel(model_par_1[0])
                    # ax3.set_ylabel(model_par_1[1])
                    # ax3.set_zlabel(y_params)
                    # st.pyplot(fig3,clear_figure=True)
            
            with col002:
                
                model_par_2 = st.multiselect('Input Parameters for Model 2', x_params_num,key='par2')
                chart_angle2 =st.slider('Select Chart Rotation Angle',0,360,30,key='angle2')
                
                if len(model_par_2)==2:
                    x2 = np.linspace(df[model_par_2[0]].min(),df[model_par_2[0]].max(),n2)
                    y2 = np.linspace(df[model_par_2[1]].min(),df[model_par_2[1]].max(),n2)
                    
                    df_par2 = pd.DataFrame()
                    df_par2['x'] = x2
                    df_par2['y'] = y2
        
                    prod = product(df_par2['x'].unique(), df_par2['y'].unique())
                    z_cols = [x2 for x2 in df_par2.columns if x2 not in ('x', 'y')]
                    z2 = df_par2[z_cols].drop_duplicates().values.tolist()
                    
                    df_par_full2 = pd.DataFrame([s + list(p) for p in prod for s in z2],
                                       columns=list(z_cols+['x', 'y'])).sort_values(list(z_cols+['x', 'y'])).drop_duplicates()
                    
                    
                    df_par_full2['z'] = reg_par(test_setx0[model_par_2],test_sety0,df_par_full2)
                    df_final2 = df_par_full2.pivot_table(values='z', index='x', columns='y')
                    X2,Y2 = np.meshgrid(df_final2.index,df_final2.columns)
                    
                    Z2 = np.array(df_final2.values)
                    
                    Xi2 = X2.flatten()
                    Yi2 = Y2.flatten()
                    Zi2 = np.ones(Z2.size)*np.min(Z2)
                    
                    dx2 = (np.max(X2)-np.min(X2))/(len(Xi2))*bar_factor*np.ones(Z2.size)
                    dy2 = (np.max(Y2)-np.min(Y2))/(len(Yi2))*bar_factor*np.ones(Z2.size)
                    dz2 = Z2.flatten()-np.ones(Z2.size)*np.min(Z2)
                    
                    fig4 = plt.figure(figsize=(12,6))
                    ax4 = Axes3D(fig4)
                    
                    cmap = cm.get_cmap('jet') # Get desired colormap
                    max_height2 = np.max(dz2)   # get range of colorbars
                    min_height2 = np.min(Zi2)
    
                    # scale each z to [0,1], and get their rgb values
                    rgba2 = [cmap((k-min_height2)/max_height2) for k in dz2] 
                    
                    surf4 = ax4.bar3d(Xi2,Yi2,Zi2,dx2,dy2,dz2,color=rgba2,alpha=transparency)
                    norm2 = matplotlib.colors.Normalize(vmin=min_height2, vmax=max_height2, clip=False)
                    fig4.colorbar(cm.ScalarMappable(norm = norm2 ,cmap=cmap), ax=ax4, shrink=0.75, aspect=10)
                    
                    ax4.set_xlabel(model_par_2[0])
                    ax4.set_ylabel(model_par_2[1])
                    ax4.set_zlabel(y_params)
                    ax4.view_init(azim=chart_angle2)
                    st.pyplot(fig4,clear_figure=True)
                    
                    # Z2 = df_final.values
                    
                    # fig4 = plt.figure(figsize=(12,6))
                    # ax4 = Axes3D(fig4)
                    
                    # surf4 = ax4.plot_surface(X2,Y2,Z2,cmap='viridis')
                    # fig4.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)
                    # ax4.set_xlabel(model_par_2[0])
                    # ax4.set_xlim(np.min(X2),np.max(X2))
                    # ax4.set_ylabel(model_par_2[1])
                    # ax4.set_ylim(np.min(Y2),np.max(Y2))
                    # ax4.set_zlabel(y_params)
                    # st.pyplot(fig4,clear_figure=True)
        
        pair_plots = st.beta_expander("Show Pairplots for Comparative Data Analysis",expanded=False)
        #Build Columns for regression fit data and plotting 
        with pair_plots:
                
                sns.pairplot(df)
                st.pyplot(clear_figure=True)                        
                    
         
            
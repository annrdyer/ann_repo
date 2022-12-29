import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Add and resize an image to the top of the app
img_fuel = Image.open("../img/fuel_efficiency.png")
st.image(img_fuel, width=700)

st.markdown("<h1 style='text-align: center; color: black;'>Fuel Efficiency</h1>", unsafe_allow_html=True)

# Import train dataset to DataFrame
#train_df = pd.read_csv("../dat/train.csv.gz", compression="gzip")
train_df = pd.read_csv('/home/annrdyer/ann_repo/assignments/week-07-intro-dl/nb/train_dataset.csv')
train_df['Country'] = np.where(train_df['Europe']==1, 'Europe', 'Other')
train_df['Country'] = np.where(train_df['USA']==1, 'USA', train_df['Country'])
train_df['Country'] = np.where(train_df['Japan']==1, 'Japan', train_df['Country'])

model_results_df = pd.read_csv('/home/annrdyer/ann_repo/assignments/week-07-intro-dl/nb/results.csv')

# Create sidebar for user selection
with st.sidebar:
    # Add FB logo
    st.image("https://user-images.githubusercontent.com/37101144/161836199-fdb0219d-0361-4988-bf26-48b0fad160a3.png" )    

    # Available models for selection

    # YOUR CODE GOES HERE!
    models = ['Linear Regression', "DNN", "TPOT"]

    # Add model select boxes
    model1_select = st.selectbox(
        "Choose Model 1:",
        (models)
    )
    
    # Remove selected model 1 from model list
    # App refreshes with every selection change.
    if model1_select == 'Linear Regression' : models2 = ["DNN", "TPOT"]
    elif model1_select == 'DNN' : models2 = ["Linear Regression", "TPOT"]
    elif model1_select == 'TPOT' : models2 = ["Linear Regression", "DNN"]
        
    
    model2_select = st.selectbox(
        "Choose Model 2:",
        (models2)
    )

# Create tabs for separation of tasks
tab1, tab2, tab3 = st.tabs(["🗃 Data", "🔎 Model Results", "🤓 Model Explainability"])

with tab1:    
    # Data Section Header
    st.header("Raw Data")

    # Display first 100 samples of the dateframe
    st.dataframe(train_df.head(100))

    st.header("Correlations")

    # Heatmap
    corr = train_df.corr()
    fig = px.imshow(corr)
    st.write(fig)
    
    
    import plotly.express as px
    
    fig = px.scatter(train_df, 
                     x="Displacement", 
                     y="MPG", 
                     color="Horsepower",
                     hover_data=['Weight'], facet_col="Cylinders", facet_row="Country")
    
    st.plotly_chart(fig, use_container_width=True)
    

with tab2:    
    
    # YOUR CODE GOES HERE!
    st.dataframe(model_results_df)
    
    expected = model_results_df['MPG'].tolist()
    
    if model1_select == 'Linear Regression':
        predicted1 = model_results_df['Linear Regression'].tolist()
    elif model1_select == 'DNN':  
        predicted1 = model_results_df['DNN'].tolist()
    elif model1_select == 'TPOT':  
        predicted1 = model_results_df['TPOT'].tolist()                                  
    
         
    if model2_select == 'Linear Regression':
        predicted2 = model_results_df['Linear Regression'].tolist()
    elif model2_select == 'DNN':  
        predicted2 = model_results_df['DNN'].tolist()
    elif model2_select == 'TPOT':  
        predicted2 = model_results_df['TPOT'].tolist()                                 
                                     
                                            
    rmse1 = mean_squared_error(expected, predicted1, squared=False)
    rmse2 = mean_squared_error(expected, predicted2, squared=False)                                   
    
    mae1 = mean_absolute_error(expected, predicted1)
    mae2 = mean_absolute_error(expected, predicted2)
    
    rsquared1 = r2_score(expected, predicted1)
    rsquared2 = r2_score(expected, predicted2)                                   

    # Columns for side-by-side model comparison
    col1, col2 = st.columns(2)

    # Build the confusion matrix for the first model.
    with col1:
        st.header(model1_select)

        # YOUR CODE GOES HERE!
        st.write('RMSE: '+ str(rmse1))                             
        st.write('MAE: ' + str(mae1))                                 
        st.write('R Square: ' + str(rsquared1))     

    # Build confusion matrix for second model
    with col2:
        st.header(model2_select)

        # YOUR CODE GOES HERE!
        st.write('RMSE: ' + str(rmse2))                            
        st.write('MAE: ' + str(mae2))                                 
        st.write('R Square: ' + str(rsquared2))  

with tab3: 
    # YOUR CODE GOES HERE!
        # Use columns to separate visualizations for models
        # Include plots for local and global explanability!
     
    st.header(model1_select)
    
    st.header(model2_select)

    

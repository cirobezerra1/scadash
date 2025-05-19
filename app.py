from os import waitstatus_to_exitcode
from pathlib import Path as path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import functions as fc
from plotly.subplots import make_subplots
import math
import visuals as vs


##### Starting the Dashboard ####
st.set_page_config(layout="wide")

with st.sidebar:

    # Step 1: File Upload Interface
    uploaded_file = st.file_uploader("Upload a CSV file", type='csv')

    uploaded_location_file = st.file_uploader("Upload a location file", type='csv')

    if (uploaded_file is not None) and (uploaded_location_file is not None):

        # Read CSV file
        df = pd.read_csv(uploaded_file)

        location = pd.read_csv(uploaded_location_file, index_col = 0)

        # Step 2: Specifying Columns in the CSV File
        st.subheader('Specify Data Columns')
        columns = df.columns.tolist()
        IDNode = st.selectbox('Turbine_ID column', columns,key = 1)
        TS = st.selectbox('Timestamp column', columns,key = 2)
        nws = st.selectbox('Windspeed column', columns, key  = 3)
        pwr = st.selectbox('Power column', columns, key = 4)
        wdir = st.selectbox('Wind Direction column', columns, key = 5)
        temperatures = st.multiselect('Temperature columns', [c for c in columns if c not in [IDNode, TS, nws, pwr, wdir]], key = 6)
        other_columns = st.multiselect('Other columns to import',[c for c in columns if c not in ([IDNode, TS, nws, pwr, wdir] + temperatures)], key = 7)
        wind_farm_capacity = st.number_input('Wind Farm Capacity (kW)', min_value=0.0, format='%f')


if (uploaded_file is not None) and (uploaded_location_file is not None) and wind_farm_capacity > 0:

    show_tab1 = True
    show_tab2 = True
    show_tab3 = True
    show_tab4 = True

    cols = [IDNode,TS,nws,pwr,wdir] + temperatures + other_columns

    df = df[cols]

    features_list =  ['absolute_sum_of_changes', 'abs_energy', 'absolute_maximum', 'kurtosis', 'skewness', 'standard_deviation', 'mean_second_derivative_central']

    df[TS] = pd.to_datetime(df[TS])

    signals_toSelect = list(df.columns)
    signals_toSelect.remove(IDNode)
    signals_toSelect.remove(TS)

    CF,nWTGs = fc.calculate_wf_metrics(df, IDNode, pwr, nws, TS, wind_farm_capacity)

    # Centralized title
    st.markdown("<h1 style='text-align: center;'>SCADash</h1>", unsafe_allow_html=True)

    # Smaller and centralized subtitle
    st.markdown("<h2 style='text-align: center; font-size: 20px;'>An interactive wind turbine SCADA data exploration tool</h2>", unsafe_allow_html=True)

    #define tabs
    tab1,tab2,tab3,tab4= st.tabs(["Power Production and Wind Analysis Dashboard","Temperature Analysis","Timeseries","Scatter Plots"])

    # Generate Monthly Production and Wind Speed plot
    figMonthly = vs.create_combined_chart_plot(df, nws, pwr, TS)
    # Generate Pattern of Production plot
    figPoP = vs.create_PoP_plot(df, IDNode, pwr, location)
    # Generate turbine top performers plot
    figTopTurbines = vs.create_top_turbines_plot(df,IDNode,pwr)
    # Generate frequency distribution plot based on nacelle wind speed
    figPlotWindDistribution = vs.create_wind_distribution(df,nws,TS)
    #Generate power heatmap
    figPowerHeatmap = vs.create_pwr_heatmap(df,IDNode, pwr,TS)
    #Generate circular power map
    figCircPowerMap = vs.create_circular_pwr_map(df,TS,nws,wdir,pwr)

    with tab1:

        if show_tab1:

            # create three kpi columns
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            # fill in those three columns with respective metrics or KPIs
            kpi1.metric(
                label="🔢 Number of wind turbines",
                value= int(nWTGs),
            )

            kpi2.metric(
                label="Installed Capacity 🔋⚡",
                value=f" {round(wind_farm_capacity/1000,2)} MW",
            )

            kpi3.metric(
                label="Capacity Factor 📊",
                value=f" {round(100*CF,1)}%",
            )

            kpi4.metric(
                    label="Average Wind Speed 💨",
                    value=f" {round(df[nws].mean(),1)} m/s",
            )

            st.plotly_chart(figMonthly, use_container_width= True)

            column1,column2 = st.columns(2)

            with column1:

                st.plotly_chart(figTopTurbines)

            with column2:

                st.plotly_chart(figPoP)

            st.plotly_chart(figPowerHeatmap, use_container_width= True)

            column3,column4 = st.columns(2)

            with column3:

                st.plotly_chart(figPlotWindDistribution)

            with column4:

                st.plotly_chart(figCircPowerMap)


    with tab2:

        if show_tab2:

            lateralSideBar = 7

            side1,visuals1 = st.columns([1,lateralSideBar])

            with side1:
                #Select wind turbine
                turbine = st.selectbox('Select Turbine to Plot:',list(df[IDNode].unique()) + ['Wind Farm'], key = 8)
                #Select ordering features
                feature = st.selectbox('Select sorting feature:',features_list, index = 0, key = 9)

            with visuals1:
            #Generate temperature heatmap
                figTempHeatmap = vs.temperature_heatmap(df,turbine,TS,IDNode,temperatures,feature)

                st.plotly_chart(figTempHeatmap, use_container_width=True)

            side2,visuals2 = st.columns([1,lateralSideBar])

            with side2:
                #Select wind turbine
                temperature = st.selectbox('Select Temperature:',temperatures, key = 10)

            with visuals2:
            #Generate temperature box plot

                figTempBox = vs.temperature_boxplot(df,IDNode,temperature)

                st.plotly_chart(figTempBox, use_container_width=True)

    with tab3:

        if show_tab3:

            sb1,  sb2, sb3, empty = st.columns([1,1,1,9])

            functions = ['count','mean','max','min','sum']

            with sb1:
                signal_toPlot = st.selectbox('Choose Signal:', signals_toSelect, key = 11)
            with sb2:
                agg_function = st.selectbox('Choose function:', functions)
            with sb3:
                resolution = st.selectbox('Select time resolution:',['Daily','Monthly','Yearly'], index = 1,key = 12)

            data = df[~df[[IDNode,TS]].duplicated()]
            data.index = data[TS]
            plotData = data[signal_toPlot]
            plotData = plotData.to_frame()
            plotData['year'] = plotData.index.year
            plotData['month'] = plotData.index.month
            plotData['day'] = plotData.index.day

            if resolution == 'Yearly':
                plotData['Timestamp_plot'] = pd.to_datetime(plotData['year'], format = '%Y')
            else:
                if resolution == 'Monthly':
                    plotData['Timestamp_plot'] = pd.to_datetime(plotData['year'].apply(str) + '-' +  plotData['month'].apply(lambda x: str(x).zfill(2)))
                else:
                    plotData['Timestamp_plot'] = pd.to_datetime(plotData['year'].apply(str) + '-' +  plotData['month'].apply(lambda x: str(x).zfill(2)) +  '-' + plotData['day'].apply(lambda x: str(x).zfill(2)))

            plotData.index = plotData.Timestamp_plot

            figTimeSeries = vs.create_univariate_timeseries_plot(plotData.groupby(plotData.index)[signal_toPlot].agg(agg_function),TS)

            st.plotly_chart(figTimeSeries, use_container_width=True)

            datapiv = data.pivot(index = TS,columns = IDNode, values = signal_toPlot)

            datapiv['year'] = datapiv.index.year
            datapiv['month'] = datapiv.index.month
            datapiv['day'] = datapiv.index.day

            turbinecol,emptycol = st.columns([1,11])

            with turbinecol:
                turbine = st.selectbox('Highlighted turbine:',list(df[IDNode].unique()), key = 13)

            if resolution == 'Yearly':
                datapiv['Timestamp_plot'] = pd.to_datetime(datapiv['year'], format = '%Y')
            else:
                if resolution == 'Monthly':
                    datapiv['Timestamp_plot'] = pd.to_datetime(datapiv['year'].apply(str) + '-' +  datapiv['month'].apply(lambda x: str(x).zfill(2)))
                else:
                    datapiv['Timestamp_plot'] = pd.to_datetime(datapiv['year'].apply(str) + '-' +  datapiv['month'].apply(lambda x: str(x).zfill(2)) +  '-' + datapiv['day'].apply(lambda x: str(x).zfill(2)))

            plotDf = datapiv.groupby('Timestamp_plot')[df[IDNode].unique()].agg(agg_function)

            figMultTimeSeries = vs.create_multivariate_timeseries_plot(plotDf.reset_index(), 'Timestamp_plot', signal_toPlot,turbine)

            st.plotly_chart(figMultTimeSeries, use_container_width=True)


    with tab4:

        if show_tab4:

            min_date =  df[TS].min().date()
            max_date = df[TS].max().date() + pd.to_timedelta('1 day')

            values = st.slider('Select a range of values',min_date,max_date, value = [min_date,max_date])

            col1,dummy,col2 = st.columns([1,1,3])

            with col1:

                default_index_x = signals_toSelect.index(nws)

                default_index_y = signals_toSelect.index(pwr)

                turbine = st.selectbox('Select Turbine to Plot:', df[IDNode].unique(), key = 14)

                xaxis = st.selectbox('x axis:',signals_toSelect, index = default_index_x , key = 15)

                yaxis = st.selectbox('y axis:',signals_toSelect, index = default_index_y , key = 16)

            with col2:

                figScatter = vs.plotScatter(df,turbine,xaxis,yaxis,values,TS,IDNode)

                st.plotly_chart(figScatter)

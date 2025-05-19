#def temperature_box_plot(data):
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import functions as fc
import streamlit as st

@st.cache_data
def create_multivariate_timeseries_plot_distribution(df):
    """
    Create a multivariate time series plot with lines for max, min, mean,
    upper quartile, and lower quartile, and a shaded area between max and min.

    Args:
    df (pd.DataFrame): A DataFrame with datetime index and multiple columns with time series data.

    Returns:
    go.Figure: Plotly figure object with the time series plot.
    """
    # Calculating statistics
    df_max = df.max(axis=1)
    df_min = df.min(axis=1)
    df_mean = df.mean(axis=1)
    df_upper_quartile = df.quantile(0.75, axis=1)
    df_lower_quartile = df.quantile(0.25, axis=1)

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for max, min, mean, upper quartile, and lower quartile
    fig.add_trace(go.Scatter(x=df.index, y=df_max, mode='lines', name='Max'))
    fig.add_trace(go.Scatter(x=df.index, y=df_min, mode='lines', name='Min', fill='tonexty'))  # Fill to max
    fig.add_trace(go.Scatter(x=df.index, y=df_mean, mode='lines', name='Mean'))
    fig.add_trace(go.Scatter(x=df.index, y=df_upper_quartile, mode='lines', name='Upper Quartile'))
    fig.add_trace(go.Scatter(x=df.index, y=df_lower_quartile, mode='lines', name='Lower Quartile'))

    # Update layout
    fig.update_layout(title='Distribution Plot', xaxis_title='Date', yaxis_title='Value')

    return fig

@st.cache_data
def create_multivariate_timeseries_plot(df, TS, signal, IDNode):
    # Create a subplot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Iterate over each column to create a line plot
    for column in df.columns:
        if column != TS:
            # Set line transparency for non-highlighted columns
            line_opacity = 1 if column == IDNode else 0.3

            # Add line plot for each column
            fig.add_trace(
                go.Scatter(
                    x=df[TS],
                    y=df[column],
                    name=column,
                    line=dict(width=2),
                    opacity=line_opacity,
                    mode='lines'
                ), secondary_y=False
            )

    # Shade area between min and max
    fig.add_trace(
        go.Scatter(
            x=df[TS].tolist() + df[TS].tolist()[::-1],  # x, then x reversed
            y=df.drop(columns=[TS]).max(axis=1).tolist() + df.drop(columns=[TS]).min(axis=1).tolist()[::-1],  # y1, then y2 reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Shaded Area',
            showlegend=False
        ), secondary_y=False
    )

    # Update plot layout
    fig.update_layout(
        title_text="Timeseries per Turbine",
        xaxis_title="Timestamp",
        yaxis_title= signal,
         legend=dict(
             orientation='h'
         ),
    )

    return fig

@st.cache_data
def create_univariate_timeseries_plot(series,TS):
    """
    Create a univariate time series line plot

    Args:
    series (pd.Series): A DataFrame with datetime index and time series data.

    Returns:
    go.Figure: Plotly figure object with the time series plot.
    """
      # Create a Plotly figure
    fig = go.Figure()

    # Add traces for max, min, mean, upper quartile, and lower quartile
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=series.name))

    # Update layout
    fig.update_layout(title='Wind Farm Timeseries Plot', xaxis_title=TS, yaxis_title=series.name)

    return fig

@st.cache_data
def create_PoP_plot(dataframe, IDNode, pwr, location):

    """
    IDNode is the string name of the turbine id column
    pwr is the string name of the power column
    location is a dataframe
    """

    df = dataframe.groupby(IDNode)[pwr].sum().to_frame().reset_index()

    df['Lat'] = df[IDNode].map(location.to_dict()['Lat'])

    df['Long'] = df[IDNode].map(location.to_dict()['Long'])

    df['Relativepercentage'] = df[pwr]/df[pwr].mean()

    df['tooltip'] = df.apply(lambda row: f"Turbine_ID: {row[IDNode]}\n Latitude : {row['Lat']} \n Longitude = {row['Long']} \n Relative Percentage: {round(row['Relativepercentage'],2)} \n", axis=1)

    # Create the scatter plot

    fig = go.Figure(data=go.Scatter(
        x=df['Lat'],
        y=df['Long'],
        mode='markers',
        marker=dict(
            size=20,
            line = dict(width = 3.5),
            color=df['Relativepercentage'],             # Set color of markers
            colorscale='Portland',               # Color scale to highlight extremes
            colorbar=dict(title='Relative Percentage'),  # Colorbar to interpret color scale
            symbol='y-up-open'               # Using triangle-down as an upside-down "Y"
        ),
        text= df['tooltip'],                     # Custom tooltip text
        hoverinfo='text'


    ))

    # Update layout
    fig.update_layout(
        title='Pattern of Production per turbine',
        xaxis_title='Latitude',
        yaxis_title='Longitude',
    )
    # Set the aspect ratio and figure size
    # fig.update_layout(
    #     autosize=False,
    #     width=700,
    #     height=700,
    #     yaxis=dict(
    #         scaleanchor="x",
    #         scaleratio=1,
    #     )
    # )
    # Set the aspect ratio of the axes

    fig.update_layout(
        xaxis=dict(
            constraintoward='center',
            scaleratio=1,
            scaleanchor='y'
        ),
        yaxis=dict(
            constraintoward='center'
        )
    )

    return fig

@st.cache_data
def create_top_turbines_plot(df,IDNode,pwr):

    # Group by 'Turbine_ID' and sum 'Grd_Prod_Pwr_Avg'
    grouped_data = df.groupby(IDNode)[pwr].sum().reset_index()

    # Sorting the data in descending order
    grouped_data_sorted = grouped_data.sort_values(by=pwr, ascending=True)

    # Create a horizontal bar chart
    fig = px.bar(grouped_data_sorted, y=IDNode, x=pwr, orientation='h',
                labels={pwr : 'Total Power [kW]', IDNode : 'Turbine'},
                 title='Power Production per turbine')

    return fig

@st.cache_data
def create_combined_chart_plot(df, nws, pwr, TS):
    # Grouping by month for total energy production
    total_energy_per_month = df.resample('M', on= TS)[pwr].sum()

    # Grouping by month for average wind speed
    avg_wind_speed_per_month = df.resample('M', on= TS)[nws].mean()

    # Create a subplot with secondary_y for two different y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add total energy production as a bar chart
    fig.add_trace(
        go.Bar(x=total_energy_per_month.index, y=total_energy_per_month, name='Total Energy Production', marker_color='turquoise'),
        secondary_y=False,
    )

    # Add average wind speed as a line chart
    fig.add_trace(
        go.Scatter(x=avg_wind_speed_per_month.index, y=avg_wind_speed_per_month, mode='lines+markers', name='Average Wind Speed', line=dict(color='red')),
        secondary_y=True,
    )

    # Add titles and labels
    fig.update_layout(
        title_text="Total Energy Production and Average Wind Speed per Month",
        legend=dict(
            x=0.5,
            y=-0.3,
            xanchor='center',
            yanchor='bottom',
            orientation='h'
        )
    )
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Total Wind Farm Power [kW]", secondary_y=False)
    fig.update_yaxes(title_text="Average Wind Speed [m/s]", secondary_y=True)

    return fig

@st.cache_data
def create_wind_distribution(df,nws,TS):
    # Extract the time of day and date
    df['TimeOfDay'] = df[TS].dt.time
    df['MonthDay'] = df[TS].dt.strftime('%m-%d %H:%M:%S')

    # Average wind speed across all turbines per timestamp
    avg_wind_speed_per_timestamp = df.groupby(['MonthDay', 'TimeOfDay'])[nws].mean().reset_index()

    # Average the wind speed for each time of day across all dates (years)
    typical_year_wind_speed = avg_wind_speed_per_timestamp.groupby('MonthDay')[nws].mean()

    # Create a histogram of the wind speed distribution
    fig = px.histogram(typical_year_wind_speed, x=nws, nbins=30,title='Wind Speed Frequency Distribution in a Typical Year',labels={nws: 'Wind Speed [m/s]'},histnorm='percent')

    # Update y-axis title to "Frequency[%]"
    fig.update_yaxes(title_text='Frequency [%]')

    # Show the plot
    return fig

@st.cache_data
def create_pwr_heatmap(df,IDNode, pwr,TS):

    df_pivoted = df.loc[df[[TS,IDNode]].drop_duplicates(keep = 'first').index].pivot(index = TS,columns= IDNode, values = pwr)

    fig = px.imshow(df_pivoted [df_pivoted > 0].transpose(),color_continuous_scale='Viridis',labels={IDNode:'Turbine',pwr:'Power [kW]'}, title = 'Historical Power Production per turbine')

    return fig

@st.cache_data
def create_circular_pwr_map(df,TS,nws,ndir,pwr):

    df_avg = df.groupby(TS).agg({nws: 'mean', ndir: 'mean', pwr: 'mean'}).reset_index()

    # Bin wind speed and wind direction
    wind_speed_bins = 27  # Adjust as needed
    wind_dir_bins = 36    # Adjust as needed (360 degrees / 10 degrees per bin)
    df_avg['wind_speed_bin'] = pd.cut(df_avg[nws].values, bins=wind_speed_bins, labels=range(wind_speed_bins))
    df_avg['wind_dir_bin'] = pd.cut(df_avg[ndir].values, bins=wind_dir_bins, labels=False) * (360 / wind_dir_bins)

    # Group by bins and calculate average power
    heatmap_data = df_avg.groupby(['wind_speed_bin', 'wind_dir_bin'])[[pwr,nws]].mean().reset_index()

    # Create polar bar chart to simulate heatmap
    fig = go.Figure()

    for speed_bin in range(wind_speed_bins):
        df_subset = heatmap_data[heatmap_data['wind_speed_bin'] == speed_bin].dropna()
        fig.add_trace(go.Barpolar(
            r=df_subset[nws],  # Use wind speed for the radial coordinate
            theta=df_subset['wind_dir_bin'],
            width=360 / wind_dir_bins,
            marker=dict(
                color=df_subset[pwr],
                coloraxis="coloraxis"
            ),
            opacity=1,
            hoverinfo='text',
            text=df_subset.apply(lambda row: f'Wind Speed Bin: {row["wind_speed_bin"]}<br>Wind Direction Bin: {row["wind_dir_bin"]}<br>Power: {row[pwr]:.2f}', axis=1), # Power values in the tooltip
            showlegend=False  # Hide the trace legend
        ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(showticklabels=False, ticks='', tickcolor='black', tickfont=dict(color='black')),  # Set radial tick font color to black
            angularaxis=dict(showticklabels=True, ticks='', direction='clockwise', rotation=90)
        ),
        coloraxis=dict(colorscale='Viridis'),  # You can choose any colorscale
        title="Wind Farm Power vs Wind Speed and Direction",
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'   # Transparent background
    )

    return fig

@st.cache_data
def temperature_heatmap(dataframe,wtg,TS,IDNode,temperatures,feature):

    if wtg == 'Wind Farm':
        df = pd.melt(dataframe.groupby(dataframe.index)[temperatures].mean())
    else:
        df = dataframe[dataframe[IDNode] == wtg]
        df = pd.melt(df[temperatures])

    temperatures_sorted = fc.extract_features(df, 'variable', 'value').sort_values(by=feature,ascending = False).index

    plotdata = dataframe[temperatures_sorted]

    plotdata.index = dataframe[TS]

    fig = px.imshow(plotdata.sort_index(),color_continuous_scale='Plasma',  labels=dict(x="Temperature Sensor", y="Timestamp", color="Temperature"),title = 'Historical Temperature Heatmap')

    fig.update_layout(
    height=800 # Increase the height of the plot (value is in pixels)
    )
    return fig

@st.cache_data
def temperature_boxplot(dataframe,IDNode,temperature):

    df = dataframe[[IDNode,temperature]]

    fig = px.box(df, x=temperature, y=IDNode, color= IDNode, color_discrete_sequence=px.colors.qualitative.Bold,labels={IDNode:'Turbine'})

    return fig

@st.cache_data
def plotScatter(df,turbine,xaxis,yaxis,values,TS,IDNode):

    datatoPlot = df[df[IDNode] == turbine]

    datatoPlot['Color'] = 'Selection'

    selectionIndex = datatoPlot[(datatoPlot[TS] <= pd.Timestamp(values[0])) |  (datatoPlot[TS] > pd.Timestamp(values[1]))].index

    datatoPlot.loc[selectionIndex,'Color'] = 'Background'

    figScatter = px.scatter(datatoPlot, x = xaxis , y = yaxis, color = 'Color', hover_data = [TS])

    # Apply the color map
    figScatter.update_traces(marker=dict(size=9), selector=dict(mode='markers'))
    figScatter.update_layout(width=1120,  height=840)

    return figScatter

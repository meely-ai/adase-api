import numpy as np
import plotly.graph_objects as go
import pandas as pd
from geo.docs.config import Path


def plot_paths(affinity, origin=None, destination=None, n_top=30):
    airport = pd.read_csv(Path.AIRPORTS).set_index('IATA_station_code')
    airport['station_size_cat'] = airport['station_size'].map({"global": 1, "int": 2, "nat": 3, "reg": 4})
    gps_lat_map = airport.GPS_lat.to_dict()
    gps_long_map = airport.GPS_long.to_dict()

    affinity = affinity.tail(n_top).copy()
    airports = affinity.index.get_level_values('catchment_o').tolist() \
               + affinity.index.get_level_values('catchment_d').tolist()
    df_airports = airport[np.in1d(airport.index, airports)]

    affinity['start_lat'] = affinity.index.get_level_values('catchment_o').map(gps_lat_map)
    affinity['start_lon'] = affinity.index.get_level_values('catchment_o').map(gps_long_map)
    affinity['end_lat'] = affinity.index.get_level_values('catchment_d').map(gps_lat_map)
    affinity['end_lon'] = affinity.index.get_level_values('catchment_d').map(gps_long_map)

    df_flight_paths = affinity.reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        locationmode='ISO-3',
        lon=df_airports['GPS_long'],
        lat=df_airports['GPS_lat'],
        hoverinfo='text',
        text=df_airports['city'],
        mode='markers',
        marker=dict(
            size=(4 - df_airports.station_size_cat) * 3,
            color='rgb(204, 0, 0)',
            line=dict(
                width=3,
                color='rgba(68, 68, 68, 0)'
            )
        )))

    for i in range(len(df_flight_paths)):
        trace = go.Scattergeo(
            locationmode='ISO-3',
            lon=[df_flight_paths['start_lon'][i], df_flight_paths['end_lon'][i]],
            lat=[df_flight_paths['start_lat'][i], df_flight_paths['end_lat'][i]],
            mode='lines',
            line=dict(width=df_flight_paths['flow'][i] * 5, color='rgb(204, 0, 0)'),
            opacity=min([float(df_flight_paths['flow'][i]) / float(df_flight_paths['flow'].max()) * .2, 1]),
            hovertext=f"{df_flight_paths['catchment_o'][i]}-{df_flight_paths['catchment_d'][i]}={round(df_flight_paths['flow'][i], 2)}",
            hoverinfo='text'
        )
        fig.add_trace(trace)

    fig.update_layout(
        title_text=f'{origin}-{destination}<br>Multi-dimensional airport catchment',
        showlegend=False,
        geo=dict(
            scope='world',
            projection_type='azimuthal equal area',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
        ),
    )

    fig.show()


def plot_paths_generic(plot_df, origin=None, destination=None, path_line_width=7):
    fig = go.Figure()
    for df, symbol, point_color, line_color in zip([plot_df.origin, plot_df.destination],
                                                   ['circle', 'star-square-dot'],
                                                   ['red', 'blue'],
                                                   ['red', 'blue']):
        df = df.groupby('code', as_index=False).first()  # unique
        fig.add_trace(go.Scattergeo(
            locationmode='ISO-3',
            lon=df['GPS_long'],
            lat=df['GPS_lat'],
            hoverinfo='text',
            text=df.name + ' (' + df.code.astype(str) + ')',
            mode='markers',
            marker=dict(
                symbol=symbol,
                size=df['size'] * (3 if symbol == 'circle' else 3),
                color=point_color,
                opacity=0.5,
                line=dict(
                    width=3,
                    color=line_color
                )
            )))

    for ix, row in plot_df.iterrows():
        trace = go.Scattergeo(
            locationmode='ISO-3',
            lon=[row.origin['GPS_long'], row.destination['GPS_long']],
            lat=[row.origin['GPS_lat'], row.destination['GPS_lat']],
            mode='lines+markers',
            line=dict(width=(0.01 + row.weight['']) * path_line_width, color='rgb(204, 0, 0)'),
            opacity=0.5,  # min([float(row.weight['']) / float(plot_df.weight.max()) * .4, 1]),
            hovertext=f"{row.origin['code']}-{row.destination['code']}={round(row.weight[''], 2)}",
            hoverinfo='text'
        )
        fig.add_trace(trace)

    fig.update_layout(
        title_text=f'{origin}-{destination}<br>Origin-destination flow',
        showlegend=False,
        geo=dict(
            #             scope='world',
            #             scope='north america',
            #             projection_type='azimuthal equal area',
            projection_type='natural earth',
            #             showland=True,
            #             landcolor='rgb(243, 243, 243)',
            #             countrycolor='rgb(204, 204, 204)',
            showcoastlines=True, coastlinecolor="RebeccaPurple",
            showland=True, landcolor="rgb(227, 250, 222)",
            showocean=True, oceancolor="LightBlue",
            showlakes=True, lakecolor="rgb(135, 193, 255)",
            showrivers=True, rivercolor="rgb(135, 193, 255)",
            showcountries=True,
            fitbounds="locations",
            lataxis_showgrid=True, lonaxis_showgrid=True
        ),
    )

    fig.show()


def explore_matched_cities(matched_cities, airport, cities_admin, city_airport_flow):
    """Create dataframe compatible to plot_paths_generic"""
    def make_plot_df(df):
        plot_schema = ['origin__code', 'origin__name', 'origin__GPS_lat', 'origin__GPS_long', 'origin__size',
                       'destination__code', 'destination__name', 'destination__GPS_lat', 'destination__GPS_long',
                       'destination__size', 'weight__']
        df = df[plot_schema]
        df.columns = pd.MultiIndex.from_tuples([c.split("__") for c in df.columns])
        return df.fillna(3)

    def adjust_relative_size(data_to_plot, by_column='origin__size'):
        data_to_plot[by_column] = ((data_to_plot[by_column] - data_to_plot[by_column].min()) /
                                   (data_to_plot[by_column].max() - data_to_plot[by_column].min())) * 4 + 3
        return data_to_plot

    matched_cities_geonamid = set(matched_cities.geonamid).intersection(city_airport_flow.columns)
    if len(matched_cities_geonamid) == 0:
        print("none of matched cities found in `city_airport_flow`")
        return
    matched_flow = city_airport_flow[matched_cities_geonamid].dropna(axis=0, thresh=1).stack().to_frame("weight__")

    data_to_plot = cities_admin[['asciiname', 'GPS_lat', 'GPS_long', 'population']].join(matched_flow).rename(
        columns={'asciiname': 'origin__name', 'GPS_lat': 'origin__GPS_lat', 'GPS_long': 'origin__GPS_long',
                 'population': 'origin__size'}).join(
        airport[['station_name', 'GPS_lat', 'GPS_long', 'pax']]
    ).rename(
        columns={'station_name': 'destination__name', 'GPS_lat': 'destination__GPS_lat',
                 'GPS_long': 'destination__GPS_long', 'pax': 'destination__size'}).reset_index(
    ).rename(
        columns={'geonamid': 'origin__code', 'IATA_station_code': 'destination__code'})

    return make_plot_df(adjust_relative_size(adjust_relative_size(data_to_plot), by_column='destination__size'))

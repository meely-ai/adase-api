import h3
import plotly.express as px
from shapely.geometry import Polygon, Point
import geopandas as gpd


def plot_h3_density(density, column_to_color='density'):
    density['geometry'] = density.geoh3.map(
        lambda geoh: Polygon([t[::-1] for t in h3.h3_to_geo_boundary(geoh + 8 * 'f')]))

    geojson = gpd.GeoSeries(density.geometry, density.index)

    fig = px.choropleth_mapbox(density,
                               geojson=geojson, locations=density.index, color=column_to_color,
                               #                            color_continuous_scale="spectral",
                               range_color=(density[column_to_color].min(), density[column_to_color].max()),
                               mapbox_style="carto-positron",
                               zoom=1, center={"lat": 37.0902, "lon": -95.7129},
                               opacity=0.4
                               )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      title_text="H3+catchment regional capacity availability\n")
    fig.update_traces(marker_line_width=3)

    fig.add_scattermapbox(
        lat=round(density.GPS_lat, 3),
        lon=round(density.GPS_long, 3),
        mode='markers+text',
        text=density.index + ', ' + density.city,
        marker_size=5,
        opacity=0.9,
        marker_color='rgb(235, 0, 100)'
    )

    fig.show()

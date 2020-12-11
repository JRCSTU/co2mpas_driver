from os import path as osp, chdir
import pandas as pd
from plotly import graph_objs as go
from co2mpas_driver.common import reading_n_organizing as rno

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    """
    This example illustrates the specifications of vehicles in the database.

    :return:
    """
    # Vehicle database based on the Euro Car Segment classification
    db_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'), 'db', 'EuroSegmentCar_cleaned'))
    
    # load vehicle database
    db = rno.load_db_to_dictionary(db_path)
    df = pd.DataFrame.from_dict(db, orient='index')
    df = df.reset_index(drop=True)
    df['vehicle-class'] = pd.Categorical(df['vehicle-class'])
    df['vehicle-class'] = df['vehicle-class'].cat.codes

    # fig = px.parallel_coordinates(df_veh, color="Drive-Total max power",
    #                             dimensions=['vehicle-class', 'Drive-Total max power', 'Exterior sizes-Length', 'Performance-Acceleration 0-100 km/h',
    #                                         'Performance-Top speed', 'Weights-Empty mass', 'Class-Euro standard car segments'],
    #                             color_continuous_scale=px.colors.diverging.Tealrose,
    #                             color_continuous_midpoint=2)

    fig = go.Figure(data=go.Parcoords(
            line=dict(color=df['Drive-Total max power'].astype(float),
                    colorscale=[[0, 'purple'], [0.5, 'lightseagreen'], [1, 'gold']]),
            dimensions=[
                dict(
                    label='Max power (kW)', values=df['Drive-Total max power'].astype(float)),
                dict(
                    label='Max torque (Nm)', values=df['Drive-Total max torque'].astype(float)),
                dict(
                    label='Max speed (m/s)', values=df['Performance-Top speed'].astype(float)),
                dict(
                    label='Veh mass (kg)', values=df['Weights-Empty mass'].astype(float)),
                dict(
                    label='Veh length (m)', values=df['Exterior sizes-Length'].astype(float)),
                dict(
                    label='0-100km/h time (s)', values=df['Performance-Acceleration 0-100 km/h'].astype(float)),
                dict(tickvals=list(range(9)), ticktext=['A', 'B', 'C', 'D', 'E', 'F', 'J', 'M', 'S'], label='Veh class',
                     values=df['vehicle-class'])
            ]
        )
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.show()


if __name__ == '__main__':
    simple_run()
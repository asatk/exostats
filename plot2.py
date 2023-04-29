import numpy as np
import pandas as pd
import plotly.express as px

CLASS_COLORS = ['#DDBB44','#00AA00','#DD5500','#33CCCC','#000000']
CLASS_LABELS = ['subterran', 'terran', 'superterran', 'giant','no class']

if __name__ == "__main__":
    df = pd.read_csv("current-exo-data/alfven_data.csv")
    title = "Magnetic Habitability Criterion of Goldilocks Exoplanets\nfor various Stellar Activity Levels"

    df['color'] = df.apply(lambda r: CLASS_COLORS[r.mass_class], axis=1)
    df['label'] = df.apply(lambda r: CLASS_LABELS[r.mass_class], axis=1)

    xmin = np.min(df.Ro)
    xmax = np.max(df.Ro) * 1.01
    ymin = np.min(df.orbit2alfven)
    ymax = np.max(df.orbit2alfven) * 1.01
    fig = px.scatter(data_frame=df, x='Ro', y='orbit2alfven', hover_name='pl_name', range_x=(xmin, xmax), range_y=(ymin, ymax), title=title)
    # fig = px.scatter(data_frame=df, x='Ro', y='orbit2alfven', hover_name='pl_name', text='pl_name', title=title)

    fig.update_traces( marker=dict(color=df.color), textposition='bottom center')

    fig.show() 
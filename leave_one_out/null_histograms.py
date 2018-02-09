import os

import plotly.graph_objs as go
from plotly.offline import plot
import pandas as pd

for f in os.listdir():
    if f.endswith("null_accuracies.csv"):
        name = f.split(".")[0]
        null_accuracy = pd.read_csv(f, header=None).iloc[:,0]
        fig = go.Figure(data=[go.Histogram(x=null_accuracy)],
                        layout = go.Layout(title=name))
        plot(fig, filename=name + ".html")


#plotly

import plotly.express as px

df = px.data.iris()

fig = px.bar(df, y="sepal_length", x="sepal_width", color="species")

fig.show()

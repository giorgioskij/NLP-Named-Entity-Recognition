import plotly.express as px

hist = list(range(200))
texts = [str(i) for i in hist]

fig = px.line(y=hist,
              text=texts,
              title='culo vs cane',
              labels={
                  'x': 'culo',
                  'y': 'cane'
              })

fig.update_traces(textposition='bottom right')

fig.show()
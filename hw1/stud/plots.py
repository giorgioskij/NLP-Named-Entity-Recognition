import plotly.express as px
import pickle
import pandas as pd
import torch
# import matplotlib.pyplot as plt

with open('../../model/history.pkl', 'rb') as f:
    data = pickle.load(f)

# train_loss, train_f1, eval_loss, eval_f1 = data
data = torch.tensor(data)
data = data.t()

df = pd.DataFrame(data.numpy(),
                  columns=['Train loss', 'Train F1', 'Eval loss', 'Eval F1'])

names = ['Train', 'Eval']
loss = px.line(
    df,
    x=list(range(len(data))),
    y=['Train loss', 'Eval loss'],
    # template='plotly_dark',
    title='Loss over epochs',
    labels={
        'x': 'Epochs',
        'value': 'Cross-entropy loss',
    },
)

for idx, name in enumerate(names):
    loss.data[idx].name = name

loss.update_layout(legend=dict(yanchor='top',
                               y=0.98,
                               xanchor='right',
                               x=0.99,
                               title=''),
                   title=dict(xanchor='center', x=0.5, yanchor='top', y=0.85))

f1 = px.line(
    df,
    x=list(range(len(data))),
    y=['Train F1', 'Eval F1'],
    # template='plotly_dark',
    title='F1-score over epochs',
    labels={
        'x': 'Epochs',
        'value': 'F1-score',
    },
)
for idx, name in enumerate(names):
    f1.data[idx].name = name

f1.update_layout(legend=dict(yanchor='bottom',
                             y=0.02,
                             xanchor='right',
                             x=0.99,
                             title=''),
                 title=dict(xanchor='center', x=0.5, yanchor='top', y=0.85))

loss.write_image('../../notes/img/loss.jpeg')
loss.show()
f1.write_image('../../notes/img/f1.jpeg')
f1.show()

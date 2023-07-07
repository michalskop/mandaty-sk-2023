"""Probabilities from the betting agencies for the next prime minister of Slovakia."""

import datetime
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

url_fortuna = "https://github.com/michalskop/ifortuna.cz/raw/master/daily/MSK42254.csv"
url_tipsport = "https://github.com/michalskop/tipsport.cz/raw/main/v3/daily/5259903.csv"

assets_path = "frontend/assets/"
public_path = "frontend/public/shares/"
flourish_path = "backend/data/"

election_date = "2023-09-30"

df_fortuna = pd.read_csv(url_fortuna)
df_tipsport = pd.read_csv(url_tipsport)

# Fortuna - remove "Prvý premiér s vyslovenou dôverou NRSR - "
df_fortuna.columns = df_fortuna.columns.str.replace("Prvý premiér s vyslovenou dôverou NRSR - ", "")

# correct names
df_fortuna.rename(columns={"Fico Róbert": "Fico Robert"}, inplace=True)

# Calculate averages from Fortuna and Tipsport
# average of 2 and 3 is 1 / ((1/2 + 1/3) / 2) = 2.4
dfx = pd.merge(df_fortuna, df_tipsport, on="date", how="outer", suffixes=("_fortuna", "_tipsport"))
dfx.index = dfx.date
del dfx["date"]
dfx1 = 1 / dfx
# get the averages, but only if both agencies have the same name, otherwise the average is the same as the original, taking into account the suffixes
dfx1mean = pd.DataFrame()
for col in dfx1.columns:
  if col.endswith("_fortuna"):
    c = col.replace("_fortuna", "")
    dfx1mean[c] = dfx1.loc[:, [c + "_fortuna", c + "_tipsport"]].mean(axis=1)
  elif col.endswith("_tipsport"):
    pass
  else:
    dfx1mean[col] = dfx1[col]
# final odds
df = 1 / dfx1mean

# calculate probabilities, cut-off all with odds > 1/sum(odd) (100, but adjusted for company's margin)
limit = 100
current_data = df.iloc[-1, 1:].T.to_frame().reset_index()
current_data.columns = ['name', 'odd']

current_data_1_psum = (1 / current_data['odd'].apply(lambda x: 1 / x).sum())
current_data['p_raw'] = np.where(current_data['odd'] < (current_data_1_psum * limit), 1 / current_data['odd'], 0)
current_data['p'] = current_data['p_raw'] / current_data['p_raw'].sum()
current_data['perc'] = (current_data['p'] * 100)
current_data['perc'] = current_data['perc'].apply(lambda x: round(x, 1))

# prepare output
out = current_data[current_data['perc'] >= 1]
out = out.sort_values(by='perc', ascending=False)
out = out.loc[:, ['name', 'perc']]
out['perc_floor'] = out['perc'].apply(lambda x: int(np.floor(x)))
out['perc_tens'] = ((out['perc'] - out['perc_floor']) * 10).astype(int)
tmp = out['name'].str.split(' ')
out['family_name'] = tmp.apply(lambda x: x[0])
out['other_names'] = tmp.apply(lambda x: ' '.join(x[1:]))

# save
out.to_json(assets_path + "data/nrsr/nrsr_prime_minister_current_odds.json", orient='records')

# last date
last_date = pd.DataFrame([{'date': df.iloc[-1][0]}])
last_date.to_json(assets_path + "data/nrsr/nrsr_prime_minister_current_odds_date.json", orient='records')

#
# prepare flourish + plotly charts
chart_data_raw = df.loc[:, out['name']]
chart_data = (1 / chart_data_raw / current_data['p_raw'].sum()).apply(lambda x: round(x, 3))

# flourish
flourish_data = (chart_data * 100).apply(lambda x: round(x, 1))
df.reset_index(inplace=True)
nice_dates = df['date'].apply(lambda x: str(datetime.datetime.fromisoformat(x).day) + ". " + str(datetime.datetime.fromisoformat(x).month) + ". " + datetime.datetime.fromisoformat(x).strftime("%y"))
flourish_data.insert(0, 'date', nice_dates)
flourish_data.to_csv(flourish_path + "nrsr_prime_minister_odds_history.csv", index=False)

# colors:
pm_colors = {
  'Pellegrini Peter': '#8A2C51',
  'Šimečka Michal': '#00b4db',
  'Fico Robert': '#ee3a43',
  'Kollár Boris': '#011b3a',
  'Matovič Igor': '#61A229',
  'Sulík Richard': '#b4c800'
}
# plotly
fig = go.Figure()
for name in chart_data:
  try:
    color = pm_colors[name]
  except:
    color = '#BBBBBB'
  fig.add_trace(go.Scatter(
    x=df['date'],
    y=chart_data[name],
    mode='lines',
    name=name,
    line=dict(
      width=5,
      color=color
    )
  ))

# note: did not work with locale
fig.update_xaxes(tickformat="%-d.%-m.%y")
fig.update_layout(template='plotly_white')
fig.layout.yaxis.tickformat = ',.0%'
fig.update_xaxes(range=[df['date'][0], election_date])
fig.update_layout(
    autosize=False,
    width=1200,
    height=600,
    margin=dict(
        l=50,
        r=20,
        b=50,
        t=20,
        pad=0
    ),
)
fig.update_yaxes(rangemode="tozero")
fig.write_image(assets_path + "image/nrsr_prime_minister_odds_history.svg")

fig.update_layout(
    autosize=False,
    width=400,
    height=400,
    margin=dict(
        l=50,
        r=20,
        b=50,
        t=20,
        pad=0
    ),
)
fig.write_image(assets_path + "image/nrsr_prime_minister_odds_history_small.svg")

# prepare plotly thumbnail
fig = go.Figure()
for name in chart_data.iloc[:, 0:5]:
  try:
    color = pm_colors[name]
  except:
    color = '#BBBBBB'
  fig.add_trace(go.Scatter(
    x=df['date'],
    y=chart_data[name],
    mode='lines',
    name=name,
    line=dict(
      width=5,
      color=color
    )
  ))

# note: did not work with locale
fig.update_xaxes(tickformat="%-d.%-m.%y")
fig.update_layout(template='plotly_white')
fig.layout.yaxis.tickformat = ',.0%'
# fig.update_xaxes(range=[df['date'][0], election_date])
fig.update_layout(
    title="Premiér/ka 2023",
    titlefont=dict(
      family='Ubuntu, verdana, arial, sans-serif',
      size=16,
      color='#e95420'
    ),
    autosize=False,
    width=300,
    height=135,
    showlegend=False,
    margin=dict(
        l=10,
        r=10,
        b=10,
        t=50,
        pad=0
    ),
)
fig.update_yaxes(rangemode="tozero")
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
# fig.update_xaxes(visible=False)
fig.write_image(assets_path + "image/nrsr_prime_minister_thumbnail.svg")

# prepare plotly sharing picture
fig = go.Figure()
for name in chart_data.iloc[:, 0:5]:
  try:
    color = pm_colors[name]
  except:
    color = '#BBBBBB'
  fig.add_trace(go.Scatter(
    x=df['date'],
    y=chart_data[name],
    mode='lines',
    name=name,
    line=dict(
      width=5,
      color=color
    )
  ))

# note: did not work with locale
fig.update_xaxes(tickformat="%-d.%-m.%y")
fig.update_layout(template='plotly_white')
fig.layout.yaxis.tickformat = ',.0%'
fig.update_yaxes(rangemode="tozero")
# fig.update_xaxes(range=[df['date'][0], election_date])
fig.update_layout(
    # plot=dict(
    #   bgcolor="#772953"
    # ),
    
    font=dict(
      family='Ubuntu, verdana, arial, sans-serif',
      color="#bbb",
      size=30
    ),
    title=dict(
      text="Mandáty.sk: Premiér/ka 2023",
      font=dict(
        color="#772953",
        size=45,
        # bgcolor="#772953"
      )
    ),
    autosize=False,
    width=1000,
    height=500,
    # showlegend=False,
    margin=dict(
        l=80,
        r=80,
        b=10,
        t=150,
        pad=0
    ),
    legend=dict(
      font=dict(
        size=35,
        family='Ubuntu',
        color="#888",
      ),
    ),
    # paper_bgcolor="#772953",
)
# fig.update_xaxes(visible=False)

# fig.add_annotation(x=35000, y=0.1,
#             text="Mandáty.cz",
#             showarrow=False)

d = datetime.datetime.now().isoformat()
filename = public_path + d + "_nrsr_prime_minister.png"

with open(assets_path + "data/nrsr/prime_minister_share_image.json", "w") as fout:
  dd = {
    'filename': "shares/" + d + "_nrsr_prime_minister.png"
  }
  json.dump(dd, fout)

fig.write_image(filename)

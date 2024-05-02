"""Doing nrsr again, always better."""

# from calendar import month
import csv
import datetime
import json
import math
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import distance
from scipy.stats import binom

source_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vROtGTMVyDHVVRJ-kEs5FzbOqbottRfARi76-G3LlQ8fXGOKGtAYD8VQtm3NApU6bIbXHYLVmKq4tXj/pub?gid=228721411&single=true&output=csv"
choices_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vROtGTMVyDHVVRJ-kEs5FzbOqbottRfARi76-G3LlQ8fXGOKGtAYD8VQtm3NApU6bIbXHYLVmKq4tXj/pub?gid=1295890805&single=true&output=csv"

assets_path = "frontend/assets/"
public_path = "frontend/public/shares/"
app_path = "frontend/assets/data/"
flourish_path = "backend/data/"
data_path = "backend/data/"

source = pd.read_csv(source_url)
choices = pd.read_csv(choices_url)

others = 'iné strany'

possible_coalitions = [
  ['PS', 'SaS', 'OĽaNO', 'SME Rodina', 'KDH', 'Demokrati'],
  ['PS', 'SaS', 'OĽaNO', 'SME Rodina', 'KDH', 'Demokrati', 'HLAS-SD'],
  ['PS', 'SaS', 'SME Rodina', 'KDH', 'Demokrati', 'HLAS-SD'],
  ['PS', 'SaS',  'KDH', 'HLAS-SD'],
  ['PS', 'SaS', 'OĽaNO', 'KDH', 'Demokrati', 'HLAS-SD'],
  ['PS', 'SaS', 'OĽaNO', 'SME Rodina', 'Demokrati', 'HLAS-SD'],
  ['SMER-SD', 'HLAS-SD', 'SNS'],
  ['SMER-SD', 'HLAS-SD', 'SNS', 'KDH'],
  ['SMER-SD', 'HLAS-SD', 'SNS', 'Republika'],
  ['SMER-SD', 'SNS', 'Republika'],
] 

include_limit = 0.03
include_current3_limit = 0.02
include_election_limit = 0.05

sample_n = 777  # basic sample size

runs = 1000  # number of runs

coef = 2 # coefficient * sd (other error)

election_date = '2023-09-30'

# find Unnamed: x column
cn = 0
for i in range(len(source.columns)):
  if source.columns[i].find('Unnamed:') >= 0:
    cn = i      

# filter - not used: only parties, not coalitions
source1 = source  # source[source['tags'] == 'parties']
middle_dates = source1['middle_date']

# sort by middle_date
source1 = source1.sort_values(by='middle_date')

# parties, filter out coalitions
allvaluesp = source1.iloc[:, (cn + 1):]
allvaluesp = allvaluesp.dropna(axis=1, how='all')
# selected_indeces = allvaluesp.index

# change % to values
allvalues = allvaluesp.apply(lambda x: x.str.rstrip('%').astype('float') / 100.0, axis=0)

# remove others
allvalues = allvalues.drop(others, axis=1)

# join PS and PS-SPOLU-OD into PS:
allvalues['PS'] = allvalues.loc[:, ['PS', 'PS-SPOLU-OD']].mean(axis=1)
del allvalues['PS-SPOLU-OD']

# find max values
maxvalues = allvalues.max(axis=0)
maxvalueselection = allvalues.iloc[0, :]
if len(allvalues) >= 3:
  maxvalues3 = allvalues.iloc[-3:-1, :].max(axis=0)
selected_parties = maxvalues[((maxvalues > include_limit) & (maxvalues3 > include_current3_limit)) | (maxvalueselection >= include_election_limit)].index
selected_values = allvalues[selected_parties]

# correlations
corr = allvalues.loc[:, selected_parties].corr()
corr = corr.fillna(0)

# mu, sigma
# dates distance matrix
d = source1.loc[:, ['middle_date']].apply(lambda x: pd.to_datetime(x))
dd = pd.DataFrame((d - d.min()).loc[:, 'middle_date'].dt.days)
distances = pd.DataFrame(distance.cdist(dd, dd), index=d.loc[:, 'middle_date'], columns=d.loc[:, 'middle_date'])
# distances.drop_duplicates(inplace=True)
# distances.T.drop_duplicates(inplace=True)
# weight matrix
# w = pow(distances, 1/30) * 30
w = 1 / np.exp(distances / 30)
ws = w.sum(axis=1)
# weighted values - mu
# replacing missing values with 0
mu1 = pd.DataFrame(np.matmul(np.matrix(w), np.matrix(selected_values.fillna(0)))).divide(np.asarray(ws), axis=0)
mu1.index = w.index
mu1.columns = selected_parties
selected_values.index = mu1.index
V_zero = selected_values.fillna(0)
# V_zero.index = selected_values.index
row_sums = w.dot(V_zero.notna().values.astype(int))
row_sums.columns = V_zero.columns
Z = np.matmul(np.matrix(w), np.matrix(V_zero))
Z.columns = V_zero.columns
mu = Z / row_sums
mu.columns = selected_values.columns
mu.index = selected_values.index
mu = mu.where(~selected_values.isna(), np.nan)
mu.index = selected_values.index

# replace first value with election value
mu.iloc[0, :] = selected_values.iloc[0, :]

mun = mu1 * sample_n
mun.columns = selected_parties
# sigma
sigman = np.sqrt(mu1 * (1 - mu1) * sample_n)

# Hagenbach-Bischoff
def hagenbach_bischoff(sample):
  """Hagenbach-Bischoff."""
  # NOTE: may be incorect!!!
  nmps = 150
  sample = sample.copy()
  sample = sample.merge(choices, left_on="party", right_on="id", how="left")
  sample.loc[:, 'value'] = sample.apply(lambda row: row['value'] if row['value'] >= row['needs'] else 0, axis=1)
  sample['value'] = sample['value'] / sample_n
  sample = sample.merge(choices, left_on="party", right_on="id", how="left")
  sample = sample.sort_values(['value'], ascending=[False])
  sample['cumulative'] = sample['value'].cumsum()
  sample['cumulative'] = sample['cumulative'] / sample['cumulative'].max()
  sample['seats'] = sample['cumulative'].apply(lambda x: math.floor(x * nmps))
  sample['seats'] = sample['seats'].diff().fillna(sample['seats'])
  sample['seats'] = sample['seats'].astype('int')
  sample['seats'] = sample['seats'].clip(lower=0)
  return sample.loc[:, ['party', 'seats']].sort_values(['seats'], ascending=[False])


# estimates
# last row
if len(mun.loc[mun.index.max()].shape) == 1:
  m = mun.loc[mun.index.max()]
else: # if there are more polls on the same day
  m = mun.loc[mun.index.max()].iloc[-1]
diagonal = np.diag(sigman.iloc[0]) * coef
cov = np.matmul(np.matmul(diagonal, corr), diagonal)
samples = pd.DataFrame(np.random.multivariate_normal(m, cov, runs) / sample_n)
samples.columns = selected_parties

estimates = pd.DataFrame(columns=selected_parties)
for i in range(runs):
  sample = samples.iloc[i].reset_index()
  sample.columns = ['party', 'value']
  hb = hagenbach_bischoff(sample)
  hb.index = hb.loc[:, 'party']
  estimates = pd.concat([estimates, pd.DataFrame(hb.loc[:, 'seats']).T], ignore_index=True)

estimates = estimates.fillna(0)

# coallitions
coalition_probabilities = []
for pc in possible_coalitions:
  item = {
    'name': (' + ').join(pc),
    'probability': round(sum(estimates.loc[:, pc].sum(axis=1) > 75) / runs * 100, 1)
  }
  coalition_probabilities.append(item)

with open(data_path + "nrsr_coalitions.csv", "w") as fout:
  writer = csv.DictWriter(fout, fieldnames=['name', 'probability'])
  writer.writeheader()
  writer.writerows(coalition_probabilities)
  
# seats - best estimate
ms = m.reset_index().rename(columns={'index': 'party', mun.index.max(): 'value'})
ms.loc[:, 'value'] = ms['value'] / sample_n
seats = hagenbach_bischoff(ms).fillna(0)
seats.index = seats.loc[:, 'party']

# stats
stats = pd.DataFrame(index=selected_parties)
stats['party'] = selected_parties
stats['median'] = estimates.median(0)
stats['lo'] = estimates.quantile(q=0.05, interpolation='nearest')
stats['hi'] = estimates.quantile(q=0.95, interpolation='nearest')
stats['in'] = (estimates > 0).sum() / runs

stats = stats.merge(choices, left_on="party", right_on="id", how="left")
stats = stats.merge(seats.loc[:, 'seats'], on="party")

stats['difference'] = stats['seats'] - stats['mps'].replace(np.nan, 0)

stats = stats.sort_values(['seats', 'hi', 'lo'], ascending=[False, False, False])

stats = stats.fillna(0)

stats.to_json(assets_path + "data/nrsr/stats.json", force_ascii=False, orient='records')

with open(assets_path + "data/nrsr/current_seats.json", "w") as fout:
  rich = {
    "data": stats[stats['hi'] > 0].to_dict(orient='records'),
    "date": d.max()['middle_date'].isoformat()[0:10]
  }
  json.dump(rich, fout, ensure_ascii=False)


#
# prepare flourish + plotly charts
last_row = mu.iloc[-1, :]
sort_indexes = last_row.sort_values(ascending=False, na_position='last').index
mu = mu.loc[:, sort_indexes]
mu.sort_index(inplace=True)
allvalues.index = middle_dates
allvalues.sort_index(inplace=True)
middle_dates = allvalues.index.tolist()

# lo + hi: 
lo = binom.ppf(0.05, sample_n, mu) / sample_n
hi = binom.ppf(0.95, sample_n, mu) / sample_n
lo = (coef * lo - mu).clip(lower=0)  # lo + (lo - mu)
hi = coef * hi - mu  # hi + (hi - mu)

# interactive Plotly chart data
dataPlotly = {
  'mu': {},
  'lo': {},
  'hi': {},
  'allvalues': {},
  'middle_dates': middle_dates,
  'choices': choices.fillna(0).to_dict(orient='records'),
  'election_date': election_date
}

for name in mu:
  dataPlotly['mu'][name] = mu[name].replace(np.nan, None).tolist()
  dataPlotly['lo'][name] = lo[name].replace(np.nan, None).tolist()
  dataPlotly['hi'][name] = hi[name].replace(np.nan, None).tolist()
  dataPlotly['allvalues'][name] = allvalues[name].replace(np.nan, None).tolist()

with open(assets_path + "data/nrsr/data.json", "w") as fout:
  json.dump(dataPlotly, fout, ensure_ascii=False)

def _html2rgba(html, a):
  html = html.strip().strip('#')
  return 'rgba({}, {}, {}, {})'.format(int(html[0:2], 16), int(html[2:4], 16), int(html[4:6], 16), a)
  # return 'rgba(' + ','.join(map(str, a)) + ')'

# plotly
# main chart
fig = go.Figure()
x = middle_dates
# sort by last values
last_row = mu.iloc[-1, :]
sort_indexes = last_row.sort_values(ascending=False, na_position='last').index
mu = mu.loc[:, sort_indexes]

x_rev = mu.index[::-1].to_list()
for name in mu:
  try:
    color = choices[choices['id'] == name].iloc[0]['color']
  except:
    color = '#BBBBBB'
  hi_same = hi[name].to_list()
  lo_rev = lo[name][::-1].to_list()
  fig.add_trace(go.Scatter(
    x=x+x_rev,
    y=hi_same+lo_rev,
    fill='toself',
    fillcolor=_html2rgba(color, 0.05),
    line_color='rgba(255, 255, 255, 0)',
    line_shape='spline',
    showlegend=False,
    name=name
  ))
for name in mu:
  try:
    color = choices[choices['id'] == name].iloc[0]['color']
  except:
    color = '#BBBBBB'
  fig.add_trace(go.Scatter(
    x=x,
    y=mu[name],
    mode='lines',
    connectgaps=True,
    line_shape='spline',
    name=name + ": " + str(round(mu.fillna(0)[name][len(mu) - 1] * 100)) + '%',
    line=dict(
      width=7,
      color=color
    )
  ))
  # fig.add_trace(go.Scatter(
  #   x=x,
  #   y=allvalues[name],
  #   mode='markers',
  #   name=name,
  #   showlegend=False,
  #   marker_color=_html2rgba(color, 0.25),
  #   marker_size=15
  # ))

# note: did not work with locale
fig.update_xaxes(tickformat="%-d.%-m.%y")
fig.update_layout(template='plotly_white')
fig.layout.yaxis.tickformat = ',.0%'
fig.update_xaxes(range=[mu.index[0], election_date])
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
fig.write_image(assets_path + "image/nrsr_polls_history.svg")

# small main chart - continuing from above
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
fig.write_image(assets_path + "image/nrsr_polls_history_small.svg")

# thumbnail
fig = go.Figure()
for name in mu.iloc[:, 0:5]:
  try:
    color = choices[choices['id'] == name].iloc[0]['color']
  except:
    color = '#BBBBBB'
  fig.add_trace(go.Scatter(
      x=x,
      y=mu[name],
      mode='lines',
      connectgaps=True,
      line_shape='spline',
      name=name + ": " + str(round(mu[name][len(mu) - 1] * 100)) + '%',
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
  title="NRSR 2020-2023",
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
fig.write_image(assets_path + "image/nrsr_thumbnail.svg")

# sharing picture
fig = go.Figure()
for name in mu.iloc[:, 0:5]:
  try:
    color = choices[choices['id'] == name].iloc[0]['color']
  except:
    color = '#BBBBBB'
  fig.add_trace(go.Scatter(
      x=x,
      y=mu[name],
      mode='lines',
      connectgaps=True,
      line_shape='spline',
      name=name + ": " + str(round(mu[name][len(mu) - 1] * 100)) + '%',
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

fig.update_layout(
    font=dict(
      family='Ubuntu, verdana, arial, sans-serif',
      color="#bbb",
      size=30
    ),
    title=dict(
      text="Mandáty.sk: Národná rada 2020-2023",
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
        family='Ubuntu, verdana, arial, sans-serif',
        color="#888",
      ),
    ),
    # paper_bgcolor="#772953",
)

d = datetime.datetime.now().isoformat()
filename = public_path + d + "_nrsr.png"

with open(assets_path + "data/nrsr/nrsr_share_image.json", "w") as fout:
  dd = {
    'filename': "shares/" + d + "_nrsr.png"
  }
  json.dump(dd, fout)

fig.write_image(filename)


# FLOURISH
# 
# First chart - current overview
# https://public.flourish.studio/visualisation/13855551/
chartdata = pd.DataFrame(round(mu.iloc[-1] * 100).fillna(0).astype(int))
chartdata = chartdata.sort_values(by=[chartdata.columns[0]], ascending=False).T
chartdata.insert(0, 'Region name', 'SR')
chartdata.reset_index(drop=True).to_csv(flourish_path + "nrsr_polls_current_overview.csv", index=False)

# Full chart
# https://public.flourish.studio/visualisation/13854967/
# averages
pollsters = source1['pollster:id'].unique()

mus = mu.unstack().reset_index()
mus.columns = ['Priemer', 'Dátum', 'Hodnota']
mus['Strana'] = mus['Priemer']

muses = []
for pollster in pollsters:
  item = mus.copy()
  item['Agentúra'] = pollster
  muses.append(item)

fulldata = pd.concat(muses).sort_values(by=['Dátum', 'Strana']).reset_index(drop=True).loc[:, ['Agentúra', 'Dátum', 'Hodnota', 'Priemer', 'Strana']]
fulldata['size'] = 1
fulldata = fulldata.astype({'Dátum': str})

# measured values
source1.loc[:, ['pollster:id', 'middle_date']]
vals = allvalues.copy()
vals.index = source1.loc[:, ['pollster:id', 'middle_date']]
valss = vals.unstack().reset_index()
valss.columns = ['Strana', 'level_1', 'Hodnota']
valss = valss.join(pd.DataFrame(valss['level_1'].values.tolist()))
valss.columns = ['Strana', 'level_1', 'Hodnota', 'Agentúra', 'Dátum']
valss['size'] = (valss['Agentúra'] == 'voľby') * 6 + 6
valss['Priemer'] = pd.NA
valss = valss.astype({'Dátum': str})

concats = [fulldata, valss.loc[:, fulldata.columns]]

fchart = pd.concat(concats)

# filter out small parties
fchart = fchart[fchart['Strana'].isin(selected_parties)]

fchart.to_csv(flourish_path + "nrsr_polls_fchart.csv", index=False)

# Filtered chart "5 %"
# https://public.flourish.studio/visualisation/14047649/
# averages
# Sme rodina, SaS, KDH, SNS, Aliancia a OĽaNO od januára 2023
parties_filter = ['SME Rodina', 'SaS', 'KDH', 'SNS', 'OĽaNO', 'Aliancia']
agencies_filter_out = ['voľby', 'Actly']
fchart_filtered = fchart[(fchart['Strana'].isin(parties_filter)) & (fchart['Dátum'] >= '2023-01-01') & (~fchart['Agentúra'].isin(agencies_filter_out)) ]
# save to csv
fchart_filtered.to_csv(flourish_path + "nrsr_polls_fchart_filtered_around_5_percent.csv", index=False)

# TABLES
# parties
source1 = source # source[source['tags'] == 'parties']
allvaluesp = source1.iloc[:, (cn + 1):]
allvaluesp = allvaluesp.dropna(axis=1, how='all')
# change % to values %
allvaluesperc = allvaluesp.apply(lambda x: x.str.rstrip('%').astype('float'), axis=0)

# sort by last values
allvaluesperc = allvaluesperc.sort_values(by = [allvaluesperc.index[-1], allvaluesperc.index[-2]], axis = 1, ascending = False)
# move 'others' to the end
col_others = allvaluesperc.pop(others)
allvaluesperc[others] = col_others

t = source1.loc[:, ['poll:identifier', 'pollster:id', 'middle_date']].join(allvaluesperc, how='right')

t = t.sort_values(['middle_date'], ascending=[False])

t.to_json(app_path + "nrsr/nrsr_polls_parties_table.json", orient='records')

names = pd.DataFrame([{'names': allvaluesperc.columns}])
names.to_json(app_path + "nrsr/nrsr_polls_parties_candidates.json", orient='records')

# coalitions
# source1 = source[source['tags'] == 'coalitions']
# allvaluesp = source1.iloc[:, (cn + 1):]
# allvaluesp = allvaluesp.dropna(axis=1, how='all')
# # change % to values %
# allvaluesperc = allvaluesp.apply(lambda x: x.str.rstrip('%').astype('float'), axis=0)

# allvaluesperc = allvaluesperc.sort_values(by = [allvaluesperc.index[-1], allvaluesperc.index[-2]], axis = 1, ascending = False)

# t = source1.loc[:, ['poll:identifier', 'pollster:id', 'middle_date']].join(allvaluesperc, how='right')

# t = t.sort_values(['middle_date'], ascending=[False])

# t.to_json(app_path + "nrsr/nrsr_polls_coalitions_table.json", orient='records')

# names = pd.DataFrame([{'names': allvaluesperc.columns}])
# names.to_json(app_path + "nrsr/nrsr_polls_coalitions_candidates.json", orient='records')

# RACE CHART
origin = pd.read_csv(data_path + "origin_race_chart.csv")

mx = mu.index.max()

limits = pd.date_range(mu.index.min() - datetime.timedelta(days=28),mu.index.max() + datetime.timedelta(days=28), freq='MS').tolist()
midmonths = []
i = 0
for l in limits[0:-1]:
  midmonths.append((limits[i + 1] - l) / 2  + l)
  i += 1

rows = pd.DataFrame()
mu2 = mu[~mu.index.duplicated()]
for mm in midmonths:
  greater = mu2.index[mu2.index > mm]
  smaller = mu2.index[mu2.index < mm]
  if len(greater) == 0:
    row = round(mu2.iloc[-1, :] * 1000) / 10
  elif len(smaller) == 0:
    row = round(mu2.iloc[0, :] * 1000) / 10
  else:
    d1 = mm - smaller[-1]
    d2 = greater[0] - mm
    row = round((d2 / (d1 + d2) * mu2.loc[smaller[-1].isoformat()[0:10], :] + d1 / (d1 + d2) * mu2.loc[greater[0].isoformat()[0:10], :]) * 1000) / 10
  rows = pd.concat([rows, pd.DataFrame(row).T])

rows.index = [datetime.datetime.strftime(mm, "%m/%y") for mm in midmonths]

origin.index = origin['strana']
origin.join(rows.T).to_csv(flourish_path + "nrsr_race_chart.csv", index=False, decimal=',')


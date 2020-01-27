import numpy as np
import pandas as pd
from matplotlib import cm


def convert_data_to_source_and_sink_format(data, columns=None, 
                                           value='quantity', 
                                           agg_func=np.sum, root='ALL', sink=False, 
                                           starting_level=1):
    '''Takes a pandas dataframe with the columns in "columns", and aggregates the values in "value" with "agg_func" for every
    combination of values in consecutive columns in "columns".
    
    Parameters:
    ----------
    data: pd.DataFrame
        Data to convert
    
    columns: list (default None)
        Columns to use in the sankey plot as levels. The order of occurence in the list determines the order in the sankey plot. If None, data.select_dtype('object').columns is used.
    
    Example:
    ========
    
    print(data)

        continent  |country    |region     |level5_desc|quantity
        -----------|-----------|-----------|-----------|--------
        ACCESSORIES|WOMENS BAGS|WOMENS BAGS|W TOTE     |   12
        ACCESSORIES|WOMENS BAGS|WOMENS BAGS|W TOTE     |   18
        ACCESSORIES|WOMENS BAGS|WOMENS BAGS|W BOWLING  |    3
        ACCESSORIES|SCARVES    |SCARVES    |CASHMERE   |    1
        ACCESSORIES|SCARVES    |SCARVES    |CASHMERE   |    3
        ACCESSORIES|SCARVES    |SCARVES    |CASHMERE   |    1
        ACCESSORIES|SCARVES    |SCARVES    |CASHMERE   |   12
        ACCESSORIES|SCARVES    |SCARVES    |CASHMERE   |   11
        ACCESSORIES|SCARVES    |SCARVES    |CASHMERE   |   10
        ACCESSORIES|SCARVES    |SCARVES    |CASHMERE   |    6

    ans = data_to_source_sink_format(data, 
                                     columns=['level2_desc', 'level3_desc', 'level4_desc'], 
                                     value='quantity', 
                                     agg_func=np.sum, 
                                     root='ALL', 
                                     starting_level=1)
                                     
    print(ans)

        source        |target        |value
        --------------|--------------|-----
        1. ALL        |2. ACCESSORIES| 77
        2. ACCESSORIES|3. SCARVES    | 44
        2. ACCESSORIES|3. WOMENS BAGS| 33
        3. SCARVES    |4. SCARVES    | 44
        3. WOMENS BAGS|4. WOMENS BAGS| 33
    
    '''
    
    
    if columns is None:
        columns = data.select_dtype('object').columns
        
    values_all = []
    sources_all = []
    targets_all = []

    x = data.copy()
    
    if root:
        x.insert(0, root, root)
        columns = [root] + list(columns)
        
    if sink:
        assert sink not in x.columns, "Please don't try to insert column {} as the sink when that column name is already in use".format(sink)
        x[sink] = sink
        columns = list(columns) + [sink]
        
   
    # I rename the values in the i'th col as f"{i}. {val}" to avoid ambiguity of column i and column j both contain the same value
    for i, col in enumerate(columns):
        x[col] = x[col].apply(lambda val: f'{i+starting_level}. {val}')
    
    for source_col, target_col in zip(columns, columns[1:]):     
        temp = x.groupby([source_col, target_col])[value].apply(agg_func).reset_index()

        source, target, values = temp.values.T
        
        sources_all += list(source)
        targets_all += list(target)

        values_all += list(values)
        
    ans = pd.DataFrame([sources_all, targets_all, values_all], index=['source', 'target', 'value']).T
    return ans

def score_to_colour(score, best=1, worst=0, center=None, cmap=cm.RdYlGn):
    
    if center is not None:
        assert min(worst, best) <= center <= max(worst,best), 'Can not center colour map outside the range given by worst and best'
    else:
        center = (worst + best)/2
        
    reference = [min(worst, best), center, max(worst,best)]
    target = [0, .5, 1]
    score = np.interp(score, reference, target)
    
    if worst < best:
        score = score
    else:
        score = worst-score
    
    if score==score:
        colour = tuple([round(i,4) for i in cmap(score)[:3]])
    else:
        colour = (0.78125, 0.78125, 0.78125) # A pale gray
    return colour


def score_to_colour(score, best=1, worst=0, center=None, cmap=cm.RdYlGn):
    
    if center is not None:
        assert min(worst, best) <= center <= max(worst,best), 'Can not center colour map outside the range given by worst and best'
    else:
        center = (worst + best)/2
        
    reference = np.array([worst, center, best])
    reference_sorted = sorted(reference)
    
    target = np.array([0, .5, 1])[reference.argsort()]
    score = np.interp(score, reference_sorted, target)
    
    if score==score:
        colour = tuple([round(i,4) for i in cmap(score)[:3]])
    else:
        colour = (0.78125, 0.78125, 0.78125) # A pale gray
    return colour


def add_scores_to_plot(data, scores, metric='MSLE', best=1, worst=0, center=None, cmap=cm.RdYlGn):
    '''Appends to the data (in source-sink format) columns "scores" and "score" where the values in 
    each row are the scores from generate_score and score is the specific value of the target metric.'''
    return (data
     .assign(scores = lambda x: x['target'].map(scores))
     .assign(score = lambda x: x['scores'].apply(lambda x: x.get(metric, np.nan) if isinstance(x, dict) else x))
     .assign(colour = lambda x: x['score'].apply(score_to_colour, best=best, worst=worst, center=center, cmap=cmap))
    )

# These are settings,
menu_gap = dict(
                y=1.2,
                x=0,
                buttons=[
                    dict(
                        label='Large Gap',
                        method='restyle',
                        args=['node.pad', 20]
                    ),
                    dict(
                        label='No Gap',
                        method='restyle',
                        args=['node.pad', 0]
                    ),
                ]
            )

menu_arrangement = dict(
                y=1.2,
                x=0.15,
                buttons=[
                    dict(
                        label='Snap',
                        method='restyle',
                        args=['arrangement', 'snap']
                    ),
                    dict(
                        label='Perpendicular',
                        method='restyle',
                        args=['arrangement', 'perpendicular']
                    ),
                    dict(
                        label='Freeform',
                        method='restyle',
                        args=['arrangement', 'freeform']
                    ),
                    dict(
                        label='Fixed',
                        method='restyle',
                        args=['arrangement', 'fixed']
                    )       
                ]
            )

menu_orientation = dict(
                   y=1.2,
                   x=.3,
                   buttons=[             
                       dict(
                           label='Horizontal',
                           method='restyle',
                           args=['orientation', 'h']
                       ),
                       dict(
                           label='Vertical',
                           method='restyle',
                           args=['orientation', 'v']
                       )
                   ]
            )


                

updatemenus = [menu_gap, 
               menu_arrangement, 
               menu_orientation]

def make_sankey_graph(data, 
                       figsize=(800,1500), 
                       updatemenus=updatemenus
                      ):
    
    '''To be finished: Function that takes in a dataframe data, a column to 
    sum to give the thickness of connections, and a score that assigns a 
    score between 0 and 1 to each connection to act as colour.'''
    
    labels = list(set(data['source'].values) | set(data['target'].values))
    

    trace = dict(
        type='sankey',
        node = dict(
          thickness = 20,
          line = dict(
            color = "black",
            width = 0.5
          ),
          label = labels,
          color = ["rgba(0, 255, 0, 0.5)"]*len(labels)
        ),
        link = dict(
          source = [labels.index(v) for v in data['source']],
          target = [labels.index(v) for v in data['target']],
          value = data['value'],
          color = data['colour'].apply(lambda x: 'rgba({},{},{}, 0.5)'.format(*[int(i*255) for i in x])) if 'colour' in data.columns else None,
          label = data['score'].astype(str).apply(lambda x: 'Score: ' + str(x)) if 'score' in data.columns else None,
      ))

        
    layout =  dict(title = "Model Performance by Category",
               font = dict(size = 10),
               height = figsize[0],
               width = figsize[1],
               updatemenus = updatemenus
                  )

    fig = dict(data=[trace], layout=layout)
    
    return fig

def evaluate_sankey(data, columns=['level2_desc', 'level3_desc'], value='quantity', 
                    agg_func=np.sum, metric='MSLE', starting_level=1, 
                    scores={}, best=1, worst=0, center=None, cmap=cm.RdYlGn):
    
    data2 = convert_data_to_source_and_sink_format(data, columns=columns, value=value, 
                                                   agg_func=agg_func, 
                                                   starting_level=starting_level, root='BURBERRY')
    
    data3 = add_scores_to_plot(data2, scores, metric=metric, best=best, worst=worst, center=center, cmap=cmap)
    
    fig = make_sankey_graph(data3, figsize=(800,1000))
    
    fig['layout']['title'] = fig['layout']['title'] + f" ({metric})"
        
    py.iplot(fig, validate=False)

    return fig


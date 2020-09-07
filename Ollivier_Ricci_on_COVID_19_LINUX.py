#!/usr/bin/env python
# coding: utf-8

# In[31]:

print('This code generates the Forman-Ricci curvature for the COVID-19 data provided by the John Hopkins University:')

print('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
)

print('The Forman-ricci curvature for simple graphs is also implemented and can be found at:  ')

print('https://github.com/saibalmars/GraphRicciCurvature')


print('Packages required:')
print('-pandas')
print('-numpy')
print('-networkx')
print('-cvxpy')
print('-networkit')
print('-matplotlib')
print('-seaborn')
print('-ssl')

print('In case of issues, please, contact us: danillo.dbs16@gmail.com')
print('Process started. The whole process might take a several time, depending on your computation power.')

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# In[32]:


# The following Forman-Ricci curvature for networks can be found at https://github.com/saibalmars/GraphRicciCurvature


# In[33]:


def formanCurvature(G, verbose=False):
    """
     Compute Forman-ricci curvature for all nodes and edges in G.
         Node curvature is defined as the average of all it's adjacency edge.
     :param G: A connected NetworkX graph, unweighted graph only now, edge weight will be ignored.
     :param verbose: Show detailed logs.
     :return: G: A NetworkX graph with Forman-Ricci Curvature with node and edge attribute "formanCurvature"
     """

    # Edge forman curvature
    for (v1, v2) in G.edges():
        if G.is_directed():
            v1_nbr = set(list(G.predecessors(v1)) + list(G.successors(v1)))
            v2_nbr = set(list(G.predecessors(v2)) + list(G.successors(v2)))
        else:
            v1_nbr = set(G.neighbors(v1))
            v1_nbr.remove(v2)
            v2_nbr = set(G.neighbors(v2))
            v2_nbr.remove(v1)
        face = v1_nbr & v2_nbr
        # G[v1][v2]["face"]=face
        prl_nbr = (v1_nbr | v2_nbr) - face
        # G[v1][v2]["prl_nbr"]=prl_nbr

        G[v1][v2]["formanCurvature"] = len(face) + 2 - len(prl_nbr)
        if verbose:
            print("Source: %s, target: %d, Forman-Ricci curvature = %f  " % (v1, v2, G[v1][v2]["formanCurvature"]))

    # Node forman curvature
    for n in G.nodes():
        fcsum = 0  # sum of the neighbor Forman curvature
        if G.degree(n) != 0:
            for nbr in G.neighbors(n):
                if 'formanCurvature' in G[n][nbr]:
                    fcsum += G[n][nbr]['formanCurvature']

            # assign the node Forman curvature to be the average of node's adjacency edges
            G.nodes[n]['formanCurvature'] = fcsum / G.degree(n)
        if verbose:
            print("node %d, Forman Curvature = %f" % (n, G.nodes[n]['formanCurvature']))
    if verbose:
        print("Forman curvature computation done.")

    return G


# In[34]:


import ssl
ssl._create_default_https_context = ssl._create_unverified_context
JH=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')


# In[35]:


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# In[36]:


from dateutil.parser import parse


# In[37]:


import datetime


# In[38]:


countries_JH=JH['Country/Region'].unique().tolist()
dates_JH=list(JH)[4:]
N=[]
for c in countries_JH:
    L=JH[JH['Country/Region']==c][dates_JH].sum().values.tolist()
    N.append(L)
JH_df=pd.DataFrame(np.array(N).T).rename(columns={i:countries_JH[i] for i in range(len(countries_JH))})    


# In[39]:


Jh_nc=[]
for c in countries_JH:
    L=JH_df[c].to_list()

    L1=[]
    L1.append(L[0])
    for i in range(1,len(L)-1):
        L1.append([L[i+1]-L[i] if L[i+1]-L[i]>=0 else 0][0])
    Jh_nc.append(L1)    


# In[40]:


JH_nc=pd.DataFrame(np.array(Jh_nc).T).rename(columns={i:countries_JH[i] for i in range(len(countries_JH))})   


# In[41]:


Df=JH_nc.copy()


# In[42]:


def graph(t0,t1,e):
    

    data=np.array(Df[t0-1:t1].corr().fillna(0))
    M=[[data[i][j] if data[i][j]>1.0001-e else 0 for i in range(len(data))] for j in range(len(data))]

    G=nx.from_numpy_array(np.array(M))
    for n in G.nodes():
        if G.has_edge(n,n):
            G.remove_edge(n,n)
            
    return G


# In[43]:


def e(t0,t1,step=0.01):
    e=0
    G=graph(t0,t1,e)
    n=nx.number_connected_components(graph(t0,t1,2))
    while nx.number_connected_components(G)>n:
        e+=step
        G=graph(t0,t1,e)
        #print(e)
    print('filtration process Done!')
    return e
    


# In[44]:




# In[45]:

import importlib
import time
from multiprocessing import Pool, cpu_count

import cvxpy
import cvxpy as cvx
import networkx as nx
import numpy as np


# In[101]:


EPSILON = 1e-7


# In[110]:


def ricciCurvature_singleEdge(G, source, target, alpha, length, verbose):
    """
    Ricci curvature computation process for a given single edge.
    :param G: The original graph
    :param source: The index of the source node
    :param target: The index of the target node
    :param alpha: Ricci curvature parameter
    :param length: all pair shortest paths dict
    :param verbose: print detail log
    :return: The Ricci curvature of given edge
    """

    assert source != target, "Self loop is not allowed."  # to prevent self loop

    # If the weight of edge is too small, return 0 instead.
    if length[source][target] < EPSILON:
        print("Zero Weight edge detected, return ricci Curvature as 0 instead.")
        return {(source, target): 0}

    source_nbr = list(G.predecessors(source)) if G.is_directed() else list(G.neighbors(source))
    target_nbr = list(G.successors(target)) if G.is_directed() else list(G.neighbors(target))

    # Append source and target node into weight distribution matrix x,y
    if not source_nbr:
        source_nbr.append(source)
        x = [1]
    else:
        x = [(1.0 - alpha) / len(source_nbr)] * len(source_nbr)
        source_nbr.append(source)
        x.append(alpha)

    if not target_nbr:
        target_nbr.append(target)
        y = [1]
    else:
        y = [(1.0 - alpha) / len(target_nbr)] * len(target_nbr)
        target_nbr.append(target)
        y.append(alpha)

    # construct the cost dictionary from x to y
    d = np.zeros((len(x), len(y)))

    for i, s in enumerate(source_nbr):
        for j, t in enumerate(target_nbr):
            assert t in length[s], "Target node not in list, should not happened, pair (%d, %d)" % (s, t)
            d[i][j] = length[s][t]

    x = np.array([x]).T  # the mass that source neighborhood initially owned
    y = np.array([y]).T  # the mass that target neighborhood needs to received

    t0 = time.time()
    rho = cvx.Variable((len(target_nbr), len(source_nbr)))  # the transportation plan rho

    # objective function d(x,y) * rho * x, need to do element-wise multiply here
    obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))

    # \sigma_i rho_{ij}=[1,1,...,1]
    source_sum = cvx.sum(rho, axis=0, keepdims=True)
    constrains = [rho * x == y, source_sum == np.ones((1, (len(source_nbr)))), 0 <= rho, rho <= 1]
    prob = cvx.Problem(obj, constrains)

    #m = prob.solve(solver="ECOS_BB")
    m = prob.solve(solver="ECOS")# change solver here if you want
    # solve for optimal transportation cost
    if verbose:
        print(time.time() - t0, " secs for cvxpy.",)

    result = 1 - (m / length[source][target])  # divided by the length of d(i, j)
    if verbose:
        print("#source_nbr: %d, #target_nbr: %d, Ricci curvature = %f" % (len(source_nbr), len(target_nbr), result))

    return {(source, target): result}


def ricciCurvature_singleEdge_ATD(G, source, target, alpha, length, verbose):
    """
    Ricci curvature computation process for a given single edge.
    By the uniform distribution.
    :param G: The original graph
    :param source: The index of the source node
    :param target: The index of the target node
    :param alpha: Ricci curvature parameter
    :param length: all pair shortest paths dict
    :param verbose: print detail log
    :return: The Ricci curvature of given edge
    """

    assert source != target, "Self loop is not allowed."  # to prevent self loop

    # If the weight of edge is too small, return 0 instead.
    if length[source][target] < EPSILON:
        print("Zero Weight edge detected, return ricci Curvature as 0 instead.")
        return {(source, target): 0}

    t0 = time.time()
    source_nbr = list(G.neighbors(source))
    target_nbr = list(G.neighbors(target))

    share = (1.0 - alpha) / (len(source_nbr) * len(target_nbr))
    cost_nbr = 0
    cost_self = alpha * length[source][target]

    for i, s in enumerate(source_nbr):
        for j, t in enumerate(target_nbr):
            assert t in length[s], "Target node not in list, should not happened, pair (%d, %d)" % (s, t)
            cost_nbr += length[s][t] * share

    m = cost_nbr + cost_self  # Average transportation cost

    if verbose:
        print(time.time() - t0, " secs for Average Transportation Distance.", end=' ')

    result = 1 - (m / length[source][target])  # Divided by the length of d(i, j)
    if verbose:
        print("#source_nbr: %d, #target_nbr: %d, Ricci curvature = %f" % (len(source_nbr), len(target_nbr), result))
    return {(source, target): result}


def _wrapRicci(stuff):
    if stuff[-1] == "ATD":
        stuff = stuff[:-1]
        return ricciCurvature_singleEdge_ATD(*stuff)
    elif stuff[-1] == "OTD":
        stuff = stuff[:-1]
        return ricciCurvature_singleEdge(*stuff)


def ricciCurvature(G, alpha=0.5, weight=None, proc=cpu_count(), edge_list=None, method="OTD", verbose=False):
    """
     Compute ricci curvature for all nodes and edges in G.
         Node ricci curvature is defined as the average of all it's adjacency edge.
     :param G: A connected NetworkX graph.
     :param alpha: The parameter for the discrete ricci curvature, range from 0 ~ 1.
                     It means the share of mass to leave on the original node.
                     eg. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
     :param weight: The edge weight used to compute Ricci curvature.
     :param proc: Number of processing used for parallel computing
     :param edge_list: Target edges to compute curvature
     :param method: Transportation method, OTD for Optimal transportation Distance,
                                           ATD for Average transportation Distance.
     :param verbose: Set True to output the detailed log.
     :return: G: A NetworkX graph with Ricci Curvature with edge attribute "ricciCurvature"
     """
   
    print("NetworKit not found, use NetworkX for all pair shortest path instead.")
    t0 = time.time()
    length = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    print(time.time() - t0, " sec for all pair.")

    t0 = time.time()
    # compute edge ricci curvature
    p = Pool(processes=proc)

    # if there is no assigned edges to compute, compute all edges instead
    if not edge_list:
        edge_list = G.edges()
    args = [(G, source, target, alpha, length, verbose, method) for source, target in edge_list]

    result = p.map_async(_wrapRicci, args)
    result = result.get()
    p.close()
    p.join()

    # assign edge Ricci curvature from result to graph G
    for rc in result:
        for k in list(rc.keys()):
            source, target = k
            G[source][target]['ricciCurvature'] = rc[k]

    # compute node Ricci curvature
    for n in G.nodes():
        rcsum = 0  # sum of the neighbor Ricci curvature
        if G.degree(n) != 0:
            for nbr in G.neighbors(n):
                if 'ricciCurvature' in G[n][nbr]:
                    rcsum += G[n][nbr]['ricciCurvature']

            # assign the node Ricci curvature to be the average of node's adjacency edges
            G.nodes[n]['ricciCurvature'] = rcsum / G.degree(n)
            if verbose:
                print("node %d, Ricci Curvature = %f" % (n, G.nodes[n]['ricciCurvature']))

    print(time.time() - t0, " sec for Ricci curvature computation.")
    return G

    """
     Compute ricci curvature for all nodes and edges in G.
         Node ricci curvature is defined as the average of all it's adjacency edge.
     :param G: A connected NetworkX graph.
     :param alpha: The parameter for the discrete ricci curvature, range from 0 ~ 1.
                     It means the share of mass to leave on the original node.
                     eg. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
     :param weight: The edge weight used to compute Ricci curvature.
     :param proc: Number of processing used for parallel computing
     :param edge_list: Target edges to compute curvature
     :param method: Transportation method, OTD for Optimal transportation Distance,
                                           ATD for Average transportation Distance.
     :param verbose: Set True to output the detailed log.
     :return: G: A NetworkX graph with Ricci Curvature with edge attribute "ricciCurvature"
     """
  
    print("NetworKit not found, use NetworkX for all pair shortest path instead.")
    t0 = time.time()
    length = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    print(time.time() - t0, " sec for all pair.")

    t0 = time.time()
    # compute edge ricci curvature
    p = Pool(processes=proc)

    # if there is no assigned edges to compute, compute all edges instead
    if not edge_list:
        edge_list = G.edges()
    args = [(G, source, target, alpha, length, verbose, method) for source, target in edge_list]

    result = p.map_async(_wrapRicci, args)
    result = result.get()
    p.close()
    p.join()

    # assign edge Ricci curvature from result to graph G
    for rc in result:
        for k in list(rc.keys()):
            source, target = k
            G[source][target]['ricciCurvature'] = rc[k]

    # compute node Ricci curvature
    for n in G.nodes():
        rcsum = 0  # sum of the neighbor Ricci curvature
        if G.degree(n) != 0:
            for nbr in G.neighbors(n):
                if 'ricciCurvature' in G[n][nbr]:
                    rcsum += G[n][nbr]['ricciCurvature']

            # assign the node Ricci curvature to be the average of node's adjacency edges
            G.nodes[n]['ricciCurvature'] = rcsum / G.degree(n)
            if verbose:
                print("node %d, Ricci Curvature = %f" % (n, G.nodes[n]['ricciCurvature']))

    print(time.time() - t0, " sec for Ricci curvature computation.")
    return G

Df=JH_nc.copy()

Time=7

Forman=[]
for i in range(1,len(Df)-Time+2):

    E=e(i,i+Time-1)
    G=graph(i,i+Time-1,E)
    ricciCurvature(G)
    value=np.mean(list(nx.get_edge_attributes(G,'ricciCurvature').values()))
    Forman.append(value)
    print('step '+str(i)+' done!')

#pd.DataFrame(Forman).to_csv('Ollivier_ricci_covid_19_interval_'+str(Time)+'_days_by_new_cases.csv',sep=';',index=False)
# In[24]:
# In[46]:


Time=7
epi_sum=[]
for i in range(1,len(Df)-Time+2):
    epi_sum.append(Df[i:i+Time-1].sum().mean())


# In[47]:


dates=list(JH)[4:]


# In[48]:

from datetime import datetime
dates = [datetime.strptime(i, '%m/%d/%y').strftime('%d/%b') for i in dates]


# In[49]:


moving=[dates[i]+str('-')+dates[i+Time-1] for i in range(len(dates)-Time)]


PDF=pd.DataFrame()
PDF['interval']=moving
PDF['moving_abrage']=epi_sum
PDF['Ricci']=Forman
PDF.to_csv('Ollivier_ricci_covid_19_interval_'+str(Time)+'_days_by_new_cases.csv',sep=';',index=False)
# In[50]:


ticks=[]
moving_ticks=[]
for i in range(len(moving)):
    if i%2!=0:
        moving_ticks.append(moving[i])
        ticks.append(i)


# In[51]:



#fig=plt.figure(figsize=(30,10))
#fig.text(0.1, 0.4, 'COVID worldwide vs. Ollivier-Ricci Curvature',fontsize=30, ha='center', va='center', rotation='vertical')
##plt.title('Time Window: 7 days')
#
#plt.subplot(211)
#plt.plot(epi_sum,lw=2,color='crimson',marker='.',label='Moving avarage of new cases \n (last update: '+dates[-1]+'/2020)')
#plt.xticks(ticks,['' for i in ticks])
#
#plt.yticks(size=20)
#plt.ticklabel_format(style='sci',axis='y',scilimits=(0,1))
#plt.legend(framealpha=0,fontsize=20)
#plt.tick_params(width=2,length=4)
#plt.margins(x=0)
#
#plt.subplot(212)
#plt.plot(Forman,lw=2,color='green',marker='.',label='Mean Ollivier-Ricci Curvature')
#plt.plot([0 for i in Forman],color='gray',ls='dashed')
#plt.xticks(ticks,moving_ticks)
#plt.xticks(rotation='90',size=20)
#plt.xticks(size=20)
#plt.yticks(size=20)
#plt.tick_params(width=2,length=4)
#
#plt.legend(framealpha=0,fontsize=20)
#
#plt.margins(x=0)
#
#color_index=moving_ticks.index('11/Mar-17/Mar')
#plt.gca().get_xticklabels()[color_index].set_color("red")
#import seaborn
#seaborn.despine(top=True)
#plt.savefig('Ollivier_Ricci_on_COVID_new_cases.png',dpi=300,bbox_inches='tight')


# In[23]:


Df=JH_df.copy()
#m=50
Time=7
Forman=[]
now = datetime.now()
for i in range(1,len(Df)-Time+2):
    E=e(i,i+Time-1)
    G=graph(i,i+Time-1,E)
    ricciCurvature(G)
    value=np.mean(list(nx.get_edge_attributes(G,'ricciCurvature').values()))
    Forman.append(value)
    print('step '+str(i)+' done!')
#pd.DataFrame(Forman).to_csv('Ollivier_ricci_covid_19_interval_'+str(Time)+'_days_by_cumulative_cases.csv',sep=';',index=False)
# In[24]:


epi_cum=[]
for i in range(Time-1,len(JH_df)):
    epi_cum.append(JH_df[i:i+1].sum().sum())


# In[25]:


ticks=[i for i in range(len(dates[Time-1:]))]


# In[26]:


shift_dates=dates[Time-1:]


# In[27]:

PDF=pd.DataFrame()
PDF['date']=shift_dates
PDF['comulative_cases']=epi_cum
PDF['Ricci']=Forman
PDF.to_csv('Ollivier_ricci_covid_19_interval_'+str(Time)+'_days_by_cumulative_cases.csv',sep=';',index=False)

ticks=[]
label_ticks=[]
for i in range(len(shift_dates)):
    if i%2!=0:
        ticks.append(i)
        label_ticks.append(shift_dates[i])


# In[30]:



#fig=plt.figure(figsize=(30,10))
#fig.text(0.09, 0.44, 'COVID worldwide vs. Forman-Ricci Curvature',fontsize=30, ha='center', va='center', rotation='vertical')
#
#
#plt.subplot(211)
#plt.plot(epi_cum,lw=2,color='crimson',marker='.',label='Cumulative cases \n (last update: '+dates[-1]+'/2020)')
#plt.xticks(ticks,['' for i in ticks])
#
#plt.yticks(size=20)
#plt.ticklabel_format(style='sci',axis='y',scilimits=(0,1))
#plt.legend(framealpha=0,fontsize=20)
#plt.tick_params(width=2,length=4)
#plt.margins(x=0)
#
#plt.subplot(212)
#plt.plot(Forman,lw=2,color='green',marker='.',label='Mean Forman-Ricci Curvature')
#plt.plot([0 for i in Forman],color='gray',ls='dashed')
#plt.xticks(ticks,label_ticks)
#plt.xticks(rotation='90',size=20)
#plt.xticks(size=20)
#plt.yticks(size=20)
#plt.tick_params(width=2,length=4)



#plt.legend(framealpha=0,fontsize=20)

#plt.margins(x=0)
#color_index=label_ticks.index('11/Mar')
#plt.gca().get_xticklabels()[color_index].set_color("red")
#import seaborn
#seaborn.despine(top=True)
#plt.savefig('Ollivier_Ricci_on_COVID_cumulative_cases.png',dpi=300,bbox_inches='tight')

print('Process termitaded. Figures saved with the following names: ')
print('Ricci_on_COVID_new_cases.png')
print('Ricci_on_COVID_cumulative_cases.png')
# In[ ]:





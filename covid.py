#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:42:12 2020

@author: jhusson
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def slice_data(data,search):
    search_text=search
    
    if type(search)==str:
        search=[search]

    if search!=['U.S.']:
        if len(search)!=1:
            search_text=', '.join(search[0:-1]) + ' and ' + search[-1]
        else:
            search_text=search[0]
            
        data=data[data['state'].isin(search)]
        
    dates=list(set(data['date']))
    subdata=[]
    for d in dates:
        tmp=data[data['date']==d]
        subdata.append(tuple((d,sum(tmp['cases']),sum(tmp['deaths']))))

    subdata=pd.DataFrame(subdata, columns=['date', 'cases', 'deaths'])        

    subdata=subdata.sort_values(by=['date'])
    subdata=subdata.reset_index()

    return subdata,search_text
            
def fit_covid(days,subdata,kwargs,ax):
    if kwargs['fit'][1]>len(days)-1 and kwargs['fit'][0]<len(days)-1:
        note='right window edge (d = %s) out of range' % (kwargs['fit'][1])
        kwargs['fit'][1]=-1
    elif kwargs['fit'][1]>len(days)-1 and kwargs['fit'][0]>len(days)-1:
        note='left window edge (d = %s) out of range' % (kwargs['fit'][0])
        kwargs['fit'][1]=-1
        kwargs['fit'][0]=0

    else:
        note='fit window'
    
    #fit the data with an exponential curve, over a certain window
    k=np.polyfit(days[kwargs['fit'][0]:kwargs['fit'][1]],
                 np.log(subdata['cases'][kwargs['fit'][0]:kwargs['fit'][1]]),1)
    
    #make an exponential model
    y_pred=subdata['cases'][kwargs['fit'][0]]*np.exp(k[0]*(days-days[kwargs['fit'][0]]))
    
    if kwargs['fit_choice']=='both' or kwargs['fit_choice']=='exp':
        #plot the model
        ax.plot(days,
                y_pred,'--',lw=0.75,color='k',label=r'exp. model: T$_2$ = %2.1f days' % (np.log(2)/k[0]))
 
    #fit the data with an exponential curve, over a certain window
    m=np.polyfit(days[kwargs['fit'][0]:kwargs['fit'][1]],
                 subdata['cases'][kwargs['fit'][0]:kwargs['fit'][1]],1)
    
    #make an exponential model
    y_pred1=m[1]+days*m[0]
    
    if kwargs['fit_choice']=='both' or kwargs['fit_choice']=='linear':
        #plot the model
        ax.plot(days,
                y_pred1,'-',lw=0.5,color='k',label=r'linear model: %s cases day$^{-1}$' % ('{:,}'.format(int(m[0]))))

       
    #highlight which section of data was fitted
    ax.fill_between([days[kwargs['fit'][0]],days[kwargs['fit'][1]]],
                    [1,1],
                    [np.ceil(max(subdata['cases'])*0.05+max(subdata['cases'])),np.ceil(max(subdata['cases'])*0.05+max(subdata['cases']))],
                    facecolor='#EEE8AA',zorder=0,edgecolor='none',label=note)
    
    ax.legend(loc=2)
            
def plot_covid(subdata,choice,**kwargs):
    """
    Plot time series to show the progress of the COVID-19 pandemic in the 
    United states, utilizing the public database maintained by the New York 
    Times (https://github.com/nytimes/covid-19-data).
    
    Parameters
    ----------
    subdata : Pandas dataframe, a slice of either the "State" or "County" 
    level NY Times data.
    
    choice : string, describes how the full data was subsetted (i.e., "New York City")
    
    **kwargs
    (optional arguments)
    
    yscale : string, 'linear' or 'log': sets y-axis of first sub-plot to be 
    either linear or logarithmic.  Default is 'linear'
    
    xlim : list (numeric, [a, b]): set x-axes of all three sub-plots. Default
    shows the full time series for your chosen slice.
    
    fit : list (integers, [c, d]): select a subset of days for curve fitting. 
    Integers in list refer to indices of the "day since first case" vector.
    
    fit_choice  : string, 'exp', 'linear' or 'both': choose the type of model 
    to fit the data. Default is 'both', and shows exponential and linear model fits.

    Date created: April, 2020
    Last updated: April, 2020
    Author: Jon M. Husson, jhusson@uvic.ca	
    """
    
    #make a figure
    fig=plt.figure(1, figsize=(8,8))
    fig.clf()
    
    #default is linear scale
    if kwargs is not None and 'yscale' not in kwargs:
        kwargs['yscale']='linear'
        
    #default is linear scale
    if kwargs is not None and 'fit_choice' not in kwargs:
        kwargs['fit_choice']='both'
        
    if len(subdata)!=0:
        ############### MAKE FIRST SUBPLOT
        ax=fig.add_subplot(311)
        
        #label figure with chosen jurisdiction
        ax.text(0,1,
                choice + ': ' + str(subdata['date'][0])[5:10] + ' to ' + str(subdata['date'][subdata.index[-1]])[5:10],
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='bottom')

        ax.text(1,1,
                '%s total cases' % '{:,}'.format(int(max(subdata['cases']))),
                transform=ax.transAxes,
                fontweight='bold',
                horizontalalignment='right',
                verticalalignment='bottom')
    
        #find out how many days are in the data slide
        days=list(subdata['date']-subdata['date'][0])
        days=np.array([d.days for d in days])
        
        #plot cumulative cases vs. day number
        ax.fill_between(days,
                        np.zeros(len(days)),
                        subdata['cases'],
                        facecolor='#ADD8E6',zorder=1,edgecolor='none')
        
        #stylize plot
        ax.set_ylabel('total number of cases',color='#6495ED')
        ax.tick_params(axis='y', labelcolor='#6495ED')
                       
        #set y-axis limits
        ax.set_ylim([1,np.ceil(max(subdata['cases'])*0.05+max(subdata['cases']))])

        #set x-axis limits
        
        #default is linear scale
        if kwargs is None:
            kwargs=dict()
            kwargs['xlim']=[-1,days[-1]+1]
        elif kwargs is not None and 'xlim' not in kwargs:
            kwargs['xlim']=[-1,days[-1]+1]

        ax.set_xlim(kwargs['xlim'])
        ax.grid(axis='x')

        if kwargs is not None and 'fit' in kwargs:
            #function to fit the data
            fit_covid(days,subdata,kwargs,ax)
            
        #set the y-axis scale
        if kwargs is not None and 'yscale' in kwargs:
            ax.set_yscale(kwargs['yscale'])
        
        #make y-axis labels comma-separated      
        ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

        #make second y-axis
        ax2 = ax.twinx()
        
        #plot cumulative fatalities vs. day number
        ax2.plot(days,
                subdata['deaths'],'-',lw=1, color='#B22222')
                
        #stylize plot
        ax2.set_ylabel('total number of fatalities',color='#B22222')
        ax2.tick_params(axis='y', labelcolor='#B22222')
        
        #set the y-axis scale
        if kwargs is not None and 'yscale' in kwargs:
            ax2.set_yscale(kwargs['yscale'])

        #set y-axis limits
        ax2.set_ylim([1,np.ceil(max(subdata['deaths'])*0.05+max(subdata['deaths']))])

        #make y-axis labels comma-separated      
        ax2.set_yticklabels(['{:,}'.format(int(x)) for x in ax2.get_yticks().tolist()])

        ############### MAKE SECOND SUBPLOT
        ax=fig.add_subplot(312)
                
        #plot case increase per day vs. day number
        ax.bar(np.array(days[0:-1])+0.5,
               np.diff(subdata['cases'])/np.diff(days),
               width=np.diff(days),
               align='edge',
               facecolor='#ADD8E6',
               edgecolor='none',
               zorder=0)

        #stylize plot
        ax.set_ylabel('new cases per day',color='#6495ED')
        ax.tick_params(axis='y', labelcolor='#6495ED')

        #set x-axis limits
        ax.set_xlim(kwargs['xlim'])
        ax.grid(axis='x')

        #make y-axis labels comma-separated
        ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

        #make second y-axis
        ax2 = ax.twinx()
        
        #plot fatality increase per day vs. day number
        ax2.plot(days[1:],
                np.diff(subdata['deaths'])/np.diff(days),'-',lw=1,color='#B22222')
        ax2.set_ylabel('new fatalities per day',color='#B22222')
        ax2.tick_params(axis='y', labelcolor='#B22222')
        ylim=ax2.get_ylim()
        ax2.set_ylim([0,ylim[1]])
        
        #make y-axis labels comma-separated
        ax2.set_yticklabels(['{:,}'.format(int(x)) for x in ax2.get_yticks().tolist()])

        ############### MAKE THIRD SUBPLOT
        ax=fig.add_subplot(313)
        
        #only conisder days since 100th case recorded
        idx=subdata['cases']>100
        days=days[idx]
        subdata=subdata[idx]
        
        #plot percent case increase per day vs. day number
        ax.plot(days[1:],
                (np.diff(subdata['cases'])/np.diff(days))/subdata['cases'][0:-1]*100,'-',color='#6495ED')
            
        ax.set_ylabel('percent increase per day',color='#6495ED')
        ax.tick_params(axis='y', labelcolor='#6495ED')
        
        #make y-axis labels comma-separated
        ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

        #make second y-axis
        ax2 = ax.twinx()
        
        #plot percent fatality increase per day vs. day number
        ax2.plot(days[1:],
                (np.diff(subdata['deaths'])/np.diff(days))/subdata['deaths'][0:-1]*100,'-',lw=1,color='#B22222')
        ax2.set_ylabel('percent increase per day',color='#B22222')
        ax2.tick_params(axis='y', labelcolor='#B22222')
        ax2.set_yticklabels(['{:,}'.format(int(x)) for x in ax2.get_yticks().tolist()])

        ax.set_xlabel('days since first case')
        ax.set_xlim(kwargs['xlim'])
        ax.grid(axis='x')

    #case of no data found
    else:
        ax=fig.add_subplot(111)
        ax.text(0.5,0.5,'NO DATA',color='red',
                fontsize=30,horizontalalignment='center')
        ax.axis('off')

#%%
#Download NY Times US database
#https://github.com/nytimes/covid-19-data
states=pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv')
counties=pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')

#make date strings into datetime objects
states['date']=pd.to_datetime(states['date'])
counties['date']=pd.to_datetime(counties['date'])

#sort datafiles by date
states=states.sort_values(by=['date'])
counties=counties.sort_values(by=['date'])

#make dictionary of US counties, indexed to state
county_lookup=dict()

for s in list(set(counties['state'])):
    county_lookup[s]=list(set(counties['county'][counties['state']==s]))

#%%
############# PLOT STATE LEVEL DATA

#choose a state, or a list of states
choice=['New York', 'New Jersey','Pennsylvania']
choice=['California', 'Washington']
choice='Pennsylvania'

#can also choose all of US
choice='U.S.'

subdata,choice_text=slice_data(states,choice)

plot_covid(subdata,choice_text,
           fit=[60,70],
           xlim=[40,77],
           fit_choice='both',
           yscale='linear')

#%%
############# PLOT COUNTY LEVEL DATA
#choose a state and county
state_choice='New York'
county_choice='New York City'

subdata=counties[(counties['state']==state_choice) & (counties['county']==county_choice)]
subdata=subdata.reset_index()

plot_covid(subdata,county_choice,
           fit=[25,35],
           xlim=[10,37],
           fit_choice='linear',
           yscale='linear')


#%%
#VANCOUVER ISLAND STATS (updated manually)
#--> because I live on Vancouver Island, B.C., I take a special interest in the COVID-19 in this region
#http://www.bccdc.ca/health-info/diseases-conditions/covid-19/case-counts-press-statements
#http://www.bccdc.ca/PublishingImages/COVID-19_Epi_Curve.png

#total number of cases, date
VI=[("2020-03-23",  539, 12, 44, 0),
    ("2020-03-24",  617, 13, 44, 0),
    ("2020-03-25",  659, 14, 47, 0),
    ("2020-03-26",  725, 14, 52, 0),
    ("2020-03-27",  792, 16, 57, 0),
    ("2020-03-30",  970, 19, 67, 0),
    ("2020-03-31", 1013, 24, 67, 0),
    ("2020-04-01", 1066, 25, 72, 0),
    ("2020-04-02", 1121, 31, 72, 2),
    ("2020-04-03", 1174, 35, 74, 2),
    ("2020-04-06", 1266, 39, 79, 2)]

VI_days=pd.to_datetime([v[0] for v in VI])
VI_days=list(VI_days-VI_days[0])
VI_days=[d.days for d in VI_days]
#%%

############# BC and VANCOUVER ISLAND LEVEL DATA
fig=plt.figure(2,figsize=(8,8))
fig.clf()

#make first subplot
ax=fig.add_subplot(311)

#label figure with chosen jurisdiction
#ax.set_title()

#label figure with chosen jurisdiction
ax.text(0,1,
        'British Columbia: ' + VI[0][0][5:10] + ' to ' + VI[-1][0][5:10],
        transform=ax.transAxes,
        horizontalalignment='left',
        verticalalignment='bottom')

ax.text(1,1,
        '%s total cases' % '{:,}'.format(int(max([v[1] for v in VI]))),
        transform=ax.transAxes,
        fontweight='bold',
        horizontalalignment='right',
        verticalalignment='bottom')

#plot cumulative cases vs. day number
ax.fill_between(VI_days,
                np.zeros(len(VI_days)),
                [v[1] for v in VI],
                label='province',
                facecolor='#8FBC8F',
                edgecolor='none',zorder=0)
                
ax.tick_params(axis='y', labelcolor='#6B8E23')
ax.legend(loc=2)
ax2 = ax.twinx()

        
ax2.plot(VI_days,
        [v[3] for v in VI],'-o',markersize=3,color='#6495ED',label='Vancouver Island')
        
ax2.tick_params(axis='y', labelcolor='#6495ED')


ax.set_ylabel('total number of cases')
ax.set_xlim([-1,VI_days[-1]+1])
ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

ax2.legend(loc=4)

#make second y-axis
ax = fig.add_subplot(312)


#plot case increase per day vs. day number
ax.bar(np.array(VI_days[0:-1])+0.5,
       np.diff([v[1] for v in VI])/np.diff(VI_days),
       width=np.diff(VI_days),
       align='edge',
       facecolor='#8FBC8F',
       edgecolor='none')

#ax.plot(VI_days[1:],
#        np.diff([v[1] for v in VI])/np.diff(VI_days),'-o',markersize=3,color='#6B8E23')


ax.tick_params(axis='y', labelcolor='#6B8E23')

ax2 = ax.twinx()

ax2.plot(VI_days[1:],
        np.diff([v[3] for v in VI])/np.diff(VI_days),'-o',markersize=3,color='#6495ED')

ax2.tick_params(axis='y', labelcolor='#6495ED')

        
ax.set_ylabel('new cases per day')

ax.set_xlim([-1,VI_days[-1]+1])


#make second y-axis
ax = fig.add_subplot(313)

#plot fatality increase per day vs. day number
ax.plot(VI_days[1:],
        (np.diff([v[1] for v in VI])/np.diff(VI_days))/np.array([v[1] for v in VI[0:-1]])*100,'-o',markersize=3,color='#6B8E23')
ax.tick_params(axis='y', labelcolor='#6B8E23')

        
ax2 = ax.twinx()

ax2.plot(VI_days[1:],
        (np.diff([v[3] for v in VI])/np.diff(VI_days))/np.array([v[3] for v in VI[0:-1]])*100,'-o',markersize=3,color='#6495ED')

ax2.tick_params(axis='y', labelcolor='#6495ED')

        
ax.set_ylabel('percent increase per day')

ax.set_xlabel('day number')
ax.set_xlim([-1,VI_days[-1]+1])



#%%

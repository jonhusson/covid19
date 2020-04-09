#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:42:12 2020

@author: jhusson
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def slice_global_data(cases,deaths,**kwargs):
    non_date_hdrs=['Province/State',
     'Country/Region',
     'Lat',
     'Long']

    if kwargs is not None and 'admin1' in kwargs: 
        sub_confirmed=globe_confirmed[(globe_confirmed['Country/Region']==kwargs['admin0']) & (globe_confirmed['Province/State']==kwargs['admin1'])]
        sub_deaths=globe_deaths[(globe_deaths['Country/Region']==kwargs['admin0']) & (globe_deaths['Province/State']==kwargs['admin1'])]
        search_text=kwargs['admin1']
    elif kwargs is not None and 'admin0' in kwargs: 
        sub_confirmed=globe_confirmed[(globe_confirmed['Country/Region']==kwargs['admin0'])]
        sub_deaths=globe_deaths[(globe_deaths['Country/Region']==kwargs['admin0'])]
        search_text=kwargs['admin0']
    else:
        sub_confirmed=globe_confirmed
        sub_deaths=globe_deaths
        search_text='world'
        
    hdrs=[h for h in list(sub_confirmed) if h not in non_date_hdrs]
    
    subdata=[]
    for h in hdrs:
        subdata.append(tuple((h,sum(sub_confirmed[h]),sum(sub_deaths[h]))))

    subdata=pd.DataFrame(subdata, columns=['date', 'cases', 'deaths'])        
    subdata['date']=pd.to_datetime(subdata['date'])

    subdata=subdata.sort_values(by=['date'])   
    subdata=subdata[subdata['cases']!=0]
    subdata=subdata.reset_index()

    return subdata, search_text

def slice_US_data(data,**kwargs):
    if kwargs=={}:
        search='U.S.'
    else:
        search=kwargs['choice']
        
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
    
    #bundle up the model for export
    myfits=[[k,y_pred]]
    
    if kwargs['fit_choice']=='both' or kwargs['fit_choice']=='exp':
        #plot the model
        ax.plot(days,
                y_pred,'--',lw=0.75,color='k',label=r'exp. model: T$_2$ = %2.1f days' % (np.log(2)/k[0]))
 
    #fit the data with a line, over a certain window
    m=np.polyfit(days[kwargs['fit'][0]:kwargs['fit'][1]],
                 subdata['cases'][kwargs['fit'][0]:kwargs['fit'][1]],1)
    
    #make a linear model
    y_pred1=m[1]+days*m[0]
    
    #bundle up the model for export
    myfits.append([m,y_pred1])
    
    if kwargs['fit_choice']=='both' or kwargs['fit_choice']=='linear':
        #plot the model
        ax.plot(days,
                y_pred1,'-.',lw=0.75,color='k',label=r'linear model: %s cases day$^{-1}$' % ('{:,}'.format(int(m[0]))))

       
    #highlight which section of data was fitted
    ax.fill_between([days[kwargs['fit'][0]],days[kwargs['fit'][1]]],
                    [1,1],
                    [np.ceil(max(subdata['cases'])*0.05+max(subdata['cases'])),np.ceil(max(subdata['cases'])*0.05+max(subdata['cases']))],
                    facecolor='#EEE8AA',zorder=0,edgecolor='none',label=note)
    
    ax.legend(loc=2)
    
    return myfits
            
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
    
    xlow :  numeric, set lower x-limit on all three sub-plots. Default
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
                '%s total deaths' % '{:,}'.format(int(max(subdata['deaths']))),
                transform=ax.transAxes,
                fontweight='bold',
                color='#B22222',
                horizontalalignment='right',
                verticalalignment='bottom')

        ax.text(1,1.1,
                '%s total cases' % '{:,}'.format(int(max(subdata['cases']))),
                transform=ax.transAxes,
                fontweight='bold',
                color='#6495ED',
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
            kwargs['xlow']=-1
        elif kwargs is not None and 'xlow' not in kwargs:
            kwargs['xlow']=-1

        kwargs['xlim']=[kwargs['xlow'],days[-1]+1]
        ax.set_xlim(kwargs['xlim'])
        ax.grid(axis='x')

        myfits=[]
        if kwargs is not None and 'fit' in kwargs:
            #function to fit the data
            myfits=fit_covid(days,subdata,kwargs,ax)
            
            
        #set the y-axis scale
        if kwargs is not None and 'yscale' in kwargs:
            ax.set_yscale(kwargs['yscale'])
        
        #make y-axis labels comma-separated      
        ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

        #make second y-axis
        ax2 = ax.twinx()
        
        #plot cumulative deaths vs. day number
        ax2.plot(days,
                subdata['deaths'],'-',lw=1, color='#B22222')
                
        #stylize plot
        ax2.set_ylabel('total number of deaths',color='#B22222')
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
        
        ylim=ax.get_ylim()
        ax.set_ylim([0,ylim[1]])
        
        #plot case increases predicted by models
        if len(myfits)>0 and kwargs['fit_choice']=='exp':
            ax.plot(days[1:],
                    np.diff(myfits[0][1])/np.diff(days),'--',lw=1,color='k')
        elif len(myfits)>0 and kwargs['fit_choice']=='linear':
            ax.plot(days[1:],
                    np.diff(myfits[1][1])/np.diff(days),'-.',lw=1,color='k')
        elif len(myfits)>0 and kwargs['fit_choice']=='both':
            ax.plot(days[1:],
                    np.diff(myfits[0][1])/np.diff(days),'--',lw=1,color='k')
            ax.plot(days[1:],
                    np.diff(myfits[1][1])/np.diff(days),'-.',lw=1,color='k')

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
        ax2.set_ylabel('new deaths per day',color='#B22222')
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
        
        if len(myfits)>0:
            myfits[0][1]=myfits[0][1][idx]
            myfits[1][1]=myfits[1][1][idx]
            myfits[1][1][myfits[1][1]<0]=np.nan
            
        #plot percent case increase per day vs. day number
        ax.plot(days[1:],
                (np.diff(subdata['cases'])/np.diff(days))/subdata['cases'][0:-1]*100,'-',color='#6495ED')
            
        ylim=ax.get_ylim()
        ax.set_ylim([0,ylim[1]])
        
        #plot percent case increases predicted by models
        if len(myfits)>0 and kwargs['fit_choice']=='exp':
            ax.plot(days[1:],
                    (np.diff(myfits[0][1])/np.diff(days))/myfits[0][1][0:-1]*100,'--',lw=1,color='k')
        elif len(myfits)>0 and kwargs['fit_choice']=='linear':
            ax.plot(days[1:],
                    (np.diff(myfits[1][1])/np.diff(days))/myfits[1][1][0:-1]*100,'-.',lw=1,color='k')
        elif len(myfits)>0 and kwargs['fit_choice']=='both':
            ax.plot(days[1:],
                    (np.diff(myfits[0][1])/np.diff(days))/myfits[0][1][0:-1]*100,'--',lw=1,color='k')
            ax.plot(days[1:],
                    (np.diff(myfits[1][1])/np.diff(days))/myfits[1][1][0:-1]*100,'-.',lw=1,color='k')

        ax.set_ylabel('% increase per day',color='#6495ED')
        ax.tick_params(axis='y', labelcolor='#6495ED')
        
        #make y-axis labels comma-separated
        ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

        #make second y-axis
        ax2 = ax.twinx()
        
        #plot percent fatality increase per day vs. day number
        ax2.plot(days[1:],
                (np.diff(subdata['deaths'])/np.diff(days))/subdata['deaths'][0:-1]*100,'-',lw=1,color='#B22222')
        ax2.set_ylabel('% increase per day',color='#B22222')
        ax2.tick_params(axis='y', labelcolor='#B22222')
        ylim=ax2.get_ylim()
        ax2.set_ylim([0,ylim[1]])
        
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

for s in sorted(list(set(counties['state']))):
    county_lookup[s]=list(set(counties['county'][counties['state']==s]))

#%%
#JHU global time series
globe_confirmed=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
globe_deaths=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

#make dictionary of states/provinces counties, indexed to country/region
admin1_lookup=dict()

for s in sorted(list(set(globe_confirmed['Country/Region']))):
    admin1_lookup[s]=list(set(globe_confirmed['Province/State'][globe_confirmed['Country/Region']==s]))


#%%
############# PLOT US STATE(s) DATA (New York Times database)

#choose a state, or a list of states
my_state=['New York', 'New Jersey','Pennsylvania']
my_state=['California', 'Washington']
my_state='Pennsylvania'

#if no choice is given, default is whole U.S.
subdata,choice_text=slice_US_data(states,
#                                  choice=my_state
                                  )

plot_covid(subdata,choice_text,
           fit=[60,70],
           xlow=40,
           fit_choice='both',
           yscale='linear')

#%%
############# PLOT US COUNTY DATA (New York Times database)
#choose a state and county
#e.g. one from "county_lookup['New York']"

state_choice='New York'
county_choice='New York City'

subdata=counties[(counties['state']==state_choice) & (counties['county']==county_choice)]
subdata=subdata.reset_index()

plot_covid(subdata,county_choice,
           fit=[25,35],
           xlow=10,
           fit_choice='linear',
           yscale='linear')

#%%
############# PLOT GLOBAL DATA (JHU database)

#pick a country or region
cr='Canada'

#pick a province or state (optional argument)
#e.g. one from "admin1_lookup['Canada']"
#MUST be given with admin0 argumument
ps='British Columbia'

#default with no choice given is world
subdata,search=slice_global_data(globe_confirmed,globe_deaths,
#                          admin0=cr,
#                          admin1=ps
                          )
#
plot_covid(subdata,
           search,
           fit=[55,65],
#           xlow=20,
           fit_choice='both',
           yscale='linear')










from neo.io import AxonIO
import numpy as np
from scipy.io import loadmat
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter 
from scipy import signal
from scipy.stats import circmean, circstd
from plotting_help import *
import sys, os
import scipy.signal
from bisect import bisect
import cPickle
import math
import pandas as pd
import scipy as sp

#---------------------------------------------------------------------------#

class Flight():  
    def __init__(self, fname, protocol=''):
        if fname.endswith('.abf'):
            self.basename = ''.join(fname.split('.')[:-1])
            self.fname = fname
        else:
            self.basename = fname
            self.fname = self.basename + '.abf'  #check here for fname type 
        
        self.protocol = protocol
                  
    def open_abf(self,exclude_indicies=[]):  
        abf = read_abf(self.fname)
        # added features to exclude specific time intervals
        n_indicies = np.size(abf['stim_x']) #assume all channels have the same sample #s 
        inc_indicies = np.setdiff1d(range(n_indicies),exclude_indicies);
                   
        self.xstim = np.array(abf['stim_x'])[inc_indicies]
        self.ystim = np.array(abf['stim_y'])[inc_indicies]

        self.samples = np.arange(self.xstim.size)  #this is adjusted
        self.t = self.samples/float(1000) # sampled at 10,000 hz -- encode here?
 
        lwa_v = np.array(abf['l_wba'])[inc_indicies]
        rwa_v = np.array(abf['r_wba'])[inc_indicies]
        
        #store the raw wing beat amplitude for checking for nonflight
        self.lwa_v = lwa_v
        self.rwa_v = rwa_v
        
        #now process wing signal
        self.lwa = process_wings(lwa_v)
        self.rwa = process_wings(rwa_v)
        self.lmr = self.lwa - self.rwa
        
        self.wbf = np.array(abf['wbf'])
        if 'in 10' in abf:
            self.ao = np.array(abf['in 10'])
        else:
            self.ao = np.array(abf['ao1'])
            
            

            
    def _is_flying(self, start_i, stop_i, wba_thres=0.2, flying_time_thresh=0.95):  #fix this critera
        #check that animal is flying 
        l_nonzero_samples = np.where(self.lwa_v[start_i:stop_i] > wba_thres)[0]
        r_nonzero_samples = np.where(self.rwa_v[start_i:stop_i] > wba_thres)[0]
        n_flying_samples = np.size(np.intersect1d(l_nonzero_samples,r_nonzero_samples))
        
        total_samples = stop_i-start_i
        
        is_flying = (float(n_flying_samples)/total_samples) > flying_time_thresh   
        return is_flying
        
           
#---------------------------------------------------------------------------#

class Looming_Flight(Flight):
    
    def process_fly(self,ex_i=[]):  #does this interfere with the Flight_Phys init?
        self.open_abf(ex_i)
        #self.clean_lmr_signal()
        self.parse_trial_times(False)
        self.parse_stim_type()
        
    def show_nonflight_exclusion(self,title_txt=''):
        fig = plt.figure(figsize=(17.5,4.5))
        plt.title(title_txt)
        
        #plt.plot(self.tach*2-75,color=purple)
        plt.plot(self.raw_lmr[::10],color=blue)
        plt.plot(self.lmr,color=magenta)
        plt.plot(self.ao-100,color=black)
        plt.plot(self.tr_starts,np.ones_like(self.tr_starts),'oc')
        plt.ylabel('WBA (Degrees)')
        plt.xlabel('Samples (at 10,000 hz)')
        
        blue_line = mlines.Line2D([], [], color='blue',label='Raw lmr')
        magenta_line = mlines.Line2D([], [], color='magenta',label='Interpolated lmr')
        purple_line = mlines.Line2D([], [], color='purple',label='Tachometer')
        black_line = mlines.Line2D([], [], color='black',label='AO')
        cyan_pt = mlines.Line2D([],[],marker='o',linewidth=0, color='cyan',label='Tr starts')
        
        fontP = FontProperties()
        fontP.set_size('small')
        
        plt.legend(handles=[blue_line,magenta_line,purple_line,black_line,cyan_pt], prop = fontP, \
                            bbox_to_anchor=(1.025, 1), loc=2, borderaxespad=0.)
        
        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path + title_txt + '_nonflight exclusion.png',bbox_inches='tight',dpi=100) 
               
    def clean_lmr_signal(self,title_txt='',if_plot=False):
        lmr = np.copy(self.lmr)
        cleaned_lmr = np.copy(lmr) # make a copy here
        
        d_lmr = np.diff(lmr)
        artifacts = np.where(abs(d_lmr) > 35)[0]-1
    
        if np.size(artifacts) >= 1:
            artifact_gap_start_is = np.where(np.diff(artifacts) > .1*10000)[0]+1
            artifact_gap_start_is = np.hstack((0,artifact_gap_start_is))  # add first start
    
            artifact_gap_stop_is = artifact_gap_start_is[1:]-1
            artifact_gap_stop_is = np.hstack((artifact_gap_stop_is,np.size(artifacts)-1)) # add last

            # now loop though all of these blanked periods, fill with previous/last real value
            for start_i, stop_i in zip(artifact_gap_start_is,artifact_gap_stop_is):
                fill_i = artifacts[start_i] - 1
                if fill_i < 0: 
                    fill_i = lmr_stop_i[stop_i] + 1
            
                lmr_start_i = artifacts[start_i]
                lmr_stop_i = artifacts[stop_i] + 10
        
                cleaned_lmr[lmr_start_i:lmr_stop_i] = cleaned_lmr[fill_i]
            
        if if_plot:
            fig = plt.figure()
            plt.plot(lmr,color=blue)
            
            if artifacts:
                plt.plot(artifacts,lmr[artifacts],'*c')
                plt.plot(artifacts[artifact_gap_start_is],np.ones_like(artifact_gap_start_is),'og')
                plt.plot(artifacts[artifact_gap_stop_is]+10,np.ones_like(artifact_gap_stop_is),'om')
            plt.plot(cleaned_lmr,color=purple)
            plt.title(title_txt)
            
            # also show nonflight periods here

        self.lmr = cleaned_lmr
          
    def remove_non_flight_trs(self, iti=750):
        # loop through each trial and determine whether fly was flying continuously
        # if a short nonflight bout (but not during turn window), interpolate
        #
        # delete the trials with long nonflight bouts--change n_trs, tr_starts, 
        # tr_stops, looming stim on
        
        non_flight_trs = [];
        
        for tr_i in range(self.n_trs):
            this_tr_start = self.tr_starts[tr_i] - iti
            this_tr_stop = self.tr_stops[tr_i] + iti
            
            if not self._is_flying(this_tr_start,this_tr_stop):
                non_flight_trs.append(tr_i) 
        
        #print 'nonflight trials : ' + ', '.join(str(x) for x in non_flight_trs)
        
        print 'nonflight trials : ' + str(np.size(non_flight_trs)) + '/' + str(self.n_trs)
        
        #now remove these
        self.n_nonflight_trs = np.size(non_flight_trs)
        self.n_trs = self.n_trs - np.size(non_flight_trs)
        self.tr_starts = np.delete(self.tr_starts,non_flight_trs)  #index values of starting and stopping
        self.tr_stops = np.delete(self.tr_stops,non_flight_trs)
        #self.pre_loom_stim_ons = np.delete(self.pre_loom_stim_ons,non_flight_trs)
                    
    def parse_trial_times(self, if_debug_fig=False):
        #parse the ao signal to determine trial start and stop index values
        #include checks for unusual starting aos, early trial ends, 
        #long itis, etc
        
        ao_diff = np.diff(self.ao)
        
        tr_start = self.samples[np.where(ao_diff > .5)]
        start_diff = np.diff(tr_start)
        redundant_starts = tr_start[np.where(start_diff < 1000)]
        clean_tr_starts = np.setdiff1d(tr_start,redundant_starts)+1
        
        tr_stop = self.samples[np.where(ao_diff < -.5)]
        stop_diff = np.diff(tr_stop)
        redundant_stops = tr_stop[np.where(stop_diff < 1000)] 
        #now check that the y value is > -9 
        clean_tr_stop_candidates = np.setdiff1d(tr_stop,redundant_stops)+1
        
        clean_tr_stops = clean_tr_stop_candidates[np.where(self.ao[clean_tr_stop_candidates-5] > -9)]
        
        #check that first start is before first stop
        if clean_tr_stops[0] < clean_tr_starts[0]: 
            clean_tr_stops = np.delete(clean_tr_stops,0)
         
        #last stop is after last start
        if clean_tr_starts[-1] > clean_tr_stops[-1]:
            clean_tr_starts = np.delete(clean_tr_starts,len(clean_tr_starts)-1)
         
        #should check for same # of starts and stops
        n_trs = len(clean_tr_starts)
        
        if if_debug_fig:
            figd = plt.figure()
            plt.plot(self.ao)
            plt.plot(ao_diff,color=magenta)
            y_start = np.ones(len(clean_tr_starts))
            y_stop = np.ones(len(clean_tr_stops))
            plt.plot(clean_tr_starts,y_start*7,'go')
            plt.plot(clean_tr_stops,y_stop*7,'ro')
        
        #detect when the y stim stepped
        ystim_diff = np.diff(self.ystim)
        y_step = self.samples[np.where(ystim_diff > .03)]

        ##now discriminate first stim on for a trial, not looming steps
        #pre_loom_stim = np.zeros(n_trs)
        #for tr_i in range(n_trs):
        #    earliest_t = clean_tr_starts[tr_i] - 2000  #look within a window before looming
        #    latest_t = clean_tr_starts[tr_i] - 300 
        #    win1 = np.where(y_step > earliest_t)[0]
        #    win2 = np.where(y_step < latest_t)[0]
        #    candidate_stim_starts = y_step[np.intersect1d(win1,win2)]
        #    pre_loom_stim[tr_i] = candidate_stim_starts[0]
        
        #next encode post loom stim on, iti_dur as the median
        
        self.n_trs = n_trs 
        self.tr_starts = clean_tr_starts  #index values of starting and stopping
        self.tr_stops = clean_tr_stops
        #self.pre_loom_stim_ons = pre_loom_stim
        
        #here remove all trials in which the fly is not flying. 
        self.remove_non_flight_trs()
        
    def parse_stim_type(self):
        #calculate the stimulus type
       
        if self.protocol == '1feb2015':
            stim_types_labels =['left - 22 l/v, 50 max d, 3 min contrast',\
                                'left - 22 l/v, 50 max d, 4 min contrast',\
                                'left - 22 l/v, 50 max d, 5 min contrast',\
                                'left - 22 l/v, 90 max d, 3 min contrast',\
                                'left - 22 l/v, 90 max d, 4 min contrast',\
                                'left - 22 l/v, 90 max d, 5 min contrast',\
                                'left - 44 l/v, 50 max d, 3 min contrast',\
                                'left - 44 l/v, 50 max d, 4 min contrast',\
                                'left - 44 l/v, 50 max d, 5 min contrast',\
                                'left - 44 l/v, 90 max d, 3 min contrast',\
                                'left - 44 l/v, 90 max d, 4 min contrast',\
                                'left - 44 l/v, 90 max d, 5 min contrast',\
                                'left - 88 l/v, 50 max d, 3 min contrast',\
                                'left - 88 l/v, 50 max d, 4 min contrast',\
                                'left - 88 l/v, 50 max d, 5 min contrast',\
                                'left - 88 l/v, 90 max d, 3 min contrast',\
                                'left - 88 l/v, 90 max d, 4 min contrast',\
                                'left - 88 l/v, 90 max d, 5 min contrast',\
                                
                                'center - 22 l/v, 50 max d, 3 min contrast',\
                                'center - 22 l/v, 50 max d, 4 min contrast',\
                                'center - 22 l/v, 50 max d, 5 min contrast',\
                                'center - 22 l/v, 90 max d, 3 min contrast',\
                                'center - 22 l/v, 90 max d, 4 min contrast',\
                                'center - 22 l/v, 90 max d, 5 min contrast',\
                                'center - 44 l/v, 50 max d, 3 min contrast',\
                                'center - 44 l/v, 50 max d, 4 min contrast',\
                                'center - 44 l/v, 50 max d, 5 min contrast',\
                                'center - 44 l/v, 90 max d, 3 min contrast',\
                                'center - 44 l/v, 90 max d, 4 min contrast',\
                                'center - 44 l/v, 90 max d, 5 min contrast',\
                                'center - 88 l/v, 50 max d, 3 min contrast',\
                                'center - 88 l/v, 50 max d, 4 min contrast',\
                                'center - 88 l/v, 50 max d, 5 min contrast',\
                                'center - 88 l/v, 90 max d, 3 min contrast',\
                                'center - 88 l/v, 90 max d, 4 min contrast',\
                                'center - 88 l/v, 90 max d, 5 min contrast',\
                                
                                'right - 22 l/v, 50 max d, 3 min contrast',\
                                'right - 22 l/v, 50 max d, 4 min contrast',\
                                'right - 22 l/v, 50 max d, 5 min contrast',\
                                'right - 22 l/v, 90 max d, 3 min contrast',\
                                'right - 22 l/v, 90 max d, 4 min contrast',\
                                'right - 22 l/v, 90 max d, 5 min contrast',\
                                'right - 44 l/v, 50 max d, 3 min contrast',\
                                'right - 44 l/v, 50 max d, 4 min contrast',\
                                'right - 44 l/v, 50 max d, 5 min contrast',\
                                'right - 44 l/v, 90 max d, 3 min contrast',\
                                'right - 44 l/v, 90 max d, 4 min contrast',\
                                'right - 44 l/v, 90 max d, 5 min contrast',\
                                'right - 88 l/v, 50 max d, 3 min contrast',\
                                'right - 88 l/v, 50 max d, 4 min contrast',\
                                'right - 88 l/v, 50 max d, 5 min contrast',\
                                'right - 88 l/v, 90 max d, 3 min contrast',\
                                'right - 88 l/v, 90 max d, 4 min contrast',\
                                'right - 88 l/v, 90 max d, 5 min contrast']
                                
        else:
            stim_types_labels =['left - 14 l/v, 33 max d, 2 min contrast',\
                                'left - 14 l/v, 33 max d, 3 min contrast',\
                                'left - 14 l/v, 50 max d, 2 min contrast',\
                                'left - 14 l/v, 50 max d, 3 min contrast',\
                                'left - 14 l/v, 90 max d, 2 min contrast',\
                                'left - 14 l/v, 90 max d, 3 min contrast',\
                                'left - 22 l/v, 33 max d, 2 min contrast',\
                                'left - 22 l/v, 33 max d, 3 min contrast',\
                                'left - 22 l/v, 50 max d, 2 min contrast',\
                                'left - 22 l/v, 50 max d, 3 min contrast',\
                                'left - 22 l/v, 90 max d, 2 min contrast',\
                                'left - 22 l/v, 90 max d, 3 min contrast',\
                                'left - 44 l/v, 33 max d, 2 min contrast',\
                                'left - 44 l/v, 33 max d, 3 min contrast',\
                                'left - 44 l/v, 50 max d, 2 min contrast',\
                                'left - 44 l/v, 50 max d, 3 min contrast',\
                                'left - 44 l/v, 90 max d, 2 min contrast',\
                                'left - 44 l/v, 90 max d, 3 min contrast',\
                            
                                'center - 14 l/v, 33 max d, 2 min contrast',\
                                'center - 14 l/v, 33 max d, 3 min contrast',\
                                'center - 14 l/v, 50 max d, 2 min contrast',\
                                'center - 14 l/v, 50 max d, 3 min contrast',\
                                'center - 14 l/v, 90 max d, 2 min contrast',\
                                'center - 14 l/v, 90 max d, 3 min contrast',\
                                'center - 22 l/v, 33 max d, 2 min contrast',\
                                'center - 22 l/v, 33 max d, 3 min contrast',\
                                'center - 22 l/v, 50 max d, 2 min contrast',\
                                'center - 22 l/v, 50 max d, 3 min contrast',\
                                'center - 22 l/v, 90 max d, 2 min contrast',\
                                'center - 22 l/v, 90 max d, 3 min contrast',\
                                'center - 44 l/v, 33 max d, 2 min contrast',\
                                'center - 44 l/v, 33 max d, 3 min contrast',\
                                'center - 44 l/v, 50 max d, 2 min contrast',\
                                'center - 44 l/v, 50 max d, 3 min contrast',\
                                'center - 44 l/v, 90 max d, 2 min contrast',\
                                'center - 44 l/v, 90 max d, 3 min contrast',\
                            
                                'right - 14 l/v, 33 max d, 2 min contrast',\
                                'right - 14 l/v, 33 max d, 3 min contrast',\
                                'right - 14 l/v, 50 max d, 2 min contrast',\
                                'right - 14 l/v, 50 max d, 3 min contrast',\
                                'right - 14 l/v, 90 max d, 2 min contrast',\
                                'right - 14 l/v, 90 max d, 3 min contrast',\
                                'right - 22 l/v, 33 max d, 2 min contrast',\
                                'right - 22 l/v, 33 max d, 3 min contrast',\
                                'right - 22 l/v, 50 max d, 2 min contrast',\
                                'right - 22 l/v, 50 max d, 3 min contrast',\
                                'right - 22 l/v, 90 max d, 2 min contrast',\
                                'right - 22 l/v, 90 max d, 3 min contrast',\
                                'right - 44 l/v, 33 max d, 2 min contrast',\
                                'right - 44 l/v, 33 max d, 3 min contrast',\
                                'right - 44 l/v, 50 max d, 2 min contrast',\
                                'right - 44 l/v, 50 max d, 3 min contrast',\
                                'right - 44 l/v, 90 max d, 2 min contrast',\
                                'right - 44 l/v, 90 max d, 3 min contrast']
        
        
        stim_types = -1*np.ones(self.n_trs,'int')
        
        tr_ao_codes = np.empty(self.n_trs)
        
        #first loop through to get the unique ao values
        for tr in range(self.n_trs): 
            this_start = self.tr_starts[tr]
            this_stop = self.tr_stops[tr]
            tr_ao_codes[tr] = round(2*np.mean(self.ao[(this_start+20):(this_stop-20)]),1)   
        unique_tr_ao_codes = np.unique(tr_ao_codes) 
        print 'n stim types = ' + str(np.size(unique_tr_ao_codes))
        
        for tr in range(self.n_trs): 
            tr_ao_code = tr_ao_codes[tr]         
            
            if not np.isnan(tr_ao_code):
                stim_types[tr] = int(np.where(unique_tr_ao_codes == tr_ao_code)[0])
                
        
        self.stim_types = stim_types  #change to integer, although nans are also useful
        self.stim_types_labels = stim_types_labels
           
        
    def plot_wba_size_x_position(self,title_txt='',l_div_v_list=[0,1,2],contrast_list=[0,1],\
                        wba_lim=[-60,60],if_save=True,if_x_zoom=True): 
        
        # for each l/v stim parameter and min contrast 
        # plot lmr wba for three rows of max disk size x 
        # three columns of looming direction
        
        sampling_rate = 1000 # in hertz ********* move to fly info
        s_iti = 2 * sampling_rate #********* move to fly info
        
        baseline_win = range(0,int(1*sampling_rate))  #**************** check this.
                # time relative loom start - s_iti
                # do not average out the visual onset
        
        

        contrast_txt = ['2:7 contrast', '3:7 contrast']
        l_div_v_txt = ['14 l div v','22 l div v','44 l div v']
        max_disk_size = ['33 degrees','50 degrees', '90 degrees']
        tr_types_by_cnd = np.asarray([[0,2,4],[18,20,22],[36,38,40]]) 
        
        tr_types_by_cnd = tr_types_by_cnd.transpose()
        
        
        # define saccade window relative loom max time
        l_div_v_saccade_win = np.array([[-.25,.25],[-.75,.5],[-.5,.5]])*sampling_rate
        l_div_v_saccade_win = l_div_v_saccade_win.astype(int)               
                       
        # define zoomed x axis range relative loom start - s_iti ------------- update this
        l_div_v_zoom_win = np.array([[s_iti,s_iti+(.6*sampling_rate)],  \
                                     [s_iti,s_iti+(1*sampling_rate)], \
                                     [s_iti,s_iti+(1.6*sampling_rate)]],dtype=int)
        
        #get all traces and detect saccades ______________________________________________
        all_fly_traces, all_fly_saccades = self.get_traces_by_stim('this_fly',s_iti,get_saccades=True)
                
        n_rows = 3
        n_cols = 3
        
        #now plot one figure for each looming speed ______________________________________
        for loom_speed in l_div_v_list: 
            for contrast in contrast_list: 
                fig = plt.figure(figsize=(10.7875,7.925))       #(16.5, 9))
                gs = gridspec.GridSpec(6,n_cols,height_ratios=[1,.15,1,.15,1,.15])
                gs.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
                
                #store all subplots for formatting later           
                all_wba_ax = np.empty([n_rows,n_cols],dtype=plt.Axes)
                all_stim_ax = np.empty([n_rows,n_cols],dtype=plt.Axes)
                
                cnds_to_plot = tr_types_by_cnd + 6*loom_speed + contrast
                print '______________________'
          
            
                # get the index of the first max value of the looming ystim
                # draw a window around here to look for saccades
            
                loom_max_i = all_fly_traces.loc[:,('this_fly',slice(None),cnds_to_plot.flatten(),'ystim')].idxmax().median()
            
                # get saccade window. defined relative loom_max_i ________________________
                l_saccade_win = int(loom_max_i + l_div_v_saccade_win[loom_speed][0])
                r_saccade_win = int(loom_max_i + l_div_v_saccade_win[loom_speed][1])
                this_turn_win = np.arange(l_saccade_win,r_saccade_win) 
            
                max_t = int(loom_max_i + 1*sampling_rate)
            
                # now loop through the conditions/columns ____________________________________
           
                for row in range(n_rows):
                    for col in range(n_cols):
                        cnd = cnds_to_plot[row][col]
                        #print self.stim_types_labels[cnd]
            
                        this_cnd_trs = all_fly_traces.loc[:,('this_fly',slice(None),cnd,'lmr')].columns.get_level_values(1).tolist()
                        n_cnd_trs = np.size(this_cnd_trs)
                
                        # get colormap info ______________________________________________________
                        cmap = plt.cm.get_cmap('seismic') 
                        cNorm  = colors.Normalize(0,n_cnd_trs)
                        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
            
                        # create subplots ________________________________________________________              
                        if row == 0 and col == 0:
                            wba_ax  = plt.subplot(gs[0,col]) 
                            stim_ax = plt.subplot(gs[1,col],sharex=wba_ax)    
                        else:
                            wba_ax  = plt.subplot(gs[0+2*row,col], sharex=all_wba_ax[0][0], sharey=all_wba_ax[0][0]) 
                            stim_ax = plt.subplot(gs[1+2*row,col], sharex=all_stim_ax[0][0], sharey=all_stim_ax[0][0])    
                    
                        wba_ax.set_axis_bgcolor('grey') 
                        # stim_ax.set_axis_bgcolor('black')
                    
                        all_wba_ax[row][col] = wba_ax
                        all_stim_ax[row][col] = stim_ax
                    
                        # loop single trials and plot all signals ________________________________
                        for tr, i in zip(this_cnd_trs,range(n_cnd_trs)):
                
                            this_color = scalarMap.to_rgba(i)        
                     
                            # plot WBA signal ____________________________________________________           
                            wba_trace = all_fly_traces.loc[:,('this_fly',tr,cnd,'lmr')]
                    
                            # get saccades in this trace
                            saccade_is = ~np.isnan(all_fly_saccades.loc[:,('this_fly',tr,cnd)])
                            tr_saccades = np.asarray(all_fly_saccades.loc[saccade_is,('this_fly',tr,cnd)],dtype=int)
                    
                            baseline = np.nanmean(wba_trace[baseline_win])
                            wba_trace = wba_trace - baseline  
                     
                            non_nan_i = np.where(~np.isnan(wba_trace))[0]  ##remove nans earlier/check to make sure nans only occur at the end
                            filtered_wba_trace = butter_lowpass_filter(wba_trace[non_nan_i],30)
                        
                            #filtered_wba_trace = butter_lowpass_filter(wba_trace,30)
                            wba_ax.plot(filtered_wba_trace,color=this_color)

                            # plot all saccade times
                            #wba_ax.plot(tr_saccades,filtered_wba_trace[tr_saccades],marker='^', \
                            #            markersize=8.5,linestyle='',color=this_color)
                    
                            #now plot stimulus traces ____________________________________________
                            stim_ax.plot(all_fly_traces.loc[:,('this_fly',tr,cnd,'ystim')],color=this_color)
                                  
                                    
                    #now format all subplots _____________________________________________________  
           
                #loop though all columns again, format each row ______________________________
                for row in range(n_rows):
                    for col in range(n_cols):      
                        #all_wba_ax[row][col].fill([l_saccade_win,r_saccade_win,r_saccade_win,l_saccade_win],
                        #        [wba_lim[1],wba_lim[1],wba_lim[0],wba_lim[0]],'black',alpha=.1)
                        
                        # set the ylim for the stimulus and correlation rows ______________________
                        all_stim_ax[row][col].set_ylim([0,10])
                         
                        # label axes, show xlim and ylim __________________________________________
                 
                        # remove all time xticklabels
                        all_wba_ax[row][col].tick_params(labelbottom='off')
                         
                        if if_x_zoom:
                            all_wba_ax[row][col].set_xlim(l_div_v_zoom_win[loom_speed])
                        else:
                            all_wba_ax[row][col].set_xlim([0,max_t])
                 
                        if col == 0: #label yaxes
                            if row == 0:           
                                all_wba_ax[row][col].set_ylabel('L-R WBA (deg)')
                                
                                all_wba_ax[row][col].set_ylim(wba_lim)
                                #wba_ax_ylim = all_wba_ax[row][col].get_ylim()
                                all_wba_ax[row][col].set_yticks([wba_lim[0],0,wba_lim[1]])
                                
                                all_stim_ax[row][col].tick_params(labelleft='off')
                                all_stim_ax[row][col].tick_params(labelbottom='off')
                            elif row == n_rows-1:
                                # label time x axis for just col 0 ______________________
                                # divide by sampling rate _______________________________
                                def div_sample_rate(x, pos): 
                                    #The two args are the value and tick position 
                                    return (x-s_iti)/sampling_rate
                         
                                formatter = FuncFormatter(div_sample_rate) 
                                all_stim_ax[row][col].xaxis.set_major_formatter(formatter)
                                         
                                all_stim_ax[row][col].tick_params(labelbottom='on')
                                all_stim_ax[row][col].tick_params(labelleft='off')
                                all_stim_ax[row][col].set_xlabel('Time from loom start (s)')
                                all_wba_ax[row][col].tick_params(labelleft='off')
                            else:
                                all_wba_ax[row][col].tick_params(labelleft='off')
                                all_stim_ax[row][col].tick_params(labelleft='off')
                                all_stim_ax[row][col].tick_params(labelbottom='off')
                        else: # remove all ylabels 
                                all_wba_ax[row][col].tick_params(labelleft='off')
                                all_stim_ax[row][col].tick_params(labelleft='off')
                                all_stim_ax[row][col].tick_params(labelbottom='off')
                        
                  
                    #now annotate stimulus positions, title ______________________________________      
                    fig.text(.22,.905,'Left',fontsize=12)
                    fig.text(.495,.905,'Center',fontsize=12)
                    fig.text(.775,.905,'Right',fontsize=12)
                    
                    fig.text(.01,.87,'33d max disk',fontsize=12)
                    fig.text(.01,.6,'50d max disk',fontsize=12)
                    fig.text(.01,.33,'90d max disk',fontsize=12)
            
                    figure_txt = title_txt + '  '+l_div_v_txt[loom_speed]+'  '+contrast_txt[contrast]
                    fig.text(.2,.95,figure_txt,fontsize=18) 
            
                    #fig.text(.05,.95,tr_info_str,fontsize=14) 
                   
                   
                    plt.draw()
            
                    if if_save:
                        saveas_path = '/Users/jamie/bin/figures/'
                        if if_x_zoom:
                            plt.savefig(saveas_path + figure_txt + '_looming_wba_zoomed.png',\
                            bbox_inches='tight',dpi=100) 
                        else:
                            plt.savefig(saveas_path + figure_txt + '_looming_wba.png',\
                                        bbox_inches='tight',dpi=100) 
                        #plt.close('all')
                        
    def plot_wba_contrast_x_position(self,title_txt='',l_div_v_list=[0,1,2],max_size_list=[0,1],\
                        wba_lim=[-60,60],if_save=True,if_x_zoom=True): 
        
        # for each l/v stim parameter and min contrast 
        # plot lmr wba for three rows of max disk size x 
        # three columns of looming direction
        
        sampling_rate = 1000 # in hertz ********* move to fly info
        s_iti = 2 * sampling_rate #********* move to fly info
        
        baseline_win = range(0,int(1*sampling_rate))  #**************** check this.
                # time relative loom start - s_iti
                # do not average out the visual onset
        
        
        contrast_txt = ['3:7 contrast','4:7 contrast','5:7 contrast']
        l_div_v_txt = ['22 l div v','44 l div v','88 l div v']
        max_disk_txt = ['50 degrees', '90 degrees']
        
        tr_types_by_cnd = np.asarray([[0,18,36],[1,19,37],[2,20,38]]) 
        
        
        # define saccade window relative loom max time
        l_div_v_saccade_win = np.array([[-.25,.25],[-.75,.5],[-.5,.5]])*sampling_rate
        l_div_v_saccade_win = l_div_v_saccade_win.astype(int)               
                       
        # define zoomed x axis range relative loom start - s_iti ------------- update this
        l_div_v_zoom_win = np.array([[s_iti,s_iti+(1*sampling_rate)],  \
                                     [s_iti,s_iti+(1.5*sampling_rate)], \
                                     [s_iti,s_iti+(2.75*sampling_rate)]],dtype=int)
        
        #get all traces and detect saccades ______________________________________________
        all_fly_traces, all_fly_saccades = self.get_traces_by_stim('this_fly',s_iti,get_saccades=True)
                
        n_rows = 3
        n_cols = 3
        
        #now plot one figure for each looming speed ______________________________________
        for loom_speed in l_div_v_list: 
            for max_size in max_size_list: 
                fig = plt.figure(figsize=(10.7875,7.925))       #(16.5, 9))
                gs = gridspec.GridSpec(6,n_cols,height_ratios=[1,.15,1,.15,1,.15])
                gs.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
        
                #store all subplots for formatting later           
                all_wba_ax = np.empty([n_rows,n_cols],dtype=plt.Axes)
                all_stim_ax = np.empty([n_rows,n_cols],dtype=plt.Axes)
        
                #************ update this -- make an ndarray for conds. include title txt
                
                cnds_to_plot = tr_types_by_cnd + 3*max_size + 6*loom_speed
                print '______________________'
          
            
                # get the index of the first max value of the looming ystim
                # draw a window around here to look for saccades
            
                loom_max_i = all_fly_traces.loc[:,('this_fly',slice(None),cnds_to_plot.flatten(),'ystim')].idxmax().median()
            
                # get saccade window. defined relative loom_max_i ________________________
                l_saccade_win = int(loom_max_i + l_div_v_saccade_win[loom_speed][0])
                r_saccade_win = int(loom_max_i + l_div_v_saccade_win[loom_speed][1])
                this_turn_win = np.arange(l_saccade_win,r_saccade_win) 
            
                max_t = int(loom_max_i + 1*sampling_rate)
            
                # now loop through the conditions/columns ____________________________________
           
                for row in range(n_rows):
                    for col in range(n_cols):
                        cnd = cnds_to_plot[row][col]
                        #print self.stim_types_labels[cnd]
            
                        this_cnd_trs = all_fly_traces.loc[:,('this_fly',slice(None),cnd,'lmr')].columns.get_level_values(1).tolist()
                        n_cnd_trs = np.size(this_cnd_trs)
                
                        # get colormap info ______________________________________________________
                        cmap = plt.cm.get_cmap('seismic') 
                        cNorm  = colors.Normalize(0,n_cnd_trs)
                        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
            
                        # create subplots ________________________________________________________              
                        if row == 0 and col == 0:
                            wba_ax  = plt.subplot(gs[0,col]) 
                            stim_ax = plt.subplot(gs[1,col],sharex=wba_ax)    
                        else:
                            wba_ax  = plt.subplot(gs[0+2*row,col], sharex=all_wba_ax[0][0], sharey=all_wba_ax[0][0]) 
                            stim_ax = plt.subplot(gs[1+2*row,col], sharex=all_wba_ax[0][0], sharey=all_stim_ax[0][0])    
                    
                        wba_ax.set_axis_bgcolor('grey')
                        # stim_ax.set_axis_bgcolor('black')
                    
                        # for debugging
                        #wba_ax.set_title(self.stim_types_labels[cnd])
                        
                        all_wba_ax[row][col] = wba_ax
                        all_stim_ax[row][col] = stim_ax
                    
                        # loop single trials and plot all signals ________________________________
                        for tr, i in zip(this_cnd_trs,range(n_cnd_trs)):
                
                            this_color = scalarMap.to_rgba(i)        
                     
                            # plot WBA signal ____________________________________________________           
                            wba_trace = all_fly_traces.loc[:,('this_fly',tr,cnd,'lmr')]
                    
                            # get saccades in this trace
                            saccade_is = ~np.isnan(all_fly_saccades.loc[:,('this_fly',tr,cnd)])
                            tr_saccades = np.asarray(all_fly_saccades.loc[saccade_is,('this_fly',tr,cnd)],dtype=int)
                    
                            baseline = np.nanmean(wba_trace[baseline_win])
                            wba_trace = wba_trace - baseline  
                     
                            non_nan_i = np.where(~np.isnan(wba_trace))[0]  ##remove nans earlier/check to make sure nans only occur at the end
                            filtered_wba_trace = butter_lowpass_filter(wba_trace[non_nan_i],30)
                        
                            #filtered_wba_trace = butter_lowpass_filter(wba_trace,30)
                            wba_ax.plot(filtered_wba_trace,color=this_color)

                            # plot all saccade times
                            #wba_ax.plot(tr_saccades,filtered_wba_trace[tr_saccades],marker='^', \
                            #            markersize=8.5,linestyle='',color=this_color)
                    
                            #now plot stimulus traces ____________________________________________
                            stim_ax.plot(all_fly_traces.loc[:,('this_fly',tr,cnd,'ystim')],color=this_color)
                                  
                                    
                    #now format all subplots _____________________________________________________  
           
                    
                    #loop though all columns again, format each row ______________________________
                for row in range(n_rows):
                    for col in range(n_cols):      
                        #all_wba_ax[row][col].fill([l_saccade_win,r_saccade_win,r_saccade_win,l_saccade_win],
                        #        [wba_lim[1],wba_lim[1],wba_lim[0],wba_lim[0]],'black',alpha=.1)
                        
                        # set the ylim for the stimulus and correlation rows ______________________
                        all_stim_ax[row][col].set_ylim([0,10])
                         
                        # label axes, show xlim and ylim __________________________________________
                 
                        # remove all time xticklabels
                        all_wba_ax[row][col].tick_params(labelbottom='off')
                         
                        if if_x_zoom:
                            all_wba_ax[row][col].set_xlim(l_div_v_zoom_win[loom_speed])
                        else:
                            all_wba_ax[row][col].set_xlim([0,max_t])
                 
                        if col == 0: #label yaxes
                            if row == 0:           
                                all_wba_ax[row][col].set_ylabel('L-R WBA (deg)')
                                
                                
                                all_wba_ax[row][col].set_ylim(wba_lim)
                                #wba_ax_ylim = all_wba_ax[row][col].get_ylim()
                                all_wba_ax[row][col].set_yticks([wba_lim[0],0,wba_lim[1]])
                                
                                
                                all_stim_ax[row][col].tick_params(labelleft='off')
                                all_stim_ax[row][col].tick_params(labelbottom='off')
                            elif row == n_rows-1:
                                # label time x axis for just col 0 ______________________
                                # divide by sampling rate _______________________________
                                def div_sample_rate(x, pos): 
                                    #The two args are the value and tick position 
                                    return (x-s_iti)/sampling_rate
                         
                                formatter = FuncFormatter(div_sample_rate) 
                                all_stim_ax[row][col].xaxis.set_major_formatter(formatter)
                                         
                                all_stim_ax[row][col].tick_params(labelbottom='on')
                                all_stim_ax[row][col].tick_params(labelleft='off')
                                all_stim_ax[row][col].set_xlabel('Time from loom start (s)')
                                all_wba_ax[row][col].tick_params(labelleft='off')
                            else:
                                all_wba_ax[row][col].tick_params(labelleft='off')
                                all_stim_ax[row][col].tick_params(labelleft='off')
                                all_stim_ax[row][col].tick_params(labelbottom='off')
                        else: # remove all ylabels 
                                all_wba_ax[row][col].tick_params(labelleft='off')
                                all_stim_ax[row][col].tick_params(labelleft='off')
                                all_stim_ax[row][col].tick_params(labelbottom='off')
                        
                  
                    #now annotate stimulus positions, title ______________________________________      
                    fig.text(.22,.905,'Left',fontsize=12)
                    fig.text(.495,.905,'Center',fontsize=12)
                    fig.text(.775,.905,'Right',fontsize=12)
                    
                    fig.text(.01,.87,contrast_txt[0],fontsize=12)
                    fig.text(.01,.6,contrast_txt[1],fontsize=12)
                    fig.text(.01,.33,contrast_txt[2],fontsize=12)
            
                    figure_txt = title_txt + '  '+l_div_v_txt[loom_speed]+'  '+max_disk_txt[max_size] + ' max disk'
                    fig.text(.2,.95,figure_txt,fontsize=18) 
            
                    #fig.text(.05,.95,tr_info_str,fontsize=14) 
                   
                   
                    plt.draw()
            
                    if if_save:
                        saveas_path = '/Users/jamie/bin/figures/'
                        if if_x_zoom:
                            plt.savefig(saveas_path + figure_txt + '_looming_wba_zoomed.png',bbox_inches='tight',dpi=100) 
                        else:
                            plt.savefig(saveas_path + figure_txt + '_looming_wba.png',\
                                        bbox_inches='tight',dpi=100) 
                        
             
    def plot_each_tr_saccade(self,l_div_v_list=[0],
        wba_lim=[-45,45]): 
        #for each l/v stim parameter, 
        #make figure four rows of signals -- vm, wba, stimulus, vm-wba corr x
        #three columns of looming direction
        
        #time windows in which to examine turning behaviors. these are by eye
        sampling_rate = 10000
        l_div_v_turn_windows = []
        l_div_v_turn_windows.append(range(int(2.45*sampling_rate),int(2.8*sampling_rate)))
        l_div_v_turn_windows.append(range(int(2.95*sampling_rate),int(3.3*sampling_rate)))
        l_div_v_turn_windows.append(range(int(3.85*sampling_rate),int(4.20*sampling_rate)))
        
        s_iti = 20000   #add iti periods
        baseline_win = range(0,5000)  #be careful not to average out the visual transient here.
        
        #get all traces __________________________________________________________________
        all_fly_traces = self.get_traces_by_stim('this_fly',s_iti) 
        
        #now plot one figure for each looming speed ______________________________________
        for loom_speed in l_div_v_list: 
        
            cnds_to_plot = np.arange(0,7,3) + loom_speed
            #0 1 2 ; 3 4 5 ; 6 7 8 
            this_turn_win = l_div_v_turn_windows[loom_speed]
            
            #now loop through the conditions/columns. ____________________________________
            #the signal types are encoded in separate rows(vm, wba, stim, corr)
            for cnd in cnds_to_plot[1:3]:
            
                this_cnd_trs = all_fly_traces.loc[:,('this_fly',slice(None),cnd,'lmr')].columns.get_level_values(1).tolist()
                n_cnd_trs = np.size(this_cnd_trs)
            
                #loop single trials and plot all signals _________________________________
                for tr in this_cnd_trs:
                    
                    #plot Vm signal ______________________________________________________      
                    #vm_trace = all_fly_traces.loc[:,('this_fly',tr,cnd,'vm')]
                    
                    #plot WBA signal _____________________________________________________           
                    wba_trace = all_fly_traces.loc[:,('this_fly',tr,cnd,'lmr')]
                    baseline = np.nanmean(wba_trace[baseline_win])
                    wba_trace = wba_trace - baseline  #always subtract the baseline here
                    
                    saccade_time = find_saccades(wba_trace,True)
                    plt.plot(all_fly_traces.loc[:,('this_fly',tr,cnd,'ystim')])
    
    
                   
    def get_traces_by_stim(self,fly_name='this_fly',iti=25000,get_saccades=False):
    # here extract the traces for each of the stimulus times. 
    # align to looming start, and add the first pre stim and post stim intervals
    # here return a data frame of lwa and rwa wing traces
    # self.stim_types already holds an np.array vector of the trial type indicies
   
    # using a pandas data frame with multilevel indexing! rows = time in ms
    # columns are multileveled -- genotype, fly, trial index, trial type, trace
        
        fly_df = pd.DataFrame()
        fly_saccades_df = pd.DataFrame() #keep empty if not tracking all saccades
       
        for tr in range(self.n_trs):
            this_loom_start = self.tr_starts[tr]
            this_start = this_loom_start - iti
            this_stop = self.tr_stops[tr] + iti
            
            this_stim_type = self.stim_types[tr]
            iterables = [[fly_name],
                         [tr],
                         [this_stim_type],
                         ['lmr','lwa','rwa','ystim']]
            column_labels = pd.MultiIndex.from_product(iterables,names=['fly','tr_i','tr_type','trace']) 
                                                            #is the unsorted tr_type level a problem?    
                   
            tr_traces = np.asarray([self.lmr[this_start:this_stop],
                                         self.lwa[this_start:this_stop],
                                         self.rwa[this_start:this_stop],
                                         self.ystim[this_start:this_stop]]).transpose()  #reshape to avoid transposing
                                          
            tr_df = pd.DataFrame(tr_traces,columns=column_labels) #,index=time_points) 
            fly_df = pd.concat([fly_df,tr_df],axis=1)
            
            
            if get_saccades:
                # make a data structure of saccade times in the same format as the 
                # fly_df trace information
                # data = saccade start times. now not trying to define saccade stops
                # rows = saccade number
                # columns = fly, trial index, trial type
                
                 iterables = [[fly_name],
                             [tr],
                             [this_stim_type]]
                 column_labels = pd.MultiIndex.from_product(iterables,names=['fly','tr_i','tr_type']) 
                                                             
                 saccade_starts = find_saccades(self.lmr[this_start:this_stop])
                 tr_saccade_starts_df = pd.DataFrame(np.transpose(saccade_starts),columns=column_labels)            
                 fly_saccades_df = pd.concat([fly_saccades_df,tr_saccade_starts_df],axis=1)
            
        if get_saccades:
            return fly_df, fly_saccades_df 
        else:  
            return fly_df
        
     
        
    

#---------------------------------------------------------------------------#
def moving_average(values, window):
    #next add gaussian, kernals, etc
    #pads on either end to return an equal length structure,
    #although the edges are distorted
    
    if (window % 2): #is odd 
        window = window + 1; 
    halfwin = window/2
    
    n_values = np.size(values)
    
    padded_values = np.ones(n_values+window)*np.nan
    padded_values[0:halfwin] = np.ones(halfwin)*np.mean(values[0:halfwin])
    padded_values[halfwin:halfwin+n_values] = values
    padded_values[halfwin+n_values:window+n_values+1] = np.ones(halfwin)*np.mean(values[-halfwin:n_values])
  
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(padded_values, weights, 'valid')
    return sma[0:n_values]
    
def xcorr(a, v):
    a = (a - np.mean(a)) / (np.std(a) * (len(a)-1))
    v = (v - np.mean(v)) /  np.std(v)
    xc = np.correlate(a, v, mode='same')
    return xc
    
def read_abf(abf_filename):
        fh = AxonIO(filename=abf_filename)
        segments = fh.read_block().segments
    
        if len(segments) > 1:
            print 'More than one segment in file.'
            return 0

        analog_signals_ls = segments[0].analogsignals
        analog_signals_dict = {}
        for analog_signal in analog_signals_ls:
            analog_signals_dict[analog_signal.name.lower()] = analog_signal

        return analog_signals_dict
        
def process_wings(raw_wings):
    #here shift wing signal -12 ms in time, filling end with nans
    shifted_wings = np.zeros_like(raw_wings)
    shifted_wings[0:-12] = raw_wings[12:]
    shifted_wings[-11:] = raw_wings[-1]   
    
    #now multiply to convert volts to degrees
    processed_wings = -45 + shifted_wings*33.75
    return processed_wings
     
def find_saccades(raw_lmr_trace,test_plot=False):
    #first fill in nans with nearest signal
    lmr_trace = raw_lmr_trace[~np.isnan(raw_lmr_trace)] 
        #this may give different indexing than input
        #ideally fill in nans in wing processing

    # filter lmr signal
    filtered_trace = butter_lowpass_filter(lmr_trace) #6 hertz
    
    # differentiate, take the absolute value
    diff_trace = abs(np.diff(filtered_trace))
     
    # mark saccade start times -- this could be improved
    diff_thres = .01
    cross_d_thres = np.where(diff_trace > diff_thres)[0]
    
    # #use this to find saccade stops
#     saccade_start_candidate = diff_trace[1:-1] < diff_thres  
#     saccade_cont  = diff_trace[2:]   >= diff_thres
#     stacked_start_cont = np.vstack([saccade_start,saccade_cont])
#     candidate_saccade_starts = np.where(np.all(stacked_start_cont,axis=0))[0]
    
    # impose a refractory period for saccades
    d_cross_d_thres = np.diff(cross_d_thres)
    
    refractory_period = .2 * 10000
    if cross_d_thres.size:
        saccade_starts = [cross_d_thres[0]] #include first
        
        #then take those with gaps between saccade events
        other_is = np.where(d_cross_d_thres > refractory_period)[0]+1
        saccade_starts = np.hstack((saccade_starts,cross_d_thres[other_is]))
    else:
        saccade_starts = []
       
    if test_plot:
        fig = plt.figure()
        plt.plot(lmr_trace,'grey')
        plt.plot(filtered_trace,'black')
        plt.plot(1000*diff_trace,'green')
        
        
        plt.plot(cross_d_thres,np.zeros_like(cross_d_thres),'r.')
        plt.plot(saccade_starts,np.ones_like(saccade_starts),'mo')
    
    # return indicies of start and stop times + saccade magnitude 
    return saccade_starts
       
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sp.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=6, fs=1000, order=5): #how does the order change?
    b, a = butter_lowpass(cutoff, fs, order)
    #y = sp.signal.lfilter(b, a, data) #what's the difference here? 
    y = sp.signal.filtfilt(b, a, data)
    return y
      
def write_to_pdf(f_name,figures_list):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(fname)
    for f in figures_list:
        pp.savefig(f)
    pp.close()

def plot_many_flies(path_name, filenames_df):    

    #loop through all genotypes
    genotypes = (pd.unique(filenames_df.values[:,1]))
    print genotypes
    
    for g in genotypes:
        these_genotype_indicies = np.where(filenames_df.values[:,1] == g)[0]
    
        for index in these_genotype_indicies:
            print index
        
            fly = Looming_Behavior(path_name + filenames_df.values[index,0])
            title_txt = filenames_df.values[index,1] + '  ' + filenames_df.values[index,0]
            fly.process_fly()
            fly.plot_wba_stim(title_txt)
        
            saveas_path = '/Users/jamie/bin/figures/'
            plt.savefig(saveas_path + title_txt + '_kir_looming.png',dpi=100)
            plt.close('all')
                    
def get_pop_traces_df(path_name, population_f_names):  
    #loop through all genotypes
    #structure row = time points, aligned to looming start
    #columns: genotype, fly, trial index, trial typa, lwa/rwa
    #just collect these for all flies
    
    #genotypes must be sorted to the labels for columns 
    genotypes = (pd.unique(population_f_names.values[:,1]))
    genotypes = np.sort(genotypes)
    genotypes = genotypes[1:]
    print genotypes
    
    population_df = pd.DataFrame()
    
    #loop through each genotype  
    for g in genotypes:
        g
        these_genotype_indicies = np.where(population_f_names.values[:,1] == g)[0]
    
        for index in these_genotype_indicies:
            print index
        
            fly = Looming_Behavior(path_name + population_f_names.values[index,0])
            fly.process_fly()
            fly_df = fly.get_traces_by_stim(g)
            population_df = pd.concat([population_df,fly_df],axis=1)
    return population_df
     
def plot_pop_flight_behavior_histograms(population_df, wba_lim=[-3,3],cnds_to_plot=range(9)):  
    #for the looming data, plot histograms over time of all left-right
    #wba traces
    
    #instead send the population dataframe as a parameter
    
    #get a two-dimensional multi-indexed data frame with the population data
    #population_df = get_pop_flight_traces(path_name, population_f_names)
   
    #loop through each genotype  --- genotypes must be sorted to be column labels
    #change code so I just do this in the get_pop_flight_traces
    all_genotype_fields = population_df.columns.get_level_values(0)
    genotypes = np.unique(all_genotype_fields)
    
    x_lim = [0, 4075]
    
    for g in genotypes:
        print g
        
        #calculate the number of cells/genotype
        all_cell_names = population_df.loc[:,(g)].columns.get_level_values(0)
        n_cells = np.size(np.unique(all_cell_names))
        
        title_txt = g + ' __ ' + str(n_cells) + ' flies' #also add number of flies and trials here 
        #calculate the number of flies and trials for the caption
    
        fig = plt.figure(figsize=(16.5, 9))
        #change this so I'm not hardcoding the number of axes
        gs = gridspec.GridSpec(6,3,width_ratios=[1,1,1],height_ratios=[4,1,4,1,4,1])
    
        #loop through conditions -- later restrict these
        for cnd in cnds_to_plot:
            grid_row = int(2*math.floor(cnd/3)) #also hardcoding
            grid_col = int(cnd%3)
     
            #plot WBA histogram signal -----------------------------------------------------------    
            wba_ax = plt.subplot(gs[grid_row,grid_col])     
        
            g_lwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'lwa')].as_matrix()
            g_rwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'rwa')].as_matrix()    
            g_lmr = g_lwa - g_rwa
        
            #get baseline, substract from traces
            baseline = np.nanmean(g_lmr[200:700,:],0) #parametize this
            g_lmr = g_lmr - baseline
        
            #just plot the mean for debugging
            #wba_ax.plot(np.nanmean(g_lmr,1))
        
            #now plot the histograms over time. ------------
            max_t = np.shape(g_lmr)[0]
            n_trs = np.shape(g_lmr)[1]
                     
            t_points = range(max_t)
            t_matrix = np.tile(t_points,(n_trs,1))
            t_matrix_t = np.transpose(t_matrix)

            t_flat = t_matrix_t.flatten() 
            g_lmr_flat = g_lmr.flatten()

            #now remove nans
            g_lmr_flat = g_lmr_flat[~np.isnan(g_lmr_flat)]
            t_flat = t_flat[~np.isnan(g_lmr_flat)]

            #calc, plot histogram
            h2d, xedges, yedges = np.histogram2d(t_flat,g_lmr_flat,bins=[200,50],range=[[0, 4200],[-3,3]],normed=True)
            wba_ax.pcolormesh(xedges, yedges, np.transpose(h2d))
        
           
            #plot white line for 0 -----------
            wba_ax.axhline(color=white)
        
            wba_ax.set_xlim(x_lim) 
            
            if grid_row == 0 and grid_col == 0:
                wba_ax.yaxis.set_ticks(wba_lim)
                wba_ax.set_ylabel('L-R WBA (mV)')
            else:
                wba_ax.yaxis.set_ticks([])
            wba_ax.xaxis.set_ticks([])
              
            #now plot stim -----------------------------------------------------------
            stim_ax = plt.subplot(gs[grid_row+1,grid_col])
        
            #assume the first trace of each is typical
            y_stim = population_df.loc[:,(g,slice(None),slice(None),cnd,'ystim')]
            stim_ax.plot(y_stim.iloc[:,0],color=blue)
        
            stim_ax.set_xlim(x_lim) 
            stim_ax.set_ylim([0, 10]) 
        
            if grid_row == 4 and grid_col == 0:
                stim_ax.xaxis.set_ticks(x_lim)
                stim_ax.set_xticklabels(['0','.4075'])
                stim_ax.set_xlabel('Time (s)') 
            else:
                stim_ax.xaxis.set_ticks([])
            stim_ax.yaxis.set_ticks([])
            
        #now annotate        
        fig.text(.06,.8,'left',fontsize=14)
        fig.text(.06,.53,'center',fontsize=14)
        fig.text(.06,.25,'right',fontsize=14)
        
        fig.text(.22,.905,'22 l/v',fontsize=14)
        fig.text(.495,.905,'44 l/v',fontsize=14)
        fig.text(.775,.905,'88 l/v',fontsize=14)
        
        fig.text(.425,.95,title_txt,fontsize=18)        
        plt.draw() 

        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path + title_txt + '_population_kir_looming_histograms.png',dpi=100)
        #plt.close('all')

def plot_pop_flight_behavior_means(population_df, wba_lim=[-3,3], cnds_to_plot=range(9)):  
    #for the looming data, plot the means of all left-right
    #wba traces
    
    #instead send the population dataframe as a parameter
    
    #get a two-dimensional multi-indexed data frame with the population data
    #population_df = get_pop_flight_traces(path_name, population_f_names)
   
    #loop through each genotype  --- genotypes must be sorted to be column labels
    #change code so I just do this in the get_pop_flight_traces
    all_genotype_fields = population_df.columns.get_level_values(0)
    genotypes = np.unique(all_genotype_fields)
    
    x_lim = [0, 4075]
    speed_x_lims = [range(0,2600),range(0,3115),range(0,4075)] #restrict the xlims by condition to not show erroneously long traces
    
    for g in genotypes:
        print g
        
        #calculate the number of cells/genotype
        all_fly_names = population_df.loc[:,(g)].columns.get_level_values(0)
        unique_fly_names = np.unique(all_fly_names)
        n_cells = np.size(unique_fly_names)
        
        title_txt = g + ' __ ' + str(n_cells) + ' flies' #also add number of flies and trials here 
        #calculate the number of flies and trials for the caption
    
        fig = plt.figure(figsize=(16.5, 9))
        #change this so I'm not hardcoding the number of axes
        gs = gridspec.GridSpec(6,3,width_ratios=[1,1,1],height_ratios=[4,1,4,1,4,1])
    
        #loop through conditions -- later restrict these
        for cnd in cnds_to_plot:
            grid_row = int(2*math.floor(cnd/3)) #also hardcoding
            grid_col = int(cnd%3)
            this_x_lim = speed_x_lims[grid_col]
     
            #make the axis --------------------------------
            wba_ax = plt.subplot(gs[grid_row,grid_col])     
        
            #plot the mean of each fly --------------------------------
            for fly_name in unique_fly_names:
                fly_lwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'lwa')].as_matrix()
                fly_rwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'rwa')].as_matrix()    
                fly_lmr = fly_lwa - fly_rwa
        
                #get baseline, substract from traces
                baseline = np.nanmean(fly_lmr[200:700,:],0) #parametize this
                fly_lmr = fly_lmr - baseline
            
                wba_ax.plot(np.nanmean(fly_lmr[this_x_lim,:],1),color=black,linewidth=.5)        
        
        
            #plot the genotype mean --------------------------------   
            g_lwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'lwa')].as_matrix()
            g_rwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'rwa')].as_matrix()    
            g_lmr = g_lwa - g_rwa
        
            #get baseline, substract from traces
            baseline = np.nanmean(g_lmr[200:700,:],0) #parametize this
            g_lmr = g_lmr - baseline
            
            wba_ax.plot(np.nanmean(g_lmr[this_x_lim,:],1),color=magenta,linewidth=2)
              
            #plot black line for 0 --------------------------------
            wba_ax.axhline(color=black)
        
            #format axis --------------------------------
            wba_ax.set_xlim(x_lim) 
            wba_ax.set_ylim(wba_lim)
            
            if grid_row == 0 and grid_col == 0:
                wba_ax.yaxis.set_ticks(wba_lim)
                wba_ax.set_ylabel('L-R WBA (mV)')
            else:
                wba_ax.yaxis.set_ticks([])
            wba_ax.xaxis.set_ticks([])
              
            #now plot stim -----------------------------------------------------------
            stim_ax = plt.subplot(gs[grid_row+1,grid_col])
        
            #assume the first trace of each is typical
            y_stim = population_df.loc[:,(g,slice(None),slice(None),cnd,'ystim')]
            stim_ax.plot(y_stim.iloc[:,0],color=blue)
        
            stim_ax.set_xlim(x_lim) 
            stim_ax.set_ylim([0, 10]) 
        
            if grid_row == 4 and grid_col == 0:
                stim_ax.xaxis.set_ticks(x_lim)
                stim_ax.set_xticklabels(['0','.4075'])
                stim_ax.set_xlabel('Time (s)') 
            else:
                stim_ax.xaxis.set_ticks([])
            stim_ax.yaxis.set_ticks([])
            
        #now annotate        
        fig.text(.06,.8,'left',fontsize=14)
        fig.text(.06,.53,'center',fontsize=14)
        fig.text(.06,.25,'right',fontsize=14)
        
        fig.text(.22,.905,'22 l/v',fontsize=14)
        fig.text(.495,.905,'44 l/v',fontsize=14)
        fig.text(.775,.905,'88 l/v',fontsize=14)
        
        fig.text(.425,.95,title_txt,fontsize=18)        
        plt.draw() 

        saveas_path = '/Users/jamie/bin/figures/'
        plt.savefig(saveas_path + title_txt + '_population_kir_looming_means.png',dpi=100)
        plt.close('all')
        
def plot_pop_flight_behavior_means_overlay(population_df, two_genotypes, wba_lim=[-3,3], cnds_to_plot=range(9)):  
    #for the looming data, plot the means of all left-right
    #wba traces
    
    #instead send the population dataframe as a parameter
    
    #get a two-dimensional multi-indexed data frame with the population data
    #population_df = get_pop_flight_traces(path_name, population_f_names)
   
    #loop through each genotype  --- genotypes must be sorted to be column labels
    #change code so I just do this in the get_pop_flight_traces
    all_genotype_fields = population_df.columns.get_level_values(0)
    genotypes = np.unique(all_genotype_fields)
    
    x_lim = [0, 4075]
    speed_x_lims = [range(0,2600),range(0,3115),range(0,4075)] #restrict the xlims by condition to not show erroneously long traces
    
    fig = plt.figure(figsize=(16.5, 9))
    #change this so I'm not hardcoding the number of axes
    gs = gridspec.GridSpec(6,3,width_ratios=[1,1,1],height_ratios=[4,1,4,1,4,1])
    
    genotype_colors = [magenta, blue]
    
    i = 0 
    title_txt = '';
    for g in two_genotypes:
        print g
        
        #calculate the number of cells/genotype
        all_fly_names = population_df.loc[:,(g)].columns.get_level_values(0)
        unique_fly_names = np.unique(all_fly_names)
        n_cells = np.size(unique_fly_names)
        
        title_txt = title_txt + g + ' __ ' + str(n_cells) + ' flies ' #also add number of flies and trials here 
        #calculate the number of flies and trials for the caption
        
        #loop through conditions -- later restrict these
        for cnd in cnds_to_plot:
            grid_row = int(2*math.floor(cnd/3)) #also hardcoding
            grid_col = int(cnd%3)
            this_x_lim = speed_x_lims[grid_col]
     
            #make the axis --------------------------------
            wba_ax = plt.subplot(gs[grid_row,grid_col])     
        
            #plot the mean of each fly --------------------------------
            for fly_name in unique_fly_names:
                fly_lwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'lwa')].as_matrix()
                fly_rwa = population_df.loc[:,(g,fly_name,slice(None),cnd,'rwa')].as_matrix()    
                fly_lmr = fly_lwa - fly_rwa
        
                #get baseline, substract from traces
                baseline = np.nanmean(fly_lmr[200:700,:],0) #parametize this
                fly_lmr = fly_lmr - baseline
            
                wba_ax.plot(np.nanmean(fly_lmr[this_x_lim,:],1),color=genotype_colors[i],linewidth=.25)        
        
            #plot the genotype mean --------------------------------   
            g_lwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'lwa')].as_matrix()
            g_rwa = population_df.loc[:,(g,slice(None),slice(None),cnd,'rwa')].as_matrix()    
            g_lmr = g_lwa - g_rwa
        
            #get baseline, substract from traces
            baseline = np.nanmean(g_lmr[200:700,:],0) #parametize this
            g_lmr = g_lmr - baseline
            
            wba_ax.plot(np.nanmean(g_lmr[this_x_lim,:],1),color=genotype_colors[i],linewidth=2)
              
            #plot black line for 0 --------------------------------
            wba_ax.axhline(color=black)

            #format axis --------------------------------
            wba_ax.set_xlim(x_lim) 
            wba_ax.set_ylim(wba_lim)

            if grid_row == 0 and grid_col == 0:
                wba_ax.yaxis.set_ticks(wba_lim)
                wba_ax.set_ylabel('L-R WBA (mV)')
            else:
                wba_ax.yaxis.set_ticks([])
            wba_ax.xaxis.set_ticks([])
          
            #now plot stim -----------------------------------------------------------
            stim_ax = plt.subplot(gs[grid_row+1,grid_col])

            #assume the first trace of each is typical
            y_stim = population_df.loc[:,(g,slice(None),slice(None),cnd,'ystim')]
            stim_ax.plot(y_stim.iloc[:,0],color=black)

            stim_ax.set_xlim(x_lim) 
            stim_ax.set_ylim([0, 10]) 

            if grid_row == 4 and grid_col == 0:
                stim_ax.xaxis.set_ticks(x_lim)
                stim_ax.set_xticklabels(['0','.4075'])
                stim_ax.set_xlabel('Time (s)') 
            else:
                stim_ax.xaxis.set_ticks([])
            stim_ax.yaxis.set_ticks([])
            
        i = i + 1
        
    #now annotate        
    fig.text(.06,.8,'left',fontsize=14)
    fig.text(.06,.53,'center',fontsize=14)
    fig.text(.06,.25,'right',fontsize=14)
    
    fig.text(.22,.905,'22 l/v',fontsize=14)
    fig.text(.495,.905,'44 l/v',fontsize=14)
    fig.text(.775,.905,'88 l/v',fontsize=14)        

    fig.text(.1,.95,two_genotypes[0],color='magenta',fontsize=18)
    fig.text(.2,.95,two_genotypes[1],color='blue',fontsize=18)
    plt.draw()
    
    saveas_path = '/Users/jamie/bin/figures/'
    plt.savefig(saveas_path + title_txt + '_population_kir_looming_means_overlay_' 
        + two_genotypes[0] + '_' + two_genotypes[1] + '.png',dpi=100)
    #plt.close('all')


    

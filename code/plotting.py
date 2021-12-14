import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.special import gamma as gamma_func
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

import parameters
import cosmodata


def set_rcParams(arial=False):
    mpl.rcParams.update(mpl.rcParamsDefault) # reset defaults

    if arial:
        mpl.rcParams['font.family'] = "sans-serif"
        mpl.rcParams['font.sans-serif'] = "Arial"

    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"

    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False

    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["axes.titlesize"]= 10
    mpl.rcParams["legend.fontsize"] = 7
    mpl.rcParams["legend.title_fontsize"] = 10


def overview_agegroups(model, path=None, silent=False, arial=False, scen=1):
    set_rcParams(arial=arial)
    t = model.times
    M = model.M
    data = model.chopped_data().sum(axis=2)
    AGdata = model.chopped_data()
    ags = AGdata.shape[2]

    colors = mpl.cm.viridis_r(np.linspace(0.,1.,ags))
    scen_colors = {
        1: '#1B1919FF',
        2: '#484343FF',
        3: '#00468BFF',
        4: '#0099B4FF',
        5: '#46DCBAFF'}


    fig = plt.figure(figsize=(6, 8), constrained_layout=True)
    grid = fig.add_gridspec(ncols=3, nrows=5, hspace=0.1, wspace=0.1)
    

    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1],sharex=ax1)
    ax3 = fig.add_subplot(grid[2])
    ax4 = fig.add_subplot(grid[3],sharex=ax1)
    ax5 = fig.add_subplot(grid[4],sharex=ax1)
    ax6 = fig.add_subplot(grid[5],sharex=ax1)
    ax7 = fig.add_subplot(grid[6],sharex=ax1)
    ax8 = fig.add_subplot(grid[7],sharex=ax1)
    ax9 = fig.add_subplot(grid[8],sharex=ax1)
    ax10 = fig.add_subplot(grid[9],sharex=ax1)
    ax11 = fig.add_subplot(grid[10],sharex=ax1, sharey=ax10)
    ax12 = fig.add_subplot(grid[11],sharex=ax1, sharey=ax10)
    ax13 = fig.add_subplot(grid[12],sharex=ax1)
    ax14 = fig.add_subplot(grid[13],sharex=ax1)
    ax15 = fig.add_subplot(grid[14],sharex=ax1)

    axs = [ax1,ax2,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15]
    # (S,V,Wn,Wv,E,EBn,EBv,I,IBn,IBv,ICU,ICUv,R,Rv,UC,WC,D,C)

    # Theta*I + (1-kappa)*Theta*(IBn+IBv) + Theta_ICU*(ICU+ICUv)
    dD = model.Theta*AGdata[:,7,:] \
         + (1-model.kappa)*model.Theta*(AGdata[:,8,:]+AGdata[:,9,:]) \
         + model.Theta_ICU*(AGdata[:,10,:]+AGdata[:,11,:])

    phis = np.array(list(map(model.get_phis, t, AGdata))).sum(axis=(2))
    shift = round(model.tau_vac1/model.step_size)
    d1a = np.roll(phis[:,0,:], -shift, axis=0)
    d1a[-shift:,:] = 0
    shift = round(model.tau_vac1/2./model.step_size)
    d1b = np.roll(phis[:,0,:], -shift, axis=0)
    d1b[-shift:,:] = 0
    shift = round(model.tau_vac2/model.step_size)
    d2 = np.roll(phis[:,1,:], -shift, axis=0)
    d2[-shift:,:] = 0
    def NPI(t):
        return max(np.linalg.eigvals((np.moveaxis(model.Cs,0,2) * model.k_lowH(t)).sum(axis=2)))
    def Rt(t):
        #Stapel aus 6x6 Matrizen, 4 lang. Elementweise gewichtete Addition und anschließende Summation über Zeilen
        return np.matmul( (np.moveaxis(model.Cs,0,2) * model.k_selfregulation(t)).sum(axis=2), np.ones(6) )


    for ag in range(ags):
        ax1.plot(t,list(map(NPI,t)), color=scen_colors[scen])
        ax2.plot(t,model.Gamma(t) * np.array(list(map(Rt,t)))[:,ag], color=colors[ag])
        # ax3 legend
        ax4.plot(t,model.rho*(AGdata[:,4,ag]+AGdata[:,5,ag]+AGdata[:,6,ag])/(M[ag]/1e6), color=colors[ag])
        ax5.plot(t,(AGdata[:,6,ag])/(AGdata[:,4,ag]+AGdata[:,5,ag]+AGdata[:,6,ag]), color=colors[ag])
        ax6.plot(t,AGdata[:,16,ag]+AGdata[:,17,ag], color=colors[ag])
        ax7.plot(t,(AGdata[:,10,ag]+AGdata[:,11,ag])/(M[ag]/1e6), color=colors[ag])
        ax8.plot(t,AGdata[:,11,ag]/(AGdata[:,10,ag]+AGdata[:,11,ag]), color=colors[ag])
        ax9.plot(t, (AGdata[:,10,ag]+AGdata[:,11,ag])/ (AGdata[:,10,:]+AGdata[:,11,:]).sum(axis=1) , color=colors[ag])
        ax10.plot(t,AGdata[:,0,ag]/M[ag], color=colors[ag])
        ax11.plot(t,(AGdata[:,1,ag]+AGdata[:,12,ag]+AGdata[:,13,ag])/M[ag], color=colors[ag])
        ax12.plot(t,(AGdata[:,2,ag]+AGdata[:,3,ag])/M[ag], color=colors[ag])
        ax13.plot(t,d1a[:,ag]/(M[ag]/1e6), color=colors[ag])
        ax14.plot(t,d2[:,ag]/(M[ag]/1e6), color=colors[ag])
        ax15.plot(t,(d1a[:,ag]+d1b[:,ag]+d2[:,ag]), color=colors[ag])

        #ax13.plot(t,np.array(list(map(Rt,t)))[:,ag], color=colors[ag])
        #ax14.plot(t,model.Gamma(t), color='black')
        #ax15.plot(t,model.beta/model.gamma[ag] * model.Gamma(t) * np.array(list(map(Rt,t)))[:,ag], color=colors[ag])

    for i,ax in enumerate(axs):
#        ax.set_title(titles[i])
        ax.set_ylim(0,None)

    ax1.set_ylabel("Contact level\ninfluenced by NPIs")
    ax2.set_ylabel("Contacts including\nreduction and seasonality")
    #ax3 legend
    ax4.set_ylabel("Daily new cases\nper million in age group")
    ax5.set_ylabel("Share of reinfections\nof daily new cases")
    ax6.set_ylabel("Cumulative deaths\nper million of population")
    ax7.set_ylabel("ICU occupancy\nper million in age group")
    ax8.set_ylabel("Share of reinfections\nof ICU occupancy")
    ax9.set_ylabel("Share of total\nICU patients")
    ax10.set_ylabel("Susceptible fraction\nof the population")
    ax11.set_ylabel("Immune fraction\nof the population")
    ax12.set_ylabel("Waned immune fraction\nof the population")
    ax13.set_ylabel("Daily first-time vac.\nper million in age group")
    ax14.set_ylabel("Daily booster vac.\nper million in age group")
    ax15.set_ylabel("Daily total vac.\nper million of population")
    #ax15.set_ylabel("Total gross Rt")

    ax1.set_xticks([45, 135, 45+2*90, 45+3*90])
    ax1.set_xticklabels(['Oct.','Jan.','Apr.','July'])

    for ax in [ax5,ax8,ax10,ax11,ax12]:
        ax.set_ylim(0,1)

    for ax in axs:
        ax.axvline(180, ls=':', color='#ADB6B6FF', zorder=0)
        ax.set_xlabel('2021            2022')

#    for ax in [ax2,ax3,ax5,ax6,ax8,ax9,ax11,ax12]:
#        plt.setp(ax.get_yticklabels(), visible=False)

    # Build Legend in Panel 3
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    handles = [mpl.lines.Line2D([], [], color=colors[0], label='0-19'),
               mpl.lines.Line2D([], [], color=colors[1], label='20-39'),
               mpl.lines.Line2D([], [], color=colors[2], label='40-59'),
               mpl.lines.Line2D([], [], color=colors[3], label='60-69'),
               mpl.lines.Line2D([], [], color=colors[4], label='70-79'),
               mpl.lines.Line2D([], [], color=colors[5], label='80+')]
    ax3.legend(handles=handles, title='Age groups', loc='upper center', ncol=1, frameon=True)
    
    fig.align_ylabels()
    
    if not silent: plt.show()
    if path!=None: fig.savefig(path)



def sixpanels(models, path=None, silent=False, arial=False, ICUcap=None, full_wave=None):
    set_rcParams(arial=arial)
    mpl.rcParams["legend.fontsize"] = 7

    m1, m2, m3 = models
    t = m1.times
    data = [m1.chopped_data().sum(axis=2), m2.chopped_data().sum(axis=2), m3.chopped_data().sum(axis=2)]
    AGdata = [m1.chopped_data(), m2.chopped_data(), m3.chopped_data()]

    fig = plt.figure(figsize=(6., 3.5), constrained_layout=True)
    grid = fig.add_gridspec(ncols=3, nrows=2, wspace=0.1)

    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1], sharex=ax1)
    ax3 = fig.add_subplot(grid[2], sharex=ax1)
    ax4 = fig.add_subplot(grid[3], sharex=ax1)
    ax5 = fig.add_subplot(grid[4])
    ax6 = fig.add_subplot(grid[5])

    colors = {
        'low':'#0099B4FF', 'mid':'#00468BFF', 'high':'#1B1919FF', 'free':'#1B1919FF',
        'lowL':'#FFFFFFFF', 'midL':'#FFFFFFFF', 'highL':'#FFFFFFFF',
#        'lowL':'#0099B499', 'midL':'#00468B99', 'highL':'#1B191999',
        'line':'#ADB6B6FF', 'ICUcap':'#FFAAAA',
        'now':'#ADB6B6FF', 'nowL':'#FFFFFFFF',
#        'now':'#93dfedFF', 'nowL':'#93dfed99',
        'FW':'#ED0000FF',
    }
    main_colors = [colors['high'],colors['mid'],colors['low']]
    main_colors_L = [colors['highL'],colors['midL'],colors['lowL']]



#    ax1.plot(t[1800:], np.ones(1800), color=colors['free'])
    def Rt(m,t):
        return max(np.linalg.eigvals((np.moveaxis(m.Cs,0,2) * m.k_lowH(t)).sum(axis=2)))
    
    for i,m in enumerate([m1,m2,m3]):
#        ax1.plot(t, np.array(list(map(m.Rt, t)))/m.R0, color=main_colors[i])
#        ax1.plot(t, np.array(list(map(m.Rt, t))), color=main_colors[i])
        ax1.plot(t, list(map(Rt, [m]*len(t), t)), color=main_colors[i])
        ax2.plot(t, m.rho*(data[i][:,4]+data[i][:,5]+data[i][:,6]), color=main_colors[i])
        ax3.plot(t, data[i][:,10]+data[i][:,11], color=main_colors[i])

        phis = np.array(list(map(m.get_phis, t, AGdata[i]))).sum(axis=(2,3))
        shift = round(m.tau_vac1/m.step_size)
        d1a = np.roll(phis[:,0], -shift)
        d1a[-shift:] = 0
        shift = round(m.tau_vac1/2./m.step_size)
        d1b = np.roll(phis[:,0], -shift)
        d1b[-shift:] = 0
        shift = round(m.tau_vac2/m.step_size)
        d2 = np.roll(phis[:,1], -shift)
        d2[-shift:] = 0
        ax4.plot(t, d1a+d1b+d2, color=main_colors[i])

    ax5.bar(1, data[0][0,1]/1e6, 0.5,
        align='center', color=colors['now'], edgecolor='black', zorder=-1)
    ax5.bar(1, (data[0][0,12]+data[0][0,13])/1e6, 0.5,
        align='center', color=colors['nowL'], edgecolor='black', zorder=-1, bottom=data[0][0,1]/1e6)

    offset = 0.5
    for i in [2,4]:
        for ab,m,j in zip([-0.5,0,0.5],[m1,m2,m3],[0,1,2]):
            ax5.bar(offset+i+ab, data[j][900*i-1,1]/1e6, 0.5,  
                align='center', color=main_colors[j], edgecolor='black', zorder=-1)
            ax5.bar(offset+i+ab, (data[j][900*i-1,12]+data[j][900*i-1,13])/1e6, 0.5,
                align='center', color=main_colors_L[j], edgecolor='black', zorder=-1,
                    bottom=data[j][900*i-1,1]/1e6)
            ax6.bar(offset+i+ab, data[j][900*i-1,16], 0.5, 
                align='center', color=main_colors[j], edgecolor='black', zorder=-3)
 


    for ax in [ax1,ax2,ax3,ax4]:
        ax.axvline(180, ls=':', color=colors['line'], zorder=0)
        ax.set_xlabel('2021            2022')


    ax1.set_ylim(0,None)
    ax2.set_ylim(0,None)
    ax3.set_ylim(0,None)
    ax4.set_ylim(0,None)
    ax5.set_ylim(0,1)
    ax6.set_ylim(0,None)

#    ax5.set_yticks([0.,0.25,0.5,0.75,1.])

    ax1.set_ylabel("Contact levels\ninfluenced by NPIs")
    ax2.set_ylabel("Daily new cases\nper million")
    ax3.set_ylabel("ICU occupancy\nper million")
#    ax4.set_ylabel("Reproduction number")
    ax4.set_ylabel("Daily vaccinations\nper million")
    ax5.set_ylabel("Immune fraction\nof the population")
    ax6.set_ylabel("Total deaths\nper million")

    #Panel 1
    ax1.text(0,1+0.05,'Scenario 1', size=7, color=colors['high'])
    ax1.text(0,0.75+0.05,'Scenario 2', size=7, color=colors['mid'])
    ax1.text(0,0.5+0.05,'Scenario 3', size=7, color=colors['low'])
    ax1.text(200,0.5+0.05,'No restrictions', size=7, color=colors['line'])
    
    #Lifting of restrictions
    ax1.text(0.54,0.05,'Lifting of\nrestrictions', size=7, color=colors['line'], transform=ax1.transAxes)
    for ax in [ax2,ax3,ax4]:
        l,u = ax.get_ylim()
        ax.set_ylim(l,u+0.15*(u-l))
        ax.text(0.54,0.9,'Lifting of\nrestrictions', size=7, color=colors['line'], transform=ax.transAxes)

    for ax, label in zip([ax1,ax2,ax3,ax4,ax5,ax6], ['A','B','C','D','E','F']):
        ax.text(-.12,1.1,label, size=12, weight='bold', color='black', transform=ax.transAxes)

    if ICUcap != None:
        ax3.axhspan(ICUcap-2,ICUcap+2, xmax=0.92, facecolor=colors['ICUcap'], edgecolor=None, zorder=-1)
        ax3.text(1.0,0.3,'ICU capacity', size=7, color='red', rotation=-90, transform=ax3.transAxes)
        ax3.scatter(380,ICUcap, marker="<", color='grey')

    if full_wave != None:
        m = full_wave
        d = m.chopped_data().sum(axis=2)
        ax2.plot(t, m.rho*(d[:,4]+d[:,5]+d[:,6]), color=colors['FW'], ls=':')
        ax3.plot(t, d[:,10]+d[:,11], color=colors['FW'], ls=':')
        ax2.text(0.58,0.72,'Full wave', size=7, color=colors['FW'], transform=ax2.transAxes)
        ax3.text(0.58,0.72,'Full wave', size=7, color=colors['FW'], transform=ax3.transAxes)

    ax1.set_xticks([45, 135, 45+2*90, 45+3*90])
    ax1.set_xticklabels(['Oct.','Jan.','Apr.','July'])


    ax5_x = [1,offset+2,offset+4]
    ax5labels=['Initial','After\nwinter', 'After\none year']
    ax5.set_xticks(ax5_x)
    ax5.set_xticklabels(ax5labels)

    ax6_x = [offset+2,offset+4]
    ax6labels=['After\nwinter', 'After\none year']
    ax6.set_xticks(ax6_x)
    ax6.set_xticklabels(ax6labels)

    handles = [mpl.patches.Patch(facecolor=colors['highL'], edgecolor='black', label='By infection'),
               mpl.patches.Patch(facecolor=colors['high'], edgecolor='black', label='By vaccination')]
    ax5.legend(handles=handles, bbox_to_anchor=(0.1,0.9), ncol=1, frameon=False)

    fig.align_ylabels()

    if not silent: plt.show()
    if path!=None: fig.savefig(path)
        
        

        
def motivation(figure, path=None, silent=False, arial=False):
    set_rcParams(arial=arial)
    #mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.labelsize"] = 8

    cm=1/2.54
    fig = plt.figure(figsize=(18*cm, 10.5*cm), constrained_layout=True)
    grid = fig.add_gridspec(ncols=3, nrows=2, wspace=0.2, hspace=0.2)


    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])
    ax3 = fig.add_subplot(grid[2])
    ax4 = fig.add_subplot(grid[3])
    ax5 = fig.add_subplot(grid[4])
    ax6 = fig.add_subplot(grid[5])
    

    def plot_contact_feedback(ax):
        kmin = 0.3
        kmax = 0.8
        ICUcap = 35
        xmax = 60
        ax.plot([0,ICUcap],[kmax,kmin], color='black')
        ax.plot([ICUcap,xmax],[kmin,kmin], color='black')
        ax.set_ylim(0,1)
        ax.set_xlabel("ICU occupancy\nper million")
        ax.set_ylabel("level of contagious\ncontacts")
        ax.set_yticks([0,kmin,kmax,1])
        ax.set_yticklabels(['0','$k_{min}$','$k_{max}$','1'])
        ax.axhline(kmin, ls=':', color='grey', zorder=0)
        ax.axhline(kmax, ls=':', color='grey', zorder=0)
        ax.text(5, kmin-0.1, 'due to current NPIs', size=7, color='grey')
        ax.text(5, kmax+0.05, 'allowed by current NPIs', size=7, color='grey')



    def plot_willingness(ax):
        ICU = np.linspace(0,100,100)
        ubase = 0.5
        umax = 0.9
        alpha = 0.02
        scen1 = 15
        scen2 = 65
        def u_w(ICU): return ubase + (umax-ubase)*(1-np.exp(-alpha*ICU))
        ax.plot(ICU, u_w(ICU), color='black')
        ax.scatter(scen1,u_w(scen1), marker='d', color=figure['cVac1'], zorder=2.5)
        ax.scatter(scen2,u_w(scen2), marker='d', color=figure['cVac2'], zorder=2.5)
        ax.hlines(u_w(scen1), 0, scen1, ls='--', color=figure['cVac1'], zorder=0)
        ax.hlines(u_w(scen2), 0, scen2, ls='--', color=figure['cVac2'], zorder=0)
        ax.axhline(ubase, ls=':', color='grey', zorder=0)
        ax.axhline(umax, ls=':', color='grey', zorder=0)
        ax.text(10, ubase-0.1, 'Base acceptance', size=7, color='grey')
        ax.text(10, umax+0.05, 'Maximal acceptance', size=7, color='grey')
        ax.set_xlabel("ICU occupancy\nper million")
        ax.set_ylabel("Fraction of the population\nwanting to be vaccinated")
        ax.set_ylim(0,1)
        ax.set_yticks([0,.5,1])
#        ax.set_yticks([ubase, umax])
#        ax.set_yticklabels(['$u_{base}$', '$u_{max}$'])
        return None

    def plot_vaccination(ax):
        ubase = 0.5
        umax = 0.9
        alpha = 0.02
        time_u = 1
        epsilon_u = 0.01
        scen1 = 15
        scen2 = 65
        def u_w(ICU): return ubase + (umax-ubase)*(1-np.exp(-alpha*ICU))
        def softplus(slope, base, threshold, epsilon):
            return lambda x: slope*epsilon*np.log(np.exp(1/epsilon*(threshold-x))+1) + base
        def phi(ICU, UC, frac=1): return softplus(frac/time_u, 0, u_w(ICU), epsilon_u)(UC)
        x = np.linspace(0,1,100)
        ax.plot(x,phi(scen1, x), color=figure['cVac1'])
        ax.plot(x,phi(scen2, x), color=figure['cVac2'])
        ax.set_ylim(0,1)
        ax.set_yticks([0,.5,1])
        ax.scatter(u_w(scen1),0.6, marker='d', color=figure['cVac1'], zorder=10, clip_on=False)
        ax.scatter(u_w(scen2),0.6, marker='d', color=figure['cVac2'], zorder=10, clip_on=False)
        ax.vlines(u_w(scen1), 0, 0.5, ls='--', color=figure['cVac1'])
        ax.vlines(u_w(scen2), 0, 0.5, ls='--', color=figure['cVac2'])
        ax.axvline(ubase, ymax=0.65, ls=':', color='gray', zorder=0)
        ax.text(0.4,0.7,'minimal\nvaccine\nuptake', size=7, color='gray', transform=ax.transAxes)
        ax.axvline(umax, ymax=0.65, ls=':', color='gray', zorder=0)
        ax.text(0.76,0.7,'maximal\nvaccine\nuptake', size=7, color='gray', transform=ax.transAxes)
        ax.set_xlabel("Fraction of population\nalready vaccinated")
        ax.set_ylabel("Fraction of population\nseeking first vaccination")

    
    
    def plot_cosmovsICU(ax):
        ax.scatter(cosmodata.cosmotimelineICU[:24], 6-np.array(cosmodata.avggroup[:24]), color='red', alpha=0.5, s=8, label='Survey data')
        ax.scatter(cosmodata.cosmotimelineICU[24:], 6-np.array(cosmodata.avggroup[24:]), color='orange', alpha=0.5, s=8)
        ax.set_ylabel("Avoiding private parties\n5=never, 1=always")
        ax.set_xlabel("ICU occupancy\nper million")
        # CURVEFIT 
        ICUcap = 37
        epsilon = 1 
        def f(ICU, saturation, slope):
            return saturation - slope*epsilon*np.log(np.e**(1/epsilon*(ICUcap-ICU))+1)
        def g(ICU, sat, alpha, yaxis):
            return sat + (yaxis-sat)*np.exp(-alpha*ICU)

        popt1, pcov1 = curve_fit(f, cosmodata.cosmotimelineICU, cosmodata.avggroup)
        print('Parameters:',popt1)
        timeline = np.linspace(0,np.max(cosmodata.cosmotimelineICU), 100)
        ax.plot(timeline, 6-f(timeline, popt1[0], popt1[1]), label='Curve Fit', color=figure['cFit'])
        
        
        
        popt, pcov = curve_fit(f, cosmodata.cosmotimelineICU[24:], cosmodata.avggroup[24:])
        ax.plot(timeline, 6-f(timeline, popt[0], popt[1]), color='orange')
        popt, pcov = curve_fit(f, cosmodata.cosmotimelineICU[:24], cosmodata.avggroup[:24])
        ax.plot(timeline, 6-f(timeline, popt[0], popt[1]), color='red')
        
        xarray=[]
        yarray=[]
        for i, cosmo in enumerate(cosmodata.cosmotimelineICU):
            xarray.append(cosmodata.avggroup[i])
            yarray.append(f(cosmo, popt1[0], popt1[1]))
            
        corr3,_ = pearsonr(xarray, yarray)
        print('Softplus Correlation:', corr3)
        #ax.scatter(xarray,yarray,s=5)
        
        ax.set_ylim(1,None)
        #ax.legend(loc='upper right')
        return None 
    
    def plot_cosmovsICU_exp(ax):
        ax.scatter(cosmodata.cosmotimelineICU[:24], 6-np.array(cosmodata.avggroup[:24]), color='red', alpha=0.5, s=8, label='Survey data')
        ax.scatter(cosmodata.cosmotimelineICU[24:], 6-np.array(cosmodata.avggroup[24:]), color='orange', alpha=0.5, s=8)
        ax.set_ylabel("Avoiding private parties\n5=never, 1=always")
        ax.set_xlabel("ICU occupancy\nper million")
        # CURVEFIT 

        def g(ICU, sat, alpha, yaxis):
            return sat + (yaxis-sat)*np.exp(-alpha*ICU)

        popt1, pcov1 = curve_fit(g, cosmodata.cosmotimelineICU, cosmodata.avggroup)
        #print('Parameters:',popt1)
        timeline = np.linspace(0,np.max(cosmodata.cosmotimelineICU), 100)
        ax.plot(timeline, 6-g(timeline, popt1[0], popt1[1], popt1[2]), label='Curve Fit', color=figure['cFit'])
        
        xarray=[]
        yarray=[]
        for i, cosmo in enumerate(cosmodata.cosmotimelineICU):
            xarray.append(cosmodata.avggroup[i])
            yarray.append(g(cosmo, popt1[0], popt1[1], popt1[2]))
            
        corr3,_ = pearsonr(xarray, yarray)
        #print('Exponential Correlation:', corr3)
        
        popt, pcov = curve_fit(g, cosmodata.cosmotimelineICU[24:], cosmodata.avggroup[24:])
        ax.plot(timeline, 6-g(timeline, popt[0], popt[1], popt1[2]), color='orange')
        popt, pcov = curve_fit(g, cosmodata.cosmotimelineICU[:24], cosmodata.avggroup[:24])
        ax.plot(timeline, 6-g(timeline, popt[0], popt[1], popt1[2]), color='red')
        
        
        #ax.scatter(xarray,yarray,s=5)
        
        ax.set_ylim(1,None)
        #ax.legend(loc='upper right')
        return None 
    
    def plot_cosmoStringency(ax):
        ax.set_xlabel('2020         2021')
        ax.set_xticks([cosmodata.datesdict['2020-04-15'], cosmodata.datesdict['2020-10-15'], 
                       cosmodata.datesdict['2021-04-15'], cosmodata.datesdict['2021-10-15']])
        ax.set_xticklabels(['Apr.', 'Oct.', 'Apr.', 'Oct.'])      
        ax.plot(cosmodata.t,cosmodata.NPItime, color=figure['cStringency'], label='NPI stringency', zorder=5)
        ax.set_ylabel("1 = strict, always\n0 = mild, never")
        ax.set_ylim(0.45,1)
        ax.scatter(cosmodata.cosmotimeline,cosmodata.avggroup, label='Contact reduction',color=figure['cCosmodata'], alpha=0.5, s=8)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.3), ncol=1)       
        ax.axvspan(cosmodata.datesdict['2020-10-15'], cosmodata.datesdict['2020-12-15'], ymin=0, ymax=1, color='gray', alpha=0.2)        
        return None 
    
    def plot_cosmoICU(ax):
        ax.set_xlabel('2020         2021')
        ax.set_xticks([cosmodata.datesdict['2020-04-15'], cosmodata.datesdict['2020-10-15'], 
                       cosmodata.datesdict['2021-04-15'], cosmodata.datesdict['2021-10-15']])
        ax.set_xticklabels(['Apr.', 'Oct.', 'Apr.', 'Oct.'])      
        ax.plot(cosmodata.t,np.array(cosmodata.ICUtime), color=figure['cICU'], label='ICU', zorder=5)
        ax.set_ylabel("ICU occupancy per million")
        ax.set_ylim(0,None)
        ax2 = ax.twinx()
        ax2.scatter(cosmodata.cosmotimeline[:24],cosmodata.avggroup[:24], label='2021',color='red', alpha=0.5, s=8)
        ax2.scatter(cosmodata.cosmotimeline[24:],cosmodata.avggroup[24:], label='2020',color='orange', alpha=0.5, s=8)
        ax2.set_ylabel("Avoiding private parties\n1 = never, 5= always")
        ax2.set_ylim(3,None)
        ax2.spines['right'].set_visible(True)
        ax2.vlines(cosmodata.datesdict['2021-01-01'], 0,1, color='gray', ls=':')
        #Stringecy:
        #ax2.plot(cosmodata.t,cosmodata.NPItime, color=figure['cStringency'], label='NPI stringency', zorder=5)
             
        return None 
    
    def plot_romania(ax):
        startdate='2021-03-15'
        enddate='2021-11-01'
        startpoint = cosmodata.ROU_datesdict[startdate]
        endpoint = cosmodata.ROU_datesdict[enddate]        
        ax.set_xlabel('2021')
        ax.set_xticks([cosmodata.ROU_datesdict['2021-04-15'], cosmodata.ROU_datesdict['2021-07-15'], 
                       cosmodata.ROU_datesdict['2021-10-15']])
        ax.set_xticklabels(['Apr.', 'July', 'Oct.'])        
        lns1 = ax.plot(cosmodata.ROU_t[startpoint:endpoint], cosmodata.ROU_ICUtime[startpoint:endpoint], label='ICU', color=figure['cICU'])
        ax.set_ylabel('ICU occupancy per million')
        ax2 = ax.twinx()
        lns2 = ax2.plot(cosmodata.ROU_t[startpoint:endpoint], np.array(cosmodata.ROU_vaccinetime)[startpoint:endpoint], color=figure['cVaccines'], label='Vaccines')
        ax2.set_ylabel('Daily vaccines per million')
        ax2.spines['right'].set_visible(True)

        ax.set_ylim(0,None)
        ax2.set_ylim(0,None)
        
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        #ax.legend(lns, labs, loc='upper left')
        return None 
    
    #----------------------------------------- Plotting ------------------------------------------------
    
    translation = {
        'CosmoStringency': plot_cosmoStringency,
        'CosmoICU': plot_cosmoICU,
        'Romania': plot_romania,
        'CosmovsICU': plot_cosmovsICU,
        'CosmovsICU_exp': plot_cosmovsICU_exp,
        'Willingness': plot_willingness,
        'Vaccination': plot_vaccination,
        'ContactFeedback': plot_contact_feedback,
    }
    
    
    plotting_dict ={
    ax1: translation[figure['ax1']],
    ax2: translation[figure['ax2']],
    ax3: translation[figure['ax3']],
    ax4: translation[figure['ax4']],
    ax5: translation[figure['ax5']],
    ax6: translation[figure['ax6']],
    }
    
    for ax in [ax1,ax2, ax3, ax4, ax5, ax6]:
        plotting_dict[ax](ax)
    
    for ax, label in zip([ax1,ax2,ax3,ax4,ax5,ax6], ['a','b','c','d','e','f']):
        ax.text(-.12,1.1,label, size=12, weight='bold', color='black', transform=ax.transAxes)

    fig.align_ylabels()

    if not silent: plt.show()
    if path!=None: fig.savefig(path)
    
        
#-------------------------------------------------------------------------------------------------------------------        
      
    
def gammakernels(figure, path=None, silent=False, arial=False):
    set_rcParams(arial=arial)
    #mpl.rcParams["axes.spines.right"] = False
    
    cm=1/2.54
    fig = plt.figure(figsize=(18*cm, 4.5*cm), constrained_layout=True)

    
    grid = fig.add_gridspec(ncols=4, nrows=1, wspace=0.1)


    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])
    ax3 = fig.add_subplot(grid[2])
    ax4 = fig.add_subplot(grid[3])
    
    def plot_cosmoStringency(ax):
        ax.set_xlabel('2020         2021')
        ax.set_xticks([cosmodata.datesdict['2020-04-15'], cosmodata.datesdict['2020-10-15'], 
                       cosmodata.datesdict['2021-04-15'], cosmodata.datesdict['2021-10-15']])
        ax.set_xticklabels(['Apr.', 'Oct.', 'Apr.', 'Oct.'])      
        ax.plot(cosmodata.t,cosmodata.NPItime, color=figure['cStringency'], label='NPI stringency', zorder=5)
        ax.set_ylabel("1 = strict, always\n0 = mild, never")
        ax.set_ylim(0.45,1)
        ax.scatter(cosmodata.cosmotimeline,cosmodata.avggroup, label='Contact reduction',color=figure['cCosmodata'], alpha=0.5, s=8)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.3), ncol=1)       
        ax.axvspan(cosmodata.datesdict['2020-10-15'], cosmodata.datesdict['2020-12-15'], ymin=0, ymax=1, color='gray', alpha=0.2)        
        return None 
    
    def plot_matrix(ax):
        matrix = np.flip(parameters.calc_Cs()[1], axis=0)
        ax.imshow(matrix, origin='lower')
        ax.figure.colorbar(ax.imshow(matrix), ax=ax, cmap="YlGn")
        
        agegroups = ["0-19", "20-39", "40-59", "60-69","70-79", "80+"]
        ax.set_xticks(np.arange(6))
        ax.set_yticks(np.arange(6))
        ax.set_xticklabels(agegroups)
        ax.set_yticklabels(np.flip(agegroups))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        ax.set_title('School contacts')
        return None
    
    def plot_reduction(ax):
        data = pd.read_csv('../parameters/scenariosDefinition.csv', sep=';', header=0)
        Hmax = 35

        highs= 'schoollow'
        lows = 'schoolhigh'
        hline1 = np.linspace(0,Hmax,100)
        hline2 = np.linspace(Hmax,1.5*Hmax,100)
        
        for i, c in zip([0,2,4], ['c1','c2','c3']):
            ax.plot(hline1, (data[lows][i]-(data[lows][i]-data[highs][i])/Hmax*hline1), color=figure[c], label=f'Scen.{i+1}')
            ax.plot(hline2, (data[highs][i]*np.ones(len(hline2))), color=figure[c])
        ax.set_title('Contact reduction\nin schools')
        ax.set_xlabel('ICU occupancy per million')
        ax.set_ylabel('0=strong reduction\n1=no reduction') 
        ax.set_ylim(0,1.05)
        ax.legend()
        return None 
    
    def plot_gamma_R(ax):
        a = parameters.params_base['a_Rt']
        b = parameters.params_base['b_Rt']
        gamma_cutoff= round(parameters.params_base['gamma_cutoff'])        
        gtimes = np.arange(-gamma_cutoff, 0, 0.01)
        g = b**a * (-gtimes)**(a-1) * np.exp(-b*(-gtimes)) / gamma_func(a)
        ax.plot(gtimes, g, color=figure['cKernelR'], label='$G_R$')
        ax.set_xlim(-25,0)
        ax.set_xlabel('Past days')
        ax.set_ylabel('Prob. density')
        ax.legend()
        return None
    
    def plot_gamma_vac(ax):
        a = parameters.params_base['a_vac']
        b = parameters.params_base['b_vac'] 
        tau1 = round(parameters.params_base['tau_vac1'])
        tau2 = round(parameters.params_base['tau_vac2'])
        gamma_cutoff= round(parameters.params_base['gamma_cutoff']+np.max([tau1,tau2]))        
        gtimes = np.arange(-gamma_cutoff, 0, 0.01)
        print(gamma_cutoff)
        gu = b**a * (-gtimes)**(a-1) * np.exp(-b*(-gtimes)) / gamma_func(a) 
        gw = b**a * (-gtimes)**(a-1) * np.exp(-b*(-gtimes)) / gamma_func(a) 
        ax.plot(gtimes-tau1, gu, color=figure['cKernelu'], label='$G_u$')
        ax.plot(gtimes-tau2, gw, color=figure['cKernelw'], label='$G_w$')
        ax.set_xlim(-100,0)
        ax.set_xlabel('Past days')
        ax.set_ylabel('Prob. density')
        ax.legend()
             
        return None 
    
    def plot_kernel_R(ax):
        t1 = cosmodata.datesdict['2020-10-15']
        t2 = cosmodata.datesdict['2020-12-15']
        times = np.linspace(t1, t2, t2-t1)        
        a = parameters.params_base['a_Rt']
        b = parameters.params_base['b_Rt']        
        gamma_cutoff= round(parameters.params_base['gamma_cutoff'])        
        gtimes = np.arange(-gamma_cutoff, 0, 1)
        g = b**a * (-gtimes)**(a-1) * np.exp(-b*(-gtimes)) / gamma_func(a)        
        convolution = []
        for i, t in enumerate(times):
            convolution.append((g*cosmodata.ICUtime[t1+i-gamma_cutoff:t1+i]).sum())
            
        ax.plot(times, cosmodata.ICUtime[t1:t2], label='ICU', color=figure['cICU'])
        ax.plot(times, convolution, label='$H_R$', color=figure['cKernelR'])
        
        ax.set_ylabel('ICU occupancy\nper million')
        ax.set_xlabel('2020')        
        ax.set_xticks([cosmodata.datesdict['2020-11-01'], cosmodata.datesdict['2020-12-01']])
        ax.set_xticklabels(['1.Nov.', '1.Dec.']) 
        ax.legend()
        return None
    
    def plot_kernel_vac(ax):
        t1 = cosmodata.datesdict['2020-10-15']
        t2 = cosmodata.datesdict['2021-02-15']
        times = np.linspace(t1, t2, t2-t1)        
        a = parameters.params_base['a_vac']
        b = parameters.params_base['b_vac'] 
        tau1 = round(parameters.params_base['tau_vac1'])
        tau2 = round(parameters.params_base['tau_vac2'])
        gamma_cutoff= round(parameters.params_base['gamma_cutoff'])        
        gtimes = np.arange(-gamma_cutoff, 0, 1)
        g = b**a * (-gtimes)**(a-1) * np.exp(-b*(-gtimes)) / gamma_func(a)        
        convolution_u = []
        convolution_w = []
        for i, t in enumerate(times):
            convolution_u.append((g*cosmodata.ICUtime[t1+i-gamma_cutoff-tau1:t1+i-tau1]).sum())
            convolution_w.append((g*cosmodata.ICUtime[t1+i-gamma_cutoff-tau2:t1+i-tau2]).sum())
            
        ax.plot(times, cosmodata.ICUtime[t1:t2], label='ICU', color=figure['cICU'])
        ax.plot(times, convolution_u, label='$H_u$', color=figure['cKernelu'])
        ax.plot(times, convolution_w, label='$H_w$', color=figure['cKernelw'])
        
        ax.set_ylabel('ICU occupancy\nper million')
        ax.set_xlabel('2020    2021')        
        ax.set_xticks([cosmodata.datesdict['2020-11-01'], cosmodata.datesdict['2021-02-01']])
        ax.set_xticklabels(['1.Nov.', '1.Feb.']) 
        ax.legend()
        return None 
    
    
    translation = {
        'CosmoStringency': plot_cosmoStringency,
        'Matrix': plot_matrix,
        'Reduction': plot_reduction,
        'Gamma R': plot_gamma_R,
        'Gamma Vac': plot_gamma_vac,
        'Kernel R': plot_kernel_R,
        'Kernel Vac': plot_kernel_vac,
    }
    
    
    plotting_dict ={
    ax1: translation[figure['ax1']],
    ax2: translation[figure['ax2']],
    ax3: translation[figure['ax3']],
    ax4: translation[figure['ax4']],
    }
    
    for ax in [ax1,ax2, ax3, ax4]:   
        plotting_dict[ax](ax)
    
    for ax, label in zip([ax1,ax2,ax3,ax4], ['a','b','c','d']):
        ax.text(-.12,1.1,label, size=12, weight='bold', color='black', transform=ax.transAxes)

    #fig.align_ylabels()

    if not silent: plt.show()
    if path!=None: fig.savefig(path)

        
        
def sixpanels_flexible(models, figure, path=None, silent=False, arial=False, ICUcap=None, full_wave=None):
    set_rcParams(arial=arial)
    mpl.rcParams["legend.fontsize"] = 7

    m1, m2, m3 = models
    t = m1.times
    data = [m1.chopped_data().sum(axis=2), m2.chopped_data().sum(axis=2), m3.chopped_data().sum(axis=2)]
    AGdata = [m1.chopped_data(), m2.chopped_data(), m3.chopped_data()]
    
    ags = AGdata[0].shape[2]
    colors = mpl.cm.viridis_r(np.linspace(0.,1.,ags))

    cm = 1/2.54
    fig = plt.figure(figsize=(18*cm, 10.5*cm), constrained_layout=True)
    grid = fig.add_gridspec(ncols=3, nrows=2, wspace=0.1)

    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])
    ax3 = fig.add_subplot(grid[2])
    ax4 = fig.add_subplot(grid[3])
    ax5 = fig.add_subplot(grid[4])
    ax6 = fig.add_subplot(grid[5])

    colors = {
        'low':'#0099B4FF', 'mid':'#00468BFF', 'high':'#1B1919FF', 'free':'#1B1919FF',
        'lowL':'#FFFFFFFF', 'midL':'#FFFFFFFF', 'highL':'#FFFFFFFF',
#        'lowL':'#0099B499', 'midL':'#00468B99', 'highL':'#1B191999',
        'line':'#ADB6B6FF', 'ICUcap':'red',
        'now':'#ADB6B6FF', 'nowL':'#FFFFFFFF',
#        'now':'#93dfedFF', 'nowL':'#93dfed99',
        'FW':'#ED0000FF',
    }
    
    ccountry= {
        '60': 'darkred',
        '70': 'red',
        '80': 'orange',
        '85': 'yellow',
    }
    
    if figure['type'] == 'countries':
        main_colors = [ccountry[figure['Scens'][i]] for i in [0,1,2]]
    
    else: 
        main_colors = [figure['c1'], figure['c2'], figure['c3']]
    #main_colors = [colors['high'],colors['mid'],colors['low']]
    main_colors_L = [colors['highL'],colors['midL'],colors['lowL']]


    def Rt(m,t):
        return max(np.linalg.eigvals((np.moveaxis(m.Cs,0,2) * m.k_lowH(t)).sum(axis=2)))
    
    def plot_axes_winter(ax):
        ax.axvline(180, ls=':', color=colors['line'], zorder=-4)
        ax.set_xlabel('2021           2022')
        ax.set_ylim(0,None)
        ax.set_xticks([45, 135, 45+2*90, 45+3*90])
        ax.set_xticklabels(['Oct.','Jan.','Apr.','July'])
        return None
        
    def plot_NPI(ax):
        for i, m in enumerate([m1,m2,m3]):
            ax.plot(t, list(map(Rt, [m]*len(t), t)), color=main_colors[i], zorder=-i)
        plot_axes_winter(ax)
        ax.set_ylabel("Contact levels\ninfluenced by NPIs")
        ax.text(0.54,0.15,'Lifting of\nrestrictions', size=7, color=colors['line'], transform=ax.transAxes)
        for i, pos in enumerate([0.9,0.75,0.6]):
            s = figure['Scens'][i]
            if figure['type'] == 'countries':
                ax.text(200,pos, f'Ex.{s}', size=7, color=main_colors[i])
            else: 
                ax.text(200,pos, f'Scenario{s}', size=7, color=main_colors[i])
        return None
    
    def plot_incidence(ax):
        for i, m in enumerate([m1,m2,m3]):
            ax.plot(t, m.rho*(data[i][:,4]+data[i][:,5]+data[i][:,6]), color=main_colors[i])
        ax.set_ylabel("Daily new cases\nper million")
        plot_axes_winter(ax)
        if full_wave != None:
            mfull = full_wave
            d = mfull.chopped_data().sum(axis=2)
            ax.plot(t, mfull.rho*(d[:,4]+d[:,5]+d[:,6]), color=colors['FW'], ls=':')
        return None
    
    def plot_ICU(ax):
        for i, m in enumerate([m1,m2,m3]):
            ax.plot(t, data[i][:,10]+data[i][:,11], color=main_colors[i])
        ax.set_ylabel("ICU occupancy\nper million")
        plot_axes_winter(ax)
        if full_wave != None:
            mfull = full_wave
            d = mfull.chopped_data().sum(axis=2)
            ax.plot(t, d[:,10]+d[:,11], color=colors['FW'], ls=':')
            
        if ICUcap != None:
            maxicu = 0
            for i, m in enumerate([m1,m2,m3]):
                if np.max((data[i][:,10]+data[i][:,11])) >= maxicu:
                    maxicu = np.max(data[i][:,10]+data[i][:,11])
            if maxicu >= ICUcap-5:
                ylim = ax.get_ylim()
                ax.hlines(ICUcap,0, 0.95*360, color='black', ls='--', zorder=-1)
                #ax.axhspan(ICUcap-1,ICUcap+1, alpha=0.8, xmax=0.9, facecolor=colors['ICUcap'], edgecolor=None, zorder=-1)
                ax.text(1.0,ICUcap/ylim[1]-0.2,'Estimated\nICU capacity', size=7, color='black', rotation=-90, transform=ax3.transAxes)
                ax.scatter(370,ICUcap+1, marker="<", color='red')
        return None
    
    def plot_vaccinations(ax):
        for i,m in enumerate([m1,m2,m3]):
            phis = np.array(list(map(m.get_phis, t, AGdata[i]))).sum(axis=(2,3))
            shift = round(m.tau_vac1/m.step_size)
            d1a = np.roll(phis[:,0], -shift)
            d1a[-shift:] = 0
            shift = round(m.tau_vac1/2./m.step_size)
            d1b = np.roll(phis[:,0], -shift)
            d1b[-shift:] = 0
            shift = round(m.tau_vac2/m.step_size)
            d2 = np.roll(phis[:,1], -shift)
            d2[-shift:] = 0
            ax.plot(t, d1a+d1b+d2, color=main_colors[i])
        ax.set_ylabel("Daily vaccinations\nper million")
        plot_axes_winter(ax)
        return None 
    
    def plot_bar_immunity(ax):
        ax.bar(1, data[0][0,1]/1e6, 0.5, align='center', color=colors['now'], edgecolor='black', zorder=-1)
        ax.bar(1, (data[0][0,12]+data[0][0,13])/1e6, 0.5, align='center', 
               color=colors['nowL'], edgecolor='black', zorder=-1, bottom=data[0][0,1]/1e6)
        offset = 0.5
        for i in [2,4]:
            for ab,m,j in zip([-0.5,0,0.5],[m1,m2,m3],[0,1,2]):
                ax.bar(offset+i+ab, data[j][900*i-1,1]/1e6, 0.5,  
                    align='center', color=main_colors[j], edgecolor='black', zorder=-1)
                ax.bar(offset+i+ab, (data[j][900*i-1,12]+data[j][900*i-1,13])/1e6, 0.5,
                    align='center', color=main_colors_L[j], edgecolor='black', zorder=-1,
                        bottom=data[j][900*i-1,1]/1e6)
        ax.set_ylim(0,1)
        ax.set_ylabel("Immune fraction\nof the population")
        ax_x = [1,offset+2,offset+4]
        axlabels=['Initial','After\nwinter', 'After\none year']
        ax.set_xticks(ax_x)
        ax.set_xticklabels(axlabels)
        handles = [mpl.patches.Patch(facecolor=colors['highL'], edgecolor='black', label='By infection'),
               mpl.patches.Patch(facecolor=colors['high'], edgecolor='black', label='By vaccination')]
        
        #ax.legend(handles=handles, loc='upper right', ncol=1, frameon=False)
        return None 
    
    def plot_bar_deaths(ax):
        offset = 0.5
        for i in [2,4]:
            for ab,m,j in zip([-0.5,0,0.5],[m1,m2,m3],[0,1,2]):
                ax.bar(offset+i+ab, data[j][900*i-1,16], 0.5, 
                    align='center', color=main_colors[j], edgecolor='black', zorder=-3)    
        ax.set_ylim(0,None)
        ax.set_ylabel("Total deaths\nper million")
        ax_x = [offset+2,offset+4]
        axlabels=['After\nwinter', 'After\none year']
        ax.set_xticks(ax_x)
        ax.set_xticklabels(axlabels)
        return None
    
    def plot_Rt_eigenvalue(ax):
        for i, m in enumerate([m1,m2,m3]):
            #list(map(Rt, [m]*len(t), t))
            ax.plot(t, list(map(m.Rt,t)), color=main_colors[i], zorder=-i)
        plot_axes_winter(ax)
        ax.set_ylabel("Driving force\nof infections")
        return None 
    
    def plot_Rt_observed2(ax):
        delay = round(4/m1.step_size)
        offset = round(5/m1.step_size)
        for i, m in enumerate([m1,m2,m3]):
            ax.plot(t[offset+delay:], (data[i][offset+delay:,4]+data[i][offset+delay:,5]+data[i][offset+delay:,6])/(data[i][offset:-delay,4]+data[i][offset:-delay,5]+data[i][offset:-delay,6]), color=main_colors[i])
        ax.axvline(180, ls=':', color=colors['line'], zorder=-4)
        ax.set_xlabel('2021           2022')
        ax.set_xticks([45, 135, 45+2*90, 45+3*90])
        ax.set_xticklabels(['Oct.','Jan.','Apr.','July'])
        ax.set_ylabel("Observed\nreproduction number")
        return None 
    
    def plot_Rt_observed(ax):
        delay = round(4/m1.step_size)
        offset = round(5/m1.step_size)
        for i, m in enumerate([m1,m2,m3]):
            cases = (data[i][:,4]+data[i][:,5]+data[i][:,6])
            Robs = (cases / np.roll(cases, delay))[delay:]
            ax.plot(t[delay:],Robs, color=main_colors[i])
        ax.axvline(180, ls=':', color=colors['line'], zorder=-4)
        ax.set_xlabel('2021           2022')
        ax.set_xticks([45, 135, 45+2*90, 45+3*90])
        ax.set_xticklabels(['Oct.','Jan.','Apr.','July'])
        ax.set_ylabel("Observed\nreproduction number")
    
    def plot_bar_patients(ax):
        offset = 0.5
        for i in [2,4]:
            for ab,m,j in zip([-0.5,0,0.5],[m1,m2,m3],[0,1,2]):
                ax.bar(offset+i+ab, data[j][:900*i-1,10].sum()+data[j][:900*i-1,11].sum(), 0.5, 
                    align='center', color=main_colors[j], edgecolor='black', zorder=-3)    
        ax.set_ylim(0,None)
        ax.set_ylabel("Total ICU patients\nper million")
        ax_x = [offset+2,offset+4]
        axlabels=['After\nwinter', 'After\none year']
        ax.set_xticks(ax_x)
        ax.set_xticklabels(axlabels)
        return None
        
        
    
    
    # -----------------------------------------Plotting ----------------------------------------------
    
    translation = {
        'NPI': plot_NPI,
        'ICU': plot_ICU,
        'Incidence': plot_incidence,
        'Vaccines': plot_vaccinations,
        'Immunity': plot_bar_immunity,
        'Deaths': plot_bar_deaths,
        'Rt_EV': plot_Rt_eigenvalue,
        'Rt_OBS': plot_Rt_observed,
        'Rt_OBS2': plot_Rt_observed2,
        'Patients': plot_bar_patients
    }
    
    
    plotting_dict ={
    ax1: translation[figure['ax1']],
    ax2: translation[figure['ax2']],
    ax3: translation[figure['ax3']],
    ax4: translation[figure['ax4']],
    ax5: translation[figure['ax5']],
    ax6: translation[figure['ax6']]
    }
    
    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:   
        plotting_dict[ax](ax)


    
    # --------------------------------------- General Axis Things -----------------------------------
    

    for ax, label in zip([ax1,ax2,ax3,ax4,ax5,ax6], ['a','b','c','d','e','f']):
        ax.text(-.12,1.1,label, size=12, weight='bold', color='black', transform=ax.transAxes)
        
        

    fig.align_ylabels()

    if not silent: plt.show()
    if path!=None: fig.savefig(path)

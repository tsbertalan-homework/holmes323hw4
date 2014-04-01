'''
Created on Mar 31, 2014

@author: bertalan@princeton.edu
'''
from __future__ import division  # default to float division

import numpy as np

import kevrekidis as kv

from tomSims import logging, setLogLevel


def NLIFmodel(Iapp=0.0, EL=10.6, taum=3.3, s=0, vreset=0, vthresh=30):
    def nLIF(X, t):
        """4.2.5"""
        v = X[0]
        gL = 1 / taum
        vss = EL + Iapp / gL
        drift = (vss - v) / taum
        diff  = s / np.sqrt(taum)
        return drift, diff
    def nLIFReset(X):
        v = X[0]
        resetFlag = False
        if v > vthresh:
            v = vreset
            X[0] = v
            resetFlag = True
        return X, resetFlag
    return nLIF, nLIFReset

def eulerMaruyama(driftDiff, X0, tmax, Dt, tmin=0, resetRule=lambda X: (X, False)):
    nstep = int((tmax - tmin) / Dt)
    X0 = np.array(X0)
    X = np.empty((nstep, X0.size))
    T = np.arange(tmin, tmax, Dt)
    resetTimes = []
    X[0] = X0
    assert tmin < tmax
    
    i = 1
    while i < nstep:
        drift, diff = driftDiff(X[i - 1], T[i - 1])
        DW = np.random.normal()
        X[i] = X[i - 1] + Dt * drift + DW * diff
        resetted, resetFlag = resetRule(X[i])
        if resetFlag:
            resetTimes.append(T[i])
        i += 1
    return X, T, resetTimes

def spikeTimes2Freq(resetTimes):
    TISI = np.asarray(resetTimes[1:]) - np.asarray(resetTimes[:-1])
    return (1 / TISI).mean()

def traces(Iapp=10):
    figure, axis = kv.fa(figsize=(16, 9))
    for s in 0, .1, .5:  # , 1, 5, 10:
        driftDiff, reset = NLIFmodel(Iapp=Iapp, s=s)
        X, T, resetTimes = eulerMaruyama(driftDiff, X0=[0], tmax=50, Dt=0.05, resetRule=reset)
        f = spikeTimes2Freq(resetTimes)
        axis.plot(T, X, label=r"$s=%.1f$, $\langle 1/T_{ISI} \rangle =%.2f$" % (s, f))
    axis.set_title(r'$I_{app}=%.1f$' % Iapp)
    axis.set_xlabel('$t$')
    axis.set_ylabel('$v$')
    axis.legend()
    
def Ifreq(nIvals=42, nReplicates=16):
    IappIapp = np.linspace(0, 10, nIvals)
    data = {
            'flists': {},
            'IappIapp': IappIapp,
            }
    svals = 0, .1, .5, 1, 5, 10
    for s in svals:
        logging.debug(" ")
        logging.info('s = %.1f' % s)
        ff = []
        for Iapp in IappIapp:
            logging.debug('Iapp = %.1f' % Iapp)
            driftDiff, reset = NLIFmodel(Iapp=Iapp, s=s)
            #We'll do each Iapp value nReplicates times, then average the f values.
            replicates = []
            while len(replicates) < nReplicates:
                X, T, resetTimes = eulerMaruyama(driftDiff, X0=[0], tmax=200, Dt=0.05, resetRule=reset)
                f = spikeTimes2Freq(resetTimes)
                replicates.append(f)
            ff.append(np.asarray(replicates).mean())
        data['flists'][s] = ff
    filename = 'NLIF-nI%d-nR%d-sv%d.p' % (nIvals, nReplicates, len(svals))
    from tomSims.saver import save
    save(filename, data)
    return filename

def plotData(filename):
    logging.debug('loading %s' % filename)
    from tomSims.saver import load
    data = load(filename)
    figure, axis = kv.fa(figsize=(16, 9))
    IappIapp = data['IappIapp']
    for s, ff in data['flists'].items():
        ff = np.asarray(ff)
        # replace all NaNs with 0 (occurs before the Hopf, when there is no spiking.)
        ff[np.isnan(ff)] = 0
#         for I in IappIapp.min(), IappIapp.max():
#             axis.axvline(I)
        axis.plot(IappIapp, ff, label="$s=%.1f$" % s)
        
    cv = comparison(IappIapp)
    axis.scatter(IappIapp, cv, label="from 2a")
    
    axis.set_xlabel(r'$I_{app}$')
    axis.set_ylabel(r'$\langle 1/T_{ISI} \rangle$')
    axis.legend()


def comparison(IappIapp, tauref=0, C=1, gL = 0.3, EL=10.6, vr=0, vth=30):
    Iapp = IappIapp
    vss = EL + Iapp/gL
    return 1/(tauref + C/gL * np.log((vss - vr) / (vss - vth)))



if __name__ == '__main__':
    setLogLevel("debug")
    
#     traces()
#     filename = Ifreq(nIvals=4, nReplicates=2)
    filename = Ifreq(nIvals=42, nReplicates=16)
#     filename = "NLIF-nI42-nR16-sv3.p"
#     filename = "NLIF-nI42-nR16-sv6.p"
    plotData(filename)
    kv.plotting.show()

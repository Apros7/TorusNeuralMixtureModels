import sys
sys.path.insert(0, '.')

import numpy as np
import scipy.linalg
from tqdm import tqdm
import os

from src.data.synthetic_data import TorusGraphInformation

#X = np.array([ [-2.1,-1.4,2,2.8,-0.7,-2], [1.3,-2.9,1.2,-2.9,1.7,-0.1], [-2.9,-2.5,-1.1,-0.4,1.9,-0.3] ])
#selMode = (True, True, True)
#num = {'nodes': 3, 'trials': 6, 'nodePairs': 3, 'param': 18}
#nodePairs = {'nodes': np.array([ [0,1], [0,2], [1,2] ])}

class TorusGraphInformation:
    samples: int # trials
    nodes: int
    nodePairs: np.ndarray[np.ndarray[int, int]]
    nNodePairs: int
    params: int
    fitFCM: bool    # FirstCircularMoments
    fitPAD: bool    # PairwiseAngleDifferences
    fitPAS: bool    # PairwiseAngleSums

class SM():
    def __init__(self, X, dm: TorusGraphInformation): # dm = TorusGraph datamodel with all information needed
        super(SM, self).__init__()
        self.X = X
        self.selMode = (dm.fitFCM, dm.fitPAD, dm.fitPAS)
        self.nodePairs = dm.nodePairs
        self.nNodePairs = dm.nNodePairs
        self.params = dm.params
        self.trials = dm.samples
        self.nodes = dm.nodes

    def run(self): return self.compPhiHatAndCovPhiHat()
        
    # Sufficient Statistics
    def compSufStat(self):
        X = self.X
        nodePairs = self.nodePairs
        selMode = self.selMode
        self.nodes

        # Sufficient Statistics
        Xi = X[nodePairs[:,0],:]
        Xj = X[nodePairs[:,1],:]
        H = []
        if selMode[0]:
            sC = np.cos(X)
            sS = np.sin(X)
            H.extend(sC)
            H.extend(sS)
        else:
            sC = None
            sS = None

        if selMode[1]:
            Xdif = Xi - Xj
            sAlpha = np.cos(Xdif)
            sBeta = np.sin(Xdif)
            H.extend(2*sAlpha)
            H.extend(2*sBeta)
        else:
            sAlpha = None
            sBeta = None

        if selMode[2]:
            Xsum = Xi + Xj
            sGamma = np.cos(Xsum)
            sDelta = np.sin(Xsum)
            H.extend(2*sGamma)
            H.extend(2*sDelta)
        else:
            sGamma = None
            sDelta = None

        H = np.array(H)

        return (H, sC, sS, sAlpha, sBeta, sGamma, sDelta)

    #Gamma (Jacobian of the sufficient statistics )
    def compGammaXt(self, t, sC, sS, sAlpha, sBeta, sGamma, sDelta ):
        selMode = self.selMode
        nodePairs = self.nodePairs
        nodes = self.nodes
        nNodePairs = self.nNodePairs

        # define gammaXt (t-th trial)
        Dt = [] # Jacobian of the suficient statistics

        if selMode[0]: #
            dC = -np.diag(sS[:,t])
            dS =  np.diag(sC[:,t])
            Dt.extend(dC)
            Dt.extend(dS)
        if selMode[1]:
            dAlpha = np.zeros((nNodePairs,nodes))
            dBeta =  np.zeros((nNodePairs,nodes))
        if selMode[2]:
            dGamma = np.zeros((nNodePairs,nodes))
            dDelta = np.zeros((nNodePairs,nodes))

        for r in range(nNodePairs):
            colInds = [ nodePairs[r,0], nodePairs[r,1] ]
            if selMode[1]:
                dAlpha[r, colInds] = np.array([-1,  1]) * sBeta[r, t]
                dBeta[r, colInds]  = np.array([ 1, -1]) * sAlpha[r, t]
            if selMode[2]:
                dGamma[r, colInds] = np.array([-1, -1]) * sDelta[r, t]
                dDelta[r, colInds] = np.array([ 1,  1]) * sGamma[r, t]
        if selMode[1]:
            Dt.extend(dAlpha)
            Dt.extend(dBeta)
        if selMode[2]:
            Dt.extend(dGamma)
            Dt.extend(dDelta)
        Dt = np.array(Dt)

        gammaXt = Dt @ Dt.T # np.matmul(Dt,Dt.T)

        return gammaXt

    #Phi hat and Covariance of Phi hat
    def compPhiHatAndCovPhiHat(self):
        param = self.params
        trials = self.trials

        # Sufficient Statistics
        H, sC, sS, sAlpha, sBeta, sGamma, sDelta = self.compSufStat()
        Ht = H.T

        # define compPhiHatAndCovPhiHat
        HhatX = np.mean(H,1) # H is (n_params, n_trials)

        # gammaHatX: average of gammaXt over trials
        gammaHatX = np.zeros((param, param))
        cache_gammaXt = True
        try:
            gammaXt_trials = np.zeros((trials,param,param))
        except MemoryError:
            cache_gammaXt = False
        for t in tqdm(range(trials), "Running Score Matching", disable=os.environ.get("DISABLE_TQDM", False)):
            gammaXt = self.compGammaXt(t, sC, sS, sAlpha, sBeta, sGamma, sDelta)
            if cache_gammaXt:
                gammaXt_trials[t, :, :] = gammaXt
            gammaHatX += gammaXt
        gammaHatX /= trials

        # Estimating phiHat
        L = scipy.linalg.cholesky(gammaHatX + 1e-8 * np.eye(gammaHatX.shape[0]), lower=True)
        phiHat = scipy.linalg.cho_solve((L, True), HhatX)


        # Estimating the covariance of phiHat: covPhiHat
        vHatX = np.zeros((param,param))
        for t in range(trials):
            if cache_gammaXt:

                vVec = np.dot(gammaXt_trials[t, :, :], phiHat) - Ht[t, :]
            else:
                gammaXt = self.compGammaXt(t,sC, sS, sAlpha, sBeta, sGamma, sDelta)
                vVec = np.dot(gammaXt, phiHat) - Ht[t, :]

            vHatX += np.outer(vVec, vVec)
        vHatX /= trials 

        Linv = np.linalg.inv(L)
        covPhiHat = 1/trials * scipy.linalg.cho_solve((L, True), vHatX) @ Linv.T @ Linv

        return (phiHat, covPhiHat)

import numpy as np
import scipy.linalg


class scoreMatching:
    def __init__(self):
        pass

    # Sufficient Statistics
    def compSufStat(self, X, selMode, nodePairs):

        # Sufficient Statistics
        Xi = X[nodePairs['nodes'][:,0],:]
        Xj = X[nodePairs['nodes'][:,1],:]
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

        print("H", H, "sC", sC, "sS", sS, "sAlpha", sAlpha, "sBeta", sBeta, "sGamma", sGamma, "sDelta", sDelta)

        return (H, sC, sS, sAlpha, sBeta, sGamma, sDelta)


        


    #Gamma (Jacobian of the sufficient statistics )
    def compGammaXt(self, t, num, selMode, nodePairs, 
                    sC, sS, sAlpha, sBeta, sGamma, sDelta ):

        # define gammaXt (t-th trial)
        Dt = [] # Jacobian of the suficient statistics

        if selMode[0]: #
            dC = -np.diag(sS[:,t])
            dS =  np.diag(sC[:,t])
            Dt.extend(dC)
            Dt.extend(dS)
        if selMode[1]:
            dAlpha = np.zeros((num['nodePairs'],num['nodes']))
            dBeta =  np.zeros((num['nodePairs'],num['nodes']))
        if selMode[2]:
            dGamma = np.zeros((num['nodePairs'],num['nodes']))
            dDelta = np.zeros((num['nodePairs'],num['nodes']))

        for r in range(num['nodePairs']):
            colInds = [ nodePairs['nodes'][r,0], \
                        nodePairs['nodes'][r,1] ]
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
    def compPhiHatAndCovPhiHat(self, X, num, selMode, nodePairs):

        # Sufficient Statistics
        H, sC, sS, sAlpha, sBeta, sGamma, sDelta = \
            self.compSufStat(X, selMode, nodePairs)
        Ht = H.T

        # define compPhiHatAndCovPhiHat
        HhatX = np.mean(H,1) # H is (n_params, n_trials)

        # gammaHatX: average of gammaXt over trials
        gammaHatX = np.zeros((num['param'],num['param']))
        cache_gammaXt = True
        try:
            gammaXt_trials = np.zeros((num['trials'],num['param'],num['param']))
        except MemoryError:
            cache_gammaXt = False
        for t in range(num['trials']):
            gammaXt = self.compGammaXt(t, num, selMode, nodePairs,
                                sC, sS, sAlpha, sBeta, sGamma, sDelta)
            if cache_gammaXt:
                gammaXt_trials[t, :, :] = gammaXt
            gammaHatX += gammaXt
        gammaHatX /= num['trials']   

        # Estimating phiHat
        L = scipy.linalg.cholesky(gammaHatX + 1e-8 * np.eye(gammaHatX.shape[0]), lower=True)
        phiHat = scipy.linalg.cho_solve((L, True), HhatX)


        # Estimating the covariance of phiHat: covPhiHat
        vHatX = np.zeros((num['param'],num['param']))
        for t in range(num['trials']):
            if cache_gammaXt:

                vVec = np.dot(gammaXt_trials[t, :, :], phiHat) - Ht[t, :]
            else:
                gammaXt = self.compGammaXt(t, num, selMode, nodePairs,
                                sC, sS, sAlpha, sBeta, sGamma, sDelta)

                vVec = np.dot(gammaXt, phiHat) - Ht[t, :]

            vHatX += np.outer(vVec, vVec)
        vHatX /= num['trials']  

        Linv = np.linalg.inv(L)
        covPhiHat = 1/num['trials'] * scipy.linalg.cho_solve((L, True), vHatX) @ Linv.T @ Linv

        return (phiHat, covPhiHat)

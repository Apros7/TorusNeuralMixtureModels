import numpy as np
import scipy.linalg


class scoreMatching:
    def __init__(self):
        pass

    # Sufficient Statistics
    def compSufStat(self, X, selMode, nodePairs):
        """ Computes the sufficient statistics of the torus graph model.

        The sufficient statistics are functions of the data X;
        selMode determines the types of blocks that are included.  

        Let

        ``Xi = X[nodePairs['nodes'][:,0],:]``

        ``Xj = X[nodePairs['nodes'][:,1],:]``

        ``Xdif = Xi - Xj``

        ``Xsum = Xi + Xj``

        Args:
            described in :meth:`funsTG.torusGraphs`.
        
        Outputs:
            H : numpy array (num['param'], num['trials']).
                H is the Laplacian, with respect to the data, of the suficient statistics. 
                The sufficient statistics concatenate sC, sS, sAlpha, sBeta, sGamma, and sDelta.
                After taking the laplacian (second derivative and sum), H is equal to concatenating  
                sC, sS, 2*sAlpha, 2*sBeta, 2*sGamma, 2*sDelta.
                However, whether these blocks are included depends on selMode.

            sC : cos(X)

            sS : sin(X)

            sAlpha : cos(Xdif)

            sBeta : sin(Xdif)

            sGamma : cos(Xsum)

            sDelta = sin(Xsum)

        >>> selMode = (True, True, True) 
        >>> nodePairs = {'nodes': np.array([ [0,1], [0,2], [1,2] ]) }
        >>> X = np.array([[0, pi], [pi, pi/2], [pi/2, pi/3] ]) # 3 nodes, 2 trials
        >>> Xi = X[nodePairs['nodes'][:,0],:]
        >>> Xj = X[nodePairs['nodes'][:,1],:]
        >>> H, sC, sS, sAlpha, sBeta, sGamma, sDelta = compSufStat(X, selMode, nodePairs)
        >>> h = np.concatenate((np.cos(X), np.sin(X), 2*np.cos(Xi-Xj), 2*np.sin(Xi-Xj), 2*np.cos(Xi+Xj), 2*np.sin(Xi+Xj) ), axis = 0)
        >>> np.all(abs(H - h) < 1e-12)
        True

        """

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
        """ Computes a function Gamma for a single trial from X. 

        Let Dt (num['param'], num[''nodes]) be the Jacobian of the suficient statistics; 
        by Jacobian we mean the first derivative with respect to the data, then

        ``gammaXt = Dt @ Dt.T # np.matmul(Dt,Dt.T)``

        Args:
            described in :meth:`funsTG.torusGraphs` and :meth:`funsTG.compSufStat`.
        
        Outputs:
            gammaXt : numpy array (num['param'], num['param']).

        >>> X = np.array([ [-2.1,-1.4,2,2.8,-0.7,-2], [1.3,-2.9,1.2,-2.9,1.7,-0.1], [-2.9,-2.5,-1.1,-0.4,1.9,-0.3] ])
        >>> selMode = (True, True, True)
        >>> num = {'nodes': 3, 'trials': 6, 'nodePairs': 3, 'param': 18}
        >>> nodePairs = {'nodes': np.array([ [0,1], [0,2], [1,2] ])}
        >>> H, sC, sS, sAlpha, sBeta, sGamma, sDelta = compSufStat(X, selMode, nodePairs)
        >>> gammaXt = compGammaXt(1, num, selMode, nodePairs, sC, sS, sAlpha, sBeta, sGamma, sDelta)
        >>> sum(gammaXt.flatten())
        15.5466459624921

        """
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
        """ Computes the parameter estimates and their covariance. 
        
        Args:
            described in :meth:`funsTG.torusGraphs`.
        
        Outputs: 
            phiHat : numpy array with num['param'] elements.
                Model parameters estimates.
            
            covPhiHat : numpy matrix (num['param'], num['param']) 
                Covariance for the model parameter estimates.

        >>> X = np.array([ [-2.1,-1.4,2,2.8,-0.7,-2], [1.3,-2.9,1.2,-2.9,1.7,-0.1], [-2.9,-2.5,-1.1,-0.4,1.9,-0.3] ])
        >>> selMode = (True, True, True)
        >>> num = {'nodes': 3, 'trials': 6, 'nodePairs': 3, 'param': 18}
        >>> nodePairs = {'nodes': np.array([ [0,1], [0,2], [1,2] ])}
        >>> phiHat, covPhiHat = compPhiHatAndCovPhiHat(X, num, selMode, nodePairs)
        >>> phiHat
        array([-39.14836993,   7.87810064,  39.42105888, -16.63875025,
                64.03174047, -47.82433269,  25.35498747, -31.41571978,
                58.81343556, -50.01232377,  -5.12549116,  11.0160441 ,
            -18.56673785,  -1.31352256,  16.1610135 , -23.23622016,
            -31.69057573, -34.39433503])
        >>> sum(covPhiHat.flatten())
        23556.69806086992

        """
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
        #phiHat = np.linalg.inv(gammaHatX) @ HhatX # older version, less stable?

        # Estimating the covariance of phiHat: covPhiHat
        vHatX = np.zeros((num['param'],num['param']))
        for t in range(num['trials']):
            if cache_gammaXt:
                #vVec = (gammaXt_trials[t, :, :] @ phiHat) - H[:,t]
                vVec = np.dot(gammaXt_trials[t, :, :], phiHat) - Ht[t, :]
            else:
                gammaXt = self.compGammaXt(t, num, selMode, nodePairs,
                                sC, sS, sAlpha, sBeta, sGamma, sDelta)
                #vVec = (gammaXt @ phiHat) - H[:,t]
                vVec = np.dot(gammaXt, phiHat) - Ht[t, :]
            #vVec = np.reshape(vVec,(-1,1))
            #vHatX += vVec @ vVec.T
            vHatX += np.outer(vVec, vVec)
        vHatX /= num['trials']  
        #invGammaHatX = np.linalg.inv(gammaHatX) # older version, less stable?
        #covPhiHat = (invGammaHatX @ vHatX @ invGammaHatX)/num['trials']
        Linv = np.linalg.inv(L)
        covPhiHat = 1/num['trials'] * scipy.linalg.cho_solve((L, True), vHatX) @ Linv.T @ Linv

        return (phiHat, covPhiHat)

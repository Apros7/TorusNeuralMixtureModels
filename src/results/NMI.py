import numpy as np

# Z1 er output fra ligning 12 matsuda

def calc_MI(Z1,Z2):
   P=Z1@Z2.T
   PXY=P/np.sum(P)
   PXPY=np.outer(np.sum(PXY,axis=1),np.sum(PXY,axis=0))
   ind=np.where(PXY>0)
   MI=np.sum(PXY[ind]*np.log(PXY[ind]/PXPY[ind]))
   return MI


def calc_NMI(Z1,Z2):
   Z1 = np.double(Z1)
   Z2 = np.double(Z2)
   #Z1 and Z2 are two partition matrices of size (KxN) where K is number of components and N is number of samples
   NMI = (2*calc_MI(Z1,Z2))/(calc_MI(Z1,Z1)+calc_MI(Z2,Z2))
   return NMI

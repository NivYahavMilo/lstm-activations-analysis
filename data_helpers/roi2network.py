import numpy as np
import pickle
import os

if not os.path.exists('data//networks_ts'):
    os.mkdir('data//networks_ts')


# TODO: refactor this class

class Roi2Networks:
    def __init__(self, data):
        self.data = data
        
    def VisualNetwork(self):
        VisualNetwork = {}
        
        for key in self.data:
            
            VisualNetwork[key] = np.concatenate((self.data[key][:,0:24,:], self.data[key][:,150:173,:]),axis=1)
            
        with open('data//networks_ts//' + 'VisualNetwork.pkl', 'wb') as f:
            pickle.dump(VisualNetwork, f)
        
    
    def SomMotor(self):
        
        SomMotor = {}
        for key in self.data:
            
            SomMotor[key] = np.concatenate((self.data[key][:,24:53,:], self.data[key][:,173:201,:]),axis=1)
        
        with open('data//networks_ts//' + 'SomMotor.pkl', 'wb') as f:
            pickle.dump(SomMotor, f)
    
    def DorsalAttention(self):
        
        DorsalAttention = {}
        
        for key in self.data:
            
            DorsalAttention[key] = np.concatenate((self.data[key][:,53:69,:], self.data[key][:,201:219,:]),axis=1)
            
        with open('data//networks_ts//' + 'DorsalAttention.pkl', 'wb') as f:
            pickle.dump(DorsalAttention, f)
    
    def SalVentAttn(self):
        
        SalVentAttn = {}
        
        for key in self.data:
            
            SalVentAttn[key] = np.concatenate((self.data[key][:,69:85,:], self.data[key][:,219:237,:]),axis=1)
            
        with open('data//networks_ts//' + 'SalVenAttn.pkl', 'wb') as f:
            pickle.dump(SalVentAttn, f)
    
    def Limbic(self):
        Limbic = {}
        
        for key in self.data:
            Limbic[key] = np.concatenate((self.data[key][:,85:95,:], self.data[key][:,237:247,:]),axis=1)
            
        with open('data//networks_ts//' + 'Limbic.pkl', 'wb') as f:
            pickle.dump(Limbic, f)
    
    def Cont(self):
        Cont = {}
        for key in self.data:
            Cont[key] = np.concatenate((self.data[key][:,95:112,:], self.data[key][:,247:270,:]),axis=1)
        
        with open('data//networks_ts//' + 'Cont.pkl', 'wb') as f:
            pickle.dump(Cont, f)
    
    def DMN(self):
        DMN = {}
        for key in self.data:
            DMN[key] = np.concatenate((self.data[key][:,112:150,:], self.data[key][:,270:300,:]),axis=1)
            
        with open('data//networks_ts//' + 'DMN.pkl', 'wb') as f:
            pickle.dump(DMN, f)
import myw2v
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.externals import joblib

# model=myw2v.trainmodel('traindata1.txt',True)
# myw2v.savemodel('model_w2v.model',model)
model=myw2v.loadmodel('model_w2v.model')
XX,YY=myw2v.builddata('action.txt','condition.txt',0,1,model)
myw2v.testsvmmodel('svmmodel.model',4.99878168967,0.0171463391473,XX,YY)
# myw2v.PSO(20,model)





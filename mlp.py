"""
Multilayer Perceptron Neural Network code 

24 March 2022
"""

## Loading Libraries
#allow loading make_features from anywhere
import sys
sys.path.append('/depot/tdm-musafe/apps')
from make_features import load_data
from make_features import make_features
from make_features import make_undirected
#importing 3rd party libraries
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 


"""
Reading Data
"""

#drop_batches = False as we do not want to drop the entries which were added on the same day - questionable integrity
#drop_early = True as we do not want to read the entries before Nov 10 as they are mostly wrong
dfRawData = load_data(drop_batches = False, drop_early = True)
dfFeature = make_features(drop_batches = False, drop_early = True)
dfTestFeature = make_features(test_data=True)

"""
Defining a feed forward neural network using the nn module from torch
activation function used for forward propagation - RelU
"""

# Use this class to set up network structure
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__() 
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        return torch.sigmoid(self.layer3(x)) # Sigmoid output 

# Making use of CUDA if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu") # Decide device between CUDA (GPU) and CPU compute
#print(f'Is CUDA available? {use_cuda}')


    
"""
Training function

Defines the neural network architecture with default 125 hidden neurons
trains the neural network for 1500 default iterations
"""
def train_mlp(x_train, y_train, no_iter = 1500, no_hidden_neurons = 125):

    MLP = FeedForward(input_size = x_train.shape[1], hidden_size = no_hidden_neurons, output_size = 1).to(device)
    
    torch_x_train = torch.FloatTensor(x_train).to(device)
    torch_y_train = torch.FloatTensor(y_train).to(device)
    
    for epoch in range(no_iter):
        ## Predict values using updated model
        y_pred = MLP(torch_x_train)

        ## Find predicted class (as a number {1,0})
        pred = y_pred.gt(0.5) + 0.0
        
        loss = nn.BCELoss()
        learning_rate = 0.2
        error = loss(y_pred, torch_y_train.view(-1,1)) ## .view(-1,1) avoids a warning about the tensor dimension

        MLP.zero_grad() ## Zero current gradients (Pytorch accumulate gradients by default)
        
        error.backward() ## Compute gradients via backpropagation
        
        for f in MLP.parameters():
            f.data.sub_(f.grad.data * learning_rate)
    
    return MLP


"""
Validation function

This function uses the trained MLP and evaluates its performace using the validation data
Returns 7 evaluation metrics namely,
accuracy, sensitivity, specificity, false positive rate, 
false negative, true positive rate, true negative rate
"""
def validate_mlp(X_valid, y_valid, MLP):

    torch_X_validate = torch.FloatTensor(X_valid).to(device)
    
    y_pred = MLP(torch_X_validate)
    finalPrediction = []
    for prediction in y_pred:
        if prediction > 0.5: 
            finalPrediction.append(1)
        else:
            finalPrediction.append(0)
    
    n_fp = 0
    n_fn = 0
    n_tp = 0
    n_tn = 0
    for i in range(len(finalPrediction)):
        if finalPrediction[i] == 1 and y_valid[i] == 0:
            n_fp += 1
        elif finalPrediction[i] == 1 and y_valid[i] == 1:
            n_tp += 1
        elif finalPrediction[i] == 0 and y_valid[i] == 1:
            n_fn += 1
        else:
            n_tn +=1

    accuracy = (n_tp + n_tn)/len(y_valid)
    
    sensitivity = n_tp/(sum(y_valid)+1)
    specificity = n_tn/(len(y_valid) - sum(y_valid)+1)
    
    fpRate = n_fp/(n_fp + n_tn + 1)
    fnRate = n_fn/(n_fn + n_tp + 1)
    tpRate = n_tp/(n_tp + n_fp + 1)
    tnRate = n_tn/(n_tn + n_fn + 1)
    '''                         
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Sense: {sensitivity:.3f}, Specf: {specificity:.3f}")
    print(f"fp: {fpRate:.3f}, fn: {fnRate:.3f}")
    print(f"tp: {tpRate:.3f}, tn: {tnRate:.3f}")
    '''   
    return accuracy, sensitivity, specificity, fpRate, fnRate, tpRate, tnRate
    
    
    
"""
PCA Graph

sm = SMOTE(random_state = 42)

X_trainArr = dfFeature.to_numpy()
Y = pd.DataFrame(dfRawData[0]["motion"])
y_trainArr = Y["motion"].to_numpy()

X_train_oversampled, y_train_oversampled = sm.fit_resample(X_trainArr, y_trainArr)


y_train_oversampled[1890:] = 2
y_train_oversampled[1890:]

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train_oversampled)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

labelDf = pd.DataFrame(data = y_train_oversampled, columns = ['motions'])

finalDf = pd.concat([principalDf, labelDf['motions']], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 2, 1]
colors = ['blue', 'r', 'orange']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['labels'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(['other','danger','artificial danger'])
ax.grid()

"""


"""
Running the model.

Used 10-fold cross validation and SMOTE here
Returns accuracy, precision, and sensitivity for each specification
"""

n_folds = 10

kf = KFold(n_splits = n_folds)

X = dfFeature   #dfUndirFeature
X = X.reset_index(drop=True)

Y = pd.DataFrame(dfRawData[0]["motion"])   #dfRawData
Y['motion'] = Y['motion'].map({'trip':1, 'slip':1, 'fall':1, 'other':0})
Y = Y.reset_index(drop=True)

accuracy, sensitivity, specificity, fpRate, fnRate, tpRate, tnRate = 0, 0, 0, 0, 0, 0, 0

accuracyArr = []
percisionArr = []
senseArr = []

for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    #print(f'For fold {fold}:')
    X_trainDf = X.loc[train_index]
    y_trainDf = Y.loc[train_index]
    X_testDf = X.loc[test_index]
    y_testDf = Y.loc[test_index]

    X_trainArr = X_trainDf.to_numpy()
    y_trainArr = y_trainDf["motion"].to_numpy()
    X_testArr = X_testDf.to_numpy()
    y_testArr = y_testDf["motion"].to_numpy()   

    sm = SMOTE(random_state =42 )
    #un = NearMiss()
    #sm = SMOTEENN(sampling_strategy=0.5)
    #sm = SMOTEENN(sampling_strategy=1)
    X_train_oversampled, y_train_oversampled = sm.fit_resample(X_trainArr, y_trainArr)

    MLP = train_mlp(X_train_oversampled, y_train_oversampled, no_iter = 1750, no_hidden_neurons = neuronLevel)
    foldResult = validate_mlp(X_testArr, y_testArr, MLP)

    accuracy += foldResult[0]
    sensitivity += foldResult[1]
    specificity += foldResult[2]
    fpRate += foldResult[3]
    fnRate += foldResult[4]
    tpRate += foldResult[5]
    tnRate += foldResult[6]

    MLP = []

accuracy = accuracy/fold
sensitivity = sensitivity/fold
specificity = specificity/fold
fpRate = fpRate/fold
fnRate = fnRate/fold
tpRate = tpRate/fold
tnRate = tnRate/fold

accuracyArr.append(accuracy)
percisionArr.append(tpRate)
senseArr.append(sensitivity)

print("-----------------------------")
print(f"Average Validation Accuracy: {accuracy:.3f}")
print(f"Sense: {sensitivity:.3f}, Specf: {specificity:.3f}")
print(f"tp: {tpRate:.3f}, fn: {fnRate:.3f}")
print(f"fp: {fpRate:.3f}, tn: {tnRate:.3f}")
print("-----------------------------")



"""
Plot the performance(accuracy, sensitivity, and precision) of the MLP along the no. of neurons used

fig = plt.gcf()
fig.set_size_inches(12, 8)

x_axis = np.arange(len(neuronLevelArr))

plt.bar(x_axis -0.1, accuracyArr, width=0.1, label = 'Accuracy')
plt.bar(x_axis, senseArr , width=0.1, label = 'Sensitivity')
plt.bar(x_axis + 0.1, percisionArr, width=0.1, label = 'Precision')

plt.xticks(x_axis, neuronLevelArr, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("# of hidden neurons", fontsize=15)

plt.legend(loc=2, prop={'size': 16})

plt.show()
"""
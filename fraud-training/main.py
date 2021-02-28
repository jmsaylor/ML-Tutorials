import pandas as pd

from sklearn import preprocessing
import numpy as np

names = np.array(['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
       'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud',
       'isFlaggedFraud'])

dataset = pd.read_csv("/home/jm/Data/PS_20174392719_1491204439457_log.csv", names=names, skiprows=100000,nrows=1000000)
# csv_iter = pd.read_csv("/home/jm/Data/PS_20174392719_1491204439457_log.csv", iterator=True, chunksize=1000)
# frauds = pd.concat([chunk[chunk['isFraud'] == 1] for chunk in csv_iter])

frauds = pd.concat([dataset[dataset['isFraud'] == 1]])


frauds = frauds.iloc[:, 1:-1].values

non_tx_cashout = []
transfer_cashout = []

for tx in frauds:
    if len(non_tx_cashout) == 0:
        non_tx_cashout.append(tx)
        continue
    
    prev = non_tx_cashout.pop()
    
    if tx[1] != prev[1]:
        non_tx_cashout.append(prev)
        non_tx_cashout.append(tx)
    else:
        transfer_cashout.append(prev)
        transfer_cashout.append(tx)

non_tx_cashout = pd.DataFrame(non_tx_cashout, columns=names[1:-1])
transfer_cashout = pd.DataFrame(transfer_cashout, columns=names[1:-1])


# non_tx_cashout = np.array(non_tx_cashout)
#X = dataset.iloc[:, 1:-2].values
#y = dataset.iloc[:, -2].values

#from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
from sklearn import datasets

# Load the digits dataset
digits = datasets.load_digits()
print(digits.keys())
print('Displayed digit is ' + str(digits.target[648]))

plt.figure(1, figsize=(4, 4))
plt.imshow(digits.images[648], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

'''
explanation for team:
the datasets loads a bunch object with the training data - basically a dictionary with easier access to elements
It's keys are: 'data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'
important ones: 
data is the main data array
image is the data converted into a 2d array
target tells which digit the data is for
'''


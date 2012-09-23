import cPickle as pickle
import os
from sklearn import svm
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
  # Load Config
  config = pickle.load(open('Config.pickle','r'))
  path_pickles = config[0]

  # Load DatasetTBA
  X = pickle.load(open(os.path.join(path_pickles, 'DatasetTBA.pickle'),'r'))

  # Load Targets
  y = pickle.load(open(os.path.join(path_pickles, 'Targets.pickle'),'r'))

  # Choose Classifier Model
  #clf = svm.SVC(kernel='linear',probability=True)
  clf = LogisticRegression()

  clf.fit(X, y)

  # Dump pickle
  pickle.dump(clf)

'''
Pipelines
Pipeline can be used to chain multiple estimators into one. This is useful as there is often a fixed
sequence of steps in processing the data, for example feature selection, normalization, classification.
Pipelines serve two purposes:
    * convenience: you only have to call fit and predict once on your data to fit a whole sequence of estimators
    * Joint parameter selection : you can grid search over parameters of all estimators in the pipeline once
all estimators in a pipeline, except the last one, must be transformers (must have a transform method)
the last estimator may be any type (transformer, classifier,etc,...)
'''

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Binarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC  # support vector machine
from sklearn.decomposition import PCA  # dimentionality reduction
from sklearn.datasets import load_iris

Pipeline  # <class 'sklearn.pipeline.Pipeline'>
# clf : classifier
estimators = [('reduce_dim', PCA(n_components=2)), ('clf', SVC())]
pipe = Pipeline(estimators)
pipe

X, y = load_iris(return_X_y=True)

pipe.fit(X, y).score(X, y)

'''
The utility function "make_pipeline" is a shorthand for constructing pipelines; it takes a variable 
number of estimators and returns a pipeline, filling in the names automatically.
'''


# Pipeline(steps=[('binarizer', Binarizer()), ('multinomialnb', MultinomialNB())])
make_pipeline(Binarizer(), MultinomialNB())

pipe.steps[0]
pipe.named_steps['reduce_dim']
# Set the parameters of this estimator
pipe.set_params(clf__C=10)  # ('clf', SVC(C=10))

# search to find best parameters for each estimator in the pipeline
params = dict(reduce_dim__n_components=[2, 5, 10],
              clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=params)


# this way we can edit the pipeline and even add new estimators to it (logisticRegressor added)
params = dict(reduce_dim=[None, PCA(5), PCA(10)],
              clf=[SVC(), LogisticRegression()],
              clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=params)

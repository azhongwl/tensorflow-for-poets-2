import pickle
import os
 
import numpy as np
import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from sklearn.multiclass import OneVsRestClassifier
import pickle

def load_bottlenecks(path, unpickled, postfix):
    """

    :param path:
    :param filename:
    :return:
    """

    bottle_necks = []

    for idx in range(0, len(unpickled)):
        person = unpickled[idx][0].split()
        fname = '_'.join(person)+'_'+unpickled[idx][1].zfill(4)+postfix
        m_data = open(path+'/'+fname, 'r')
        data = m_data.read().split(',')
        numbers = [float(data[i]) for i in range(0, len(data))]
        bottle_necks.append(numbers)

    return bottle_necks


def load_ground_truth_cache(labels, lines):
    class_count = len(labels)
    ground_truth_all = []
    for line in lines:
        ground_truth = [0]*class_count
        for i in range(class_count):
            if float(line[i]) > 0:
                ground_truth[i] = 1

        ground_truth_all.append(ground_truth)

    return ground_truth_all


def train_svm_classifer(features, labels, model_output_path):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance
 
    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # save 20% of data for performance evaluation
    scaler = StandardScaler()
    features = scaler.fit(features).transform(features)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.05)

    model_to_set = OneVsRestClassifier(SVC(probability=True))

    parameters = {
        "estimator__C": [1, 10, 100],
        "estimator__kernel": ["rbf"],
        "estimator__gamma": [1e-2, 1e-3, 1e-4],
    }

    model_tunning = grid_search.GridSearchCV(model_to_set, param_grid=parameters, scoring='accuracy', n_jobs=2)

    model_tunning.fit(X_test, y_test)

    print(model_tunning.best_score_)
    print(model_tunning.best_params_)



    # clf = OneVsRestClassifier(SVC(probability=True))
    # svm = SVC(probability=True, max_iter=100)
    # clf = OneVsRestClassifier(grid_search.GridSearchCV(svm, param, cv=10, verbose=0, n_jobs=6) )
    # clf.fit(X_test, y_test)

    # print(clf.n_jobs)


 
    # request probability estimation
    # svm = SVC(probability=True)
 
    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    # clf = grid_search.GridSearchCV(svm, param,
    #         cv=10, n_jobs=2, verbose=3)
    #
    # clf.fit(X_train, y_train)
 
    if os.path.exists(model_output_path):
        joblib.dump(model_tunning.best_estimator_, model_output_path)
    else:
        print("Cannot save trained svm model to {0}.".format(model_output_path))

    print("doing good!")

    return model_tunning
    # print("\nBest parameters set:")
    # print(clf.best_params_)
    #
    # y_predict=clf.predict(X_test)
    #
    # labels=sorted(list(set(labels)))
    # print("\nConfusion matrix:")
    # print("Labels: {0}\n".format(",".join(labels)))
    # print(confusion_matrix(y_test, y_predict, labels=labels))
    #
    # print("\nClassification report:")
    # print(classification_report(y_test, y_predict))


def train_svm_classifer2(features, labels, model_output_path):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance

    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # save 20% of data for performance evaluation
    scaler = StandardScaler()
    features = scaler.fit(features).transform(features)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.20)

    best_model = None
    best_acc = 0
    kernels = ['rbf']

    gammas = [1e-2, 1e-3, 1e-4]
    Cs = [1, 10, 100, 1000, 10000]

    for mgamma in gammas:
        for mC in Cs:
            for mkernel in kernels:
                model_to_set = OneVsRestClassifier(
                    SVC( probability=False,
                         kernel=mkernel,
                         C=mC, gamma=mgamma,
                         max_iter=10),
                    n_jobs=4)

                model_to_set.fit(X_train, y_train)

                model_name = model_output_path +'_gamma_'+str(mgamma)+'_C_'+str(mC)+'_kernel_'+mkernel+'.p'
                outfile = open(model_name,'wb')
                pickle.dump(model_to_set,outfile)
                outfile.close()




def prune_data(keys, lines, included_keys = [], excluded_keys = []):
    indices = []
    if included_keys == []:
        for i in keys[2:]:
            if i not in excluded_keys:
                included_keys.append(i)
    new_keys = keys[:2] + included_keys
    for k in included_keys:
        indices.append(keys.index(k))
        assert(keys[indices[-1]] == k)
    new_lines = []
    for line in lines:
        new_line = line[:2]
        for i in indices:
            new_line.append(line[i])
        new_lines.append(new_line)
    return new_keys, new_lines


def prune_data2(keys, lines, excluded_keys={}):
    new_keys = keys[2:]
    to_remove = []

    for i in range(len(new_keys) - 1, 0):
        if (new_keys[i] in excluded_keys):
            new_keys.pop(i)
            to_remove.append(i)

    new_lines = []
    for i in range(0, len(lines)):
        new_line = lines[i][2:]

        for rem in range(len(to_remove) - 1, 0, -1):
            new_line.pop(rem)

        new_lines.append(new_line)

    return new_keys, new_lines

if __name__ == '__main__':

    fname = 'lfw_header_lines.p'
    keys_lines = pickle.load(open(fname, 'rb'))
    keys = keys_lines['header']
    lines = keys_lines['lines']
    # included_keys = ['male', 'asianindian', 'eastasian', 'african', 'latino', 'caucasian']
    # keys, lines = prune_data(keys, lines, excluded_keys = ['Male']) # duplicate keys

    # print('=================================================',len(keys))
    # print('=================================================', len(lines))
    #
    # lines_trans = np.array(lines).transpose()
    #
    # lines = lines_trans.tolist()
    # keys2 = [x for x in range(0,len(keys))]

    path = '/Users/jmarcano/dev/andrews/tensorflow-for-poets-2/tf_files/bottlenecks_funneled/lfw_all'
    postfix = '.jpg_mobilenet_0.50_224.txt'

    bottlenecks = load_bottlenecks(path, lines, postfix)

    # new_keys, new_lines = prune_data(keys, lines, excluded_keys=['Male'])

    # keys, lines = prune_data(keys, lines, excluded_keys=['Male'])  # duplicate keys

    keys, lines = prune_data2(keys, lines, {'Male':1})

    ground_truth = load_ground_truth_cache(keys, lines)


    clf = train_svm_classifer2(np.array(bottlenecks), np.array(ground_truth), '/Users/jmarcano/dev/andrews/tensorflow-for-poets-2/svm_models/svm')

    print("done")


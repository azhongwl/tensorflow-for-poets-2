from scripts.svm_script import *

from sklearn.neighbors import NearestNeighbors

def train_knn_classifier(features, labels, model_output_path):

    scaler = StandardScaler()
    features = scaler.fit(features).transform(features)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.25)



    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        for cov in covariances:
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cov, max_iter=100,
                                          init_params='kmeans')

            print(cov, n_components)
            model_to_set = OneVsRestClassifier(gmm, n_jobs=8)
            model_to_set.fit(X_train, y_train)

            model_name = model_output_path+str(n_components)+'_'+cov+'.p'
            outfile = open(model_name, 'wb')
            # print(cov, n_components)

            pickle.dump(model_to_set, outfile)
            outfile.close()


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


    train_gmm_classifier(np.array(bottlenecks),
                         np.array(ground_truth),
                         '/Users/jmarcano/dev/andrews/tensorflow-for-poets-2/knn_models/knn_')

    print("done")
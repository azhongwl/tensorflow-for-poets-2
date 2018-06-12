from scripts.svm_script import *
from sklearn import mixture

def train_gmm_classifier(features, labels, model_output_path):

    scaler = StandardScaler()
    features = scaler.fit(features).transform(features)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.25)

    best_gmm = None
    n_components_range = range(3, 16)
    covariances = ['diag']#, 'spherical']#, 'tied']

    best_acc = 0.0
    best_model = None

    classes = len(labels[0])

    tests = len(X_test)

    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        for cov in covariances:
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cov, max_iter=10000,
                                          init_params='kmeans', verbose=False)

            print(cov, n_components)
            model_to_set = OneVsRestClassifier(gmm, n_jobs=8)
            model_to_set.fit(X_train, y_train)

            results = np.zeros((tests + 1, classes), dtype=float)

            # model = pickle.load(open(path + file, 'rb'))

            summary_file_name = model_output_path + cov + '_' + str(n_components) + '_summary.p'

            try:

                predictions = model_to_set.predict(X_test)
                # model_to_set.

                # print(predictions)

                # we don't really need to store all of it, but just in case

                for i in range(0, tests):
                    for j in range(0, classes):
                        if predictions[i][j] == y_test[i][j]:
                            # 1.0 means we classified the label test correctly. 0.0 means we did not
                            results[i][j] = 1.0
                            # This is what really matters. I am storing the rest of the info just in case
                            results[tests][j] += 1.0

                # Time to find the accuracy for each label!
                # Storing the results in the last row

                overall = 0.0

                for i in range(0, classes):
                    results[tests][i] = results[tests][i] / tests
                    overall += results[tests][i]

                overall = overall / classes

                if overall > best_acc:
                    best_acc = overall
                    best_model = summary_file_name
                    pickle_dump(model_to_set, model_output_path + 'best.p')

                # write the summary file . It won't hurt

                sum_file = open(summary_file_name, 'wb')
                pickle.dump(results, sum_file)

                print(overall)

                sum_file.close()


            except:
                print("SOmething went wrong. Skipping ")

        pickle.dump([best_acc, best_model], open(model_output_path + 'best_results_info.p'))


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
                         '/Users/jmarcano/dev/andrews/tensorflow-for-poets-2/gmm_models/gmm_')

    print("done")
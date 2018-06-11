from scripts.svm_script import *
from scripts.GMM_test import *

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

    keys, lines = prune_data2(keys, lines, {'Male': 1})

    ground_truth = load_ground_truth_cache(keys, lines)

    svm_path = '/Users/jmarcano/dev/andrews/tensorflow-for-poets-2/svm_models/'
    out_dir_svm = '/Users/jmarcano/dev/andrews/tensorflow-for-poets-2/svm_summary/'

    gmm_path = '/Users/jmarcano/dev/andrews/tensorflow-for-poets-2/gmm_models/'
    out_dir_gmm = '/Users/jmarcano/dev/andrews/tensorflow-for-poets-2/gmm_summary/'

    find_best_model(svm_path, ground_truth, np.array(bottlenecks), out_dir_svm )

    find_best_model(gmm_path, ground_truth, np.array(bottlenecks), out_dir_gmm)
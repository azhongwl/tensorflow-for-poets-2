from glob import glob
import shutil

paths = glob("/Users/jmarcano/dev/andrews/tensorflow-for-poets-2/tf_files/lfw_funneled/*/*")

dest = "/Users/jmarcano/dev/andrews/tensorflow-for-poets-2/tf_files/lfw_all"

for file in paths:
    shutil.move(file, dest)
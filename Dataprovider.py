import json
import numpy as np
from scipy.misc import imread, imsave, imresize

class Dataprovider:


    @staticmethod
    def getdata(data_dir, readfromimage = 1):

        X_idx = []
        y = []
        try:
            with open(data_dir+'/data.txt', 'r') as f:
                for line in f:
                    name = line.split()
                    X_idx.append(name[0])
                    y.append(int(name[1]))

        except IOError:
            print "No data.txt provided, please run resizeimage.sh first!"
            return None

        X_size = len(X_idx)
        X_size = 3000
        data_file_dir = data_dir + "/data.json"

        if readfromimage == 0:
            with open(data_file_dir,'r') as f:
                data = json.load(f)
                X = data['X']
                y = data['y']
            return X, y

        X = []
        y = np.array(y[:X_size])
        for i in xrange(X_size):
            img = imread('../'+X_idx[i],mode='RGB')
            X.append(img)

        # with open(data_file_dir, 'w') as f:
        #     json.dump({'X':X, 'y':y.tolist()},f)

        return X, y


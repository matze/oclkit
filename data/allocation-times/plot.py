import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and f.endswith('.txt')]
    
    for fname in sorted(files):
        data = np.loadtxt(fname)
        name = fname.split('.')[0]
        xs = data[:, 0] / 1024. / 1024.
        ys = np.mean(data[:, 1:], axis=1)
        plt.loglog(xs, ys, '.-', label=name)

    plt.legend(loc=4)
    plt.xlabel('Size (MB)')
    plt.ylabel('Time (s)')
    plt.show(True)

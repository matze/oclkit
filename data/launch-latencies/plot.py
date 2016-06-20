import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and f.endswith('.txt')]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    waits = []
    execs = []
    walls = []
    marks = []

    for fname in sorted(files):
        data = np.loadtxt(fname).reshape(-1, 3)
        data = np.mean(data, axis=0)
        waits.append(data[0])
        execs.append(data[1])
        walls.append(data[2])
        marks.append(fname.split('.')[0])

    indices = np.arange(len(waits))
    width = 0.2

    r1 = ax.bar(indices, waits, width, color='#00b8a9')
    r2 = ax.bar(indices + width, execs, width, color='#f6416c')
    r3 = ax.bar(indices + 2 * width, walls, width, color='#ffde7d')

    ax.set_xticks(indices + width)
    tick_names = ax.set_xticklabels(marks)

    ax.legend((r1[0], r2[0], r3[0]), ('Wait', 'Execution', 'Wall'), loc=1)
    plt.ylabel('Time (us)')
    plt.show(True)

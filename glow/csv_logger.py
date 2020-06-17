import csv
import matplotlib.pyplot as plt
import os
import ipdb

class CSVLogger():
    def __init__(self, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


def plot_csv(fpath):
    with open(fpath, 'r') as f:
        reader = csv.reader(f)
        dict_of_lists = {}
        ks = None
        for i, r in enumerate(reader):
            if i == 0:
                for k in r:
                    dict_of_lists[k] = []
                ks = r
            else:
                for _i, v in enumerate(r):
                    dict_of_lists[ks[_i]].append(float(v))
    for k in dict_of_lists:
        if k == 'global_iteration':
            continue
        fig = plt.figure()
        plt.plot(dict_of_lists['global_iteration'], dict_of_lists[k])
        plt.title(k)
        plt.grid()
        plt.savefig(os.path.join(os.path.dirname(fpath), f'_{k}.png'))
        plt.close(fig)

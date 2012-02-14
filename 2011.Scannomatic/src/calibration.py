#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import os, sys

file_path = 'config/calibration.data'
filter_1 = 'OD'
filter_2 = 'FACS'

try:

    fs = open(file_path)

except:
    print "Path is bad!" 
    sys.exit(0)


data_store = []

for line in fs:
    line = line.strip('\n')

    try:
        data_store.append(eval(line))
    except:
        print "** WARNING: Could not parse ", line

fs.close()

data_list_1 = []
data_list_2 = []
data_labels_1 = []
data_labels_2 = []

for data in data_store:

    try:

        if filter_1 in data[1]:
            data_list_1.append(data[2:])
            data_labels_1.append(data[1].replace(filter_1,'').strip())
        elif filter_2 in data[1]:
            data_list_2.append(data[2:])
            data_labels_2.append(data[1].replace(filter_2,'').strip())

    except:

        print "** WARNING: Data entry has unknown format ", data

data_1_joint_positions = []
data_2_joint_positions = []

for label in data_labels_1:
    if label in data_labels_2:

        data_1_joint_positions.append(data_labels_1.index(label))
        data_2_joint_positions.append(data_labels_2.index(label))
        
    

data_1 = np.asarray(data_list_1)
data_2 = np.asarray(data_list_2)

#GRAPH 1
plt.plot(data_1[data_1_joint_positions,1],data_2[data_2_joint_positions,1],'b.')
plt.xlabel(filter_1)
plt.ylabel(filter_2)
plt.title("Comparison of independent measures")
plt.show()

#GRAPH 2

z2 = np.polyfit(data_1[:,0], data_1[:,1],2)
p2 = np.poly1d(z2)

z3 = np.polyfit(data_1[:,0], data_1[:,1],3)
p3 = np.poly1d(z3)

xp = np.linspace(data_1[:,0].min(), data_1[:,0].max(), 100)

plt.clf()
plt.plot(data_1[:,0], data_1[:,1], 'b.')
plt.plot(xp, p2(xp),'r-', label='2nd deg')
plt.plot(xp, p3(xp),'g-', label='3rd deg')
plt.xlabel("Mean Pixel-Darkening from Colony Growth in 'Kodak Space' (Larger negative number, more stuff on agar)")
plt.ylabel("Independet Cell Estimate per pixel")
plt.title(filter_1 + " based conversion to 'Cell Estimate Space'")
plt.legend(loc=0)
plt.show()

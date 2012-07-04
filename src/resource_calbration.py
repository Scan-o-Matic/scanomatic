#!/usr/bin/env python
"""Produces the calibration curve from calibration data"""

__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson", "Olle Nerman"]
__license__ = "GPL v3.0"
__version__ = "0.995"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"



import matplotlib.pyplot as plt
import numpy as np
import os, sys
from scipy.optimize import curve_fit


file_path = 'config/calibration.data'
file_out_path = 'config/calibration.polynomials'
filter_1 = 'OD'
filter_2 = ''

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

def expand_compressed_vector(compressed_vector):

    vector = []

    for pos in xrange(len(compressed_vector[0])):

        for ith in xrange(compressed_vector[1][pos]):

            vector.append(compressed_vector[0][pos])

    return vector
    
def vector_polynomial_sum_dropped_coeffs(X, *coefficient_array):

    
    coefficient_array = np.array((coefficient_array[0], 0, 0, coefficient_array[1], 0), dtype=np.float64)
    Y = np.zeros(len(X))
    for pos in xrange(len(X)):
        Y[pos] = (np.sum(np.polyval(coefficient_array, X[pos])))

    return Y

def vector_polynomial_sum_dropped_coeffs2(X, *coefficient_array):

    
    coefficient_array = np.array((coefficient_array[0], 0, 0, 0, coefficient_array[1], 0), dtype=np.float64)
    Y = np.zeros(len(X))
    for pos in xrange(len(X)):
        Y[pos] = (np.sum(np.polyval(coefficient_array, X[pos])))

    return Y

def vector_polynomial_sum(X, *coefficient_array):

    Y = np.zeros(len(X))
    for pos in xrange(len(X)):
        Y[pos] = (np.sum(np.polyval(coefficient_array, X[pos])))

    return Y




data_list_1 = []
data_list_2 = []
data_labels_1 = []
data_labels_2 = []

for data in data_store:

    try:

        if filter_1 in data[1] and len(data[-1]) > 0:
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
        
#
# Initialize the arrays of right shape and right dtype
X = np.empty((len(data_list_1),), dtype=object)
Y = np.zeros((len(data_list_1),), dtype=np.float64)
x_min = None
x_max = None
#
# Populating from the first label, expanding the compressed vectors
for pos in xrange(len(data_list_1)):
    X[pos] = np.asarray(expand_compressed_vector(data_list_1[pos][-1]), dtype=np.float64)
    Y[pos] = data_list_1[pos][0]
    if x_min is None or x_min > X[pos].min():
        x_min = X[pos].min()
    if x_max is None or x_max < X[pos].max():
        x_max = X[pos].max()
   
     
#coeff_guess is the initial guess for solution
#This is important as it sets the degree of the polynomial
#An array of length 5 equals a 4th deg pol
coeff_guess = np.asarray((1,1))
#X and Y should be arrays both, should contain as each element
#a vector (array) of measuers
#The length of Y should equal the length of the first dimension of X
#The curve fit will return the best solution for the polynomial
popt, pcov = curve_fit(vector_polynomial_sum_dropped_coeffs2, X, Y, p0=coeff_guess)
popt = [popt[0],0,0,0,popt[1],0]
do_save = True
try:
    fs = open(file_out_path, 'a')
except:
    do_save = False

if do_save:

    fs.write(str(['smart_poly_cal_' + str(len(coeff_guess) +2) + "_deg", list(popt)]) + "\n")
    fs.close()


#create a polynomial for the coefficient vector
pa = np.poly1d(popt)
x_points = np.linspace(x_min, x_max, 100)
plt.clf()
plt.plot(x_points, pa(x_points), label="y=ax**5 + bx")

coeff_guess = np.asarray((1,1))
popt, pcov = curve_fit(vector_polynomial_sum_dropped_coeffs, X, Y, p0=coeff_guess)
popt = [popt[0],0,0,popt[1],0]
p1 = np.poly1d(popt)
plt.plot(x_points, p1(x_points), label="y=ax**4 + bx")

coeff_guess = np.asarray((1,1,1,1))
popt, pcov = curve_fit(vector_polynomial_sum, X, Y, p0=coeff_guess)
p1 = np.poly1d(popt)
plt.plot(x_points, p1(x_points), label=str(len(coeff_guess)-1) + "th deg pol fit")

coeff_guess = np.asarray((1,1,1,1,1,1))
popt, pcov = curve_fit(vector_polynomial_sum, X, Y, p0=coeff_guess)
p1 = np.poly1d(popt)
plt.plot(x_points, p1(x_points), label=str(len(coeff_guess)-1) + "th deg pol fit")

coeff_guess = np.asarray((1,1,1))
popt, pcov = curve_fit(vector_polynomial_sum, X, Y, p0=coeff_guess)
p1 = np.poly1d(popt)
plt.plot(x_points, p1(x_points), label=str(len(coeff_guess)-1) + "th deg pol fit")

plt.legend(loc=0)
plt.xlabel("Per pixel value offset from Colony Growth in 'Kodak Space'")
plt.ylabel("Cell Estimate per pixel")
plt.title("Conversion formula independent measures")
plt.show()



print "Doing random test removing 10 data points for verification from", len(Y), "total"

#
# Initialize the arrays of right shape and right dtype
X1 = np.empty((len(data_list_1)-10,), dtype=object)
Y1 = np.zeros((len(data_list_1)-10,), dtype=np.float64)
X2 = np.empty((10,), dtype=object)
Y2 = np.zeros((10,), dtype=np.float64)

from random import randint
r_positions = []

pos_list = range(len(data_list_1))

while len(r_positions) < 10:
    r_pos  = randint(0, len(pos_list)-1)
    r_positions.append(pos_list[r_pos])
    del pos_list[r_pos]


print "Simulation set", pos_list
print "Test set", r_positions
#
# Populating from the first label, expanding the compressed vectors
in_pos = 0
r_pos = 0
for pos in xrange(len(data_list_1)):
    if pos in pos_list:
        X1[in_pos] = np.asarray(expand_compressed_vector(data_list_1[pos][-1]), dtype=np.float64)
        Y1[in_pos] = data_list_1[pos][0]
        in_pos += 1
    else:
        X2[r_pos] = np.asarray(expand_compressed_vector(data_list_1[pos][-1]), dtype=np.float64)
        Y2[r_pos] = data_list_1[pos][0]
        r_pos += 1


coeff_guess = np.asarray((1,1))
popt, pcov = curve_fit(vector_polynomial_sum_dropped_coeffs2, X1, Y1, p0=coeff_guess)
popt = [popt[0],0,0,0,popt[1],0]
ps = np.poly1d(popt)
plt.clf()
plt.plot(x_points, pa(x_points), label="Full set")
plt.plot(x_points, ps(x_points), label="Set minus 10 random")

plt.legend(loc=0)
plt.xlabel("Per pixel value offset from Colony Growth in 'Kodak Space'")
plt.ylabel("Cell Estimate per pixel")
plt.title("Conversion polynomial stability for withdrawing 10 points")
plt.show()



CY2 = vector_polynomial_sum_dropped_coeffs2(X2, popt[0], popt[4])
print CY2
print Y2
line_max = np.asarray(CY2 + Y2).max()


plt.clf()
plt.plot(Y2, CY2, 'r*')
plt.plot([0, line_max], [0, line_max], 'b-')
plt.xlabel("Lasse's independent cell count measure")
plt.ylabel("Cell Estimate calculation from fitted curve ax**5 + bx = y")
plt.title("Evaluation of polynomial")
plt.show()






sys.exit()



#if filter_2 != "":
    #data_2 = np.asarray(data_list_2)
#
    ##GRAPH 1
    #plt.plot(data_1[data_1_joint_positions,1],data_2[data_2_joint_positions,1],'b.')
#
    #z1 = np.polyfit(data_1[data_1_joint_positions,1],data_2[data_2_joint_positions,1],1)
    #p1 = np.poly1d(z1)
    #xp = np.linspace(data_1[data_1_joint_positions,1].min(), data_1[data_1_joint_positions,1].max(), 100)
#
    #plt.plot(xp, p1(xp), 'g-', label='1nd deg')
    #plt.text(500, 1000, str(p1) + ", r: " + str(p1.r[0]))
#
    #plt.xlabel(filter_1)
    #plt.ylabel(filter_2)
    #plt.title("Comparison of independent measures")
    #plt.legend(loc=1)
    #plt.show()


#GRAPH 2
#z1 = np.polyfit(data_1[:,0], data_1[:,1],1)
#p1 = np.poly1d(z1)

#z2 = np.polyfit(data_1[:,0], data_1[:,1],2)
#p2 = np.poly1d(z2)

#z3 = np.polyfit(data_1[:,0], data_1[:,1],3)
#p3 = np.poly1d(z3)

#xp = np.linspace(data_1[:,0].min(), data_1[:,0].max(), 100)
#x_span = data_1[:,0].max() - data_1[:,0].max()
#y_span = data_1[:,1].max() - data_1[:,1].max()

plt.clf()
plt.plot(data_1[:,0], data_1[:,1], 'b.')
plt.plot(xp, p1(xp),'m-', label='1nd deg')
plt.plot(xp, p2(xp),'r-', label='2nd deg')
plt.plot(xp, p3(xp),'g-', label='3rd deg')
plt.xlabel("Mean Pixel-Darkening from Colony Growth in 'Kodak Space' (Larger negative number, more stuff on agar)")
plt.ylabel("Independet Cell Estimate per pixel")
plt.title(filter_1 + " based conversion to 'Cell Estimate Space'")
plt.legend(loc=0)
#plt.xlim(xmin=data_1[:,0].min() - x_span * 0.15, xmax=data_1[:,0].max() + x_span * 0.15)
#plt.ylim(ymin=data_1[:,1].min() - y_span * 0.15, ymax=data_1[:,1].max() + y_span * 0.15)
#plt.ylim(ymax=2100, ymin=-50)
#plt.xlim(xmax=5, xmin=-100)
plt.ylim(ymax=500, ymin=-50)
plt.xlim(xmax=5, xmin=-30)
plt.show()


if filter_2 != "":
    #GRAPH 3
    z1 = np.polyfit(data_2[:,0], data_2[:,1],1)
    p1 = np.poly1d(z1)

    z2 = np.polyfit(data_2[:,0], data_2[:,1],2)
    p2 = np.poly1d(z2)

    z3 = np.polyfit(data_2[:,0], data_2[:,1],3)
    p3 = np.poly1d(z3)

    xp = np.linspace(data_2[:,0].min(), data_2[:,0].max(), 100)
    x_span = data_2[:,0].max() - data_2[:,0].max()
    y_span = data_2[:,1].max() - data_2[:,1].max()

    plt.clf()
    plt.plot(data_2[:,0], data_2[:,1], 'b.')
    plt.plot(xp, p1(xp),'m-', label='1nd deg')
    plt.plot(xp, p2(xp),'r-', label='2nd deg')
    plt.plot(xp, p3(xp),'g-', label='3rd deg')
    plt.xlabel("Mean Pixel-Darkening from Colony Growth in 'Kodak Space' (Larger negative number, more stuff on agar)")
    plt.ylabel("Independet Cell Estimate per pixel")
    plt.title(filter_2 + " based conversion to 'Cell Estimate Space'")
    plt.legend(loc=0)
    #plt.xlim(xmin=data_1[:,0].min() - x_span * 0.15, xmax=data_1[:,0].max() + x_span * 0.15)
    #plt.ylim(ymin=data_1[:,1].min() - y_span * 0.15, ymax=data_1[:,1].max() + y_span * 0.15)
    plt.ylim(ymax=2100, ymin=-50)
    plt.xlim(xmax=5, xmin=-100)
    #plt.ylim(ymax=500, ymin=-50)
    #plt.xlim(xmax=5, xmin=-30)
    plt.show()
#GRAPH 2
#print data_1[:,1]
#data_1[:,1] = np.log(np.log(data_1[:,1]))
#print data_1[:,1]

#z1 = curve_fit(np.log,data_1[:,0], data_1[:,1])
#p1 = np.poly1d(z1)

#z2 = np.polyfit(data_1[:,0], data_1[:,1],2)
#p2 = np.poly1d(z2)

#z3 = np.polyfit(data_1[:,0], data_1[:,1],3)
#p3 = np.poly1d(z3)

#xp = np.linspace(data_1[:,0].min(), data_1[:,0].max(), 100)
#x_span = data_1[:,0].max() - data_1[:,0].max()
#y_span = data_1[:,1].max() - data_1[:,1].max()

#plt.clf()
#plt.plot(data_1[:,0], data_1[:,1], 'b.')
#plt.plot(xp, p1(xp),'m-', label='1nd deg')
#plt.plot(xp, p2(xp),'r-', label='2nd deg')
#plt.plot(xp, p3(xp),'g-', label='3rd deg')
#plt.xlabel("Mean Pixel-Darkening from Colony Growth in 'Kodak Space' (Larger negative number, more stuff on agar)")
#plt.ylabel("Log Independet Cell Estimate per pixel")
#plt.title(filter_1 + " based conversion to a logged 'Cell Estimate Space'")
#plt.legend(loc=0)
#plt.xlim(xmin=data_1[:,0].min() - x_span * 0.15, xmax=data_1[:,0].max() + x_span * 0.15)
#plt.ylim(ymin=data_1[:,1].min() - y_span * 0.15, ymax=data_1[:,1].max() + y_span * 0.15)
#plt.ylim(ymax=2100, ymin=-50)
#plt.xlim(xmax=5, xmin=-100)
#plt.show()

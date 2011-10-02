# This may not be an actual python file...

import numpy as np
import matplotlib.pyplot as plt
import random
import data_viewer as dv

def find_distance(a=1.0, b=1.0, P=np.array([2.0, 2.0])):
    """Find the orthogonal distance from P to the line described by:
        y = a*x + b
    """

    # First, offset the line and P by b, so that the line goes though
    # the origin:
  
    P_new = P - np.array([0, b])
    b_new = 0 # this is never used, but the new line is y = a*x + b_new = a*x

    # Construct the normalised vector poining along the offset line (parallel):
    e_par = np.array([1.0, a])
    e_par /= np.sqrt(e_par[0]**2 + e_par[1]**2)
    
    # Find the nomralised vector that is orthogonal (in the right hand sense) to e:
    # Since we have e_par, the simple way to do this is:
    e_ort = np.array([-e_par[1], e_par[0]])

    # It may be instructive, however, to do it from scratch:
    e_ort = np.array([-a, 1.0]) # Not the minus sign!
    e_ort /= np.sqrt(e_ort[0]**2 + e_ort[1]**2)

    # Since e_ort has lenght one, the distance to the new line is the projection
    # of P on e_ort, that is to say: the scalar (or 'dot') product between the two:

    dist = P_new[0]*e_ort[0] + P_new[1]*e_ort[1] # ord np.dot(P, e_ort)
    print dist

    # It may be instructive to write that out:

    dist = (P[0]*(-a) + (P[1] - b)*1.0)/np.sqrt((-a)**2 + 1.0**2)
    print dist

    print "the sign indicates whether the point si above or below the line, with - being above"

    return dist


def find_projection(a=1.0, b=1.0, P=np.array([2.0, 2.0])):
    """Find the point Q on the line described by:
        y = a*x + b
    that best approximates the point P.
    """


    # First, offset the line and P by b, so that the line goes though
    # the origin:
  
    P_new = P - np.array([0, b])
    b_new = 0 # this is never used, but the new line is y = a*x + b_new = a*x

    # Construct the normalised vector poining along the offset line (parallel):
    e_par = np.array([1.0, a])
    e_par /= np.sqrt(e_par[0]**2 + e_par[1]**2)

    # The point Q_new on the new line that best approximates P_new is given by
    # the projection of P_new on e_par, that is to say: the scalar (or 'dot') 
    # product between the two, times e_par:

    Q_new = (P_new[0]*e_par[0] + P_new[1]*e_par[1])*e_par
    # where the first step can be replaced by np.dot(P_new, e_par)

    # Q_new relates to Q by the same offset as the others, so:
    Q = Q_new + np.array([0, b])
    print Q

    # It may be instructive to write that out:

    Q = (P[0]*1.0 + (P[1] - b)*a)*np.array([1.0, a])/(1**2 + a**2)  + np.array([0, b])
    
    print Q

    print "The question for you is 'Where did the sqrt go in the last step?'"  

    return Q


def test(fign=1, clearit=True, a=None, b=None, P=None):
    plt.figure(fign)
    if clearit:
        plt.clf()
    if a == None:
        a = random.uniform(-2, 2)
    if b == None:
        b = random.uniform(-2, 2)
    
    if P == None:
        P = np.array([random.uniform(-3, 3), random.uniform(-3, 3)])

    x = np.linspace(-5, 5)
    y = a*x + b

    Q = find_projection(a, b, P)
    d = find_distance(a, b, P) 
    
    #print a, b, P, Q, d

    plt.plot(x, y, '-r')
    plt.plot(P[0], P[1], 'ob')
    plt.plot(Q[0], Q[1], 'xk')
    plt.plot([P[0], Q[0]], [P[1], Q[1]], 'y--')

    
    return d
    





def test2():
	#Loading data
	D = dv.Data_Object(filename="2011 G-tow toxicity/Data.txt", skip_header=6)
	predicted_line = np.array([[-2.191841795, 0], [0, -2.191841795]])

	#ploting original data
	D.plot_time_series(x_well=4, y_well=5, x_label='', y_label='', name="test", char='.b')
	dv.plt.plot(predicted_line[:,0], predicted_line[:,1], '-r')

	#points vector
	points_vector = D.data[:,4:6]

	#the line info
	line_delta = predicted_line[1,:] - predicted_line[0,:]

	#line's a
	line_factor = line_delta[1] / line_delta[0]

	#line's b
	line_intercept = predicted_line[0,1] - line_factor * predicted_line[0,0]

	tmp_dists = []
	for row in xrange(len(points_vector[:,0])):
		tmp_dists.append(test(clearit=False, a=line_factor, b=line_intercept, P=points_vector[row,:]))

	distances = np.array(tmp_dists)
	SStot = sum((points_vector[:,1] - np.mean(points_vector[:,1]))**2)
	SSerr = sum(distances**2)
	Rsquare = 1 - (SSerr/SStot)

	plt.grid('on')
	plt.axis('equal')
	#plt.show()

	return points_vector, distances, Rsquare


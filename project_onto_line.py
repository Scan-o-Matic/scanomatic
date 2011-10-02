import matplotlib.pyplot as plt
import numpy as np
import types
import data_viewer as dv

#Loading data
D = dv.Data_Object(filename="2011 G-tow toxicity/Data.txt", skip_header=6)
predicted_line = np.array([[-2.191841795, 0], [0, -2.191841795]])

#ploting original data
D.plot_time_series(x_well=4, y_well=5, x_label='', y_label='', name="", char='.b')
dv.plt.plot(predicted_line[:,0], predicted_line[:,1], '-r')

#points vector
points_vector = D.data[:,4:6]


def project_onto_line_trig(points_vector, predicted_line):

	#the line info
	line_delta = predicted_line[1,:] - predicted_line[0,:]
	line_factor = line_delta[1] / line_delta[0]
	line_intercept = predicted_line[0,1] - line_factor * predicted_line[0,0]
	line_cross_x_axis = np.array([(0-line_intercept)/line_factor, 0])
	line_angle = np.arctan(line_factor)


	relative_points = points_vector - line_cross_x_axis
	point_dists = np.sqrt(relative_points[:,0]**2 + relative_points[:,1]**2)
	point_angles = np.arcsin(relative_points[:,1]/point_dists)

	delta_angles = line_angle - point_angles

	points_rel_closest_line_pt = np.array([point_dists * np.cos(point_angles)\
		, point_dists * np.sin(point_angles)])	

	points_on_the_line = np.array([points_rel_closest_line_pt[0,:] * np.cos(line_angle)\
		, points_rel_closest_line_pt[0,:] * np.sin(line_angle)]).transpose()

	points_on_the_line_non_rel_intercept = points_on_the_line + line_cross_x_axis

	dv.plt.plot(points_on_the_line_non_rel_intercept[:,0], points_on_the_line_non_rel_intercept[:,1], '.g')

def project_onto_line(points_vector, line_factor, line_intercept):
	if type(points_vector) == types.TupleType:
		points_vector = [points_vector]

	if type(points_vector) == types.ListType:
		projected_vector = []
		for point in points_vector:
			Q = ((point[0] + line_factor * (point[1] - line_intercept)) / (1 + line_factor**2)) * (1, line_factor)
			projected_vector.append(Q)
		return projected_vector
	else:
		print "points_vector should be a list containing tuples..."
		print "that is to say. It should look like this"
		print "[(0,1), (2,2), (1,3)]"	
		return None

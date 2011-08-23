#!/usr/bin/env python

#Importin what is needed
import sys, os, gtk
from numpy import *
import matplotlib.pyplot as mp

class Data_File():
	def file_loader(self, location=''):
		if location == None:
			loader = gtk.FileChooserDialog(title="Select Bioscreen C - file", action=gtk.FILE_CHOOSER_ACTION_OPEN, \
				buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN, gtk.RESPONSE_OK))

			loader.set_default_response(gtk.RESPONSE_OK)

			response = loader.run()

			if response == gtk.RESPONSE_OK:
				location = loader.get_filename()

			loader.destroy()

		print "Loading the file: " + location

		try:
			fs = open(location,'r')
		except:
			print "Error: Failed to load the file: " + location
			halt()				

		return fs

class Bioscreen_Run(Data_File):
	def __init__(self, file_path = None):

		self.wells = []
		self.source = file_path
		self.times = None
		self.good_measurements = None
		self.plot_figure = 1

		if file_path != None:
			load_from_file(file_path)

	def load_from_file(self, location=None):

		fs = self.file_loader(location=location)
		file_contents = fs.read()
		fs.close()
		file_contents = file_contents.replace("\r", "\n")
		file_contents = file_contents.split("\n")

		if file_contents[0].split(" ")[0] == "READER:":
			no_data = True
			head_row = True
			well_names = []
			values = []

			for line in file_contents:
				line_tuple = line.split("\t")
				if len(line_tuple) > 1:
					if no_data == True:
						if line_tuple[0][:9] == "TenthSec." or line_tuple[0][:4] == "0000":
							no_data = False
	
					if no_data == False:
						if head_row == True:
							if line_tuple[0].strip() == "TenthSec.":
								for item in line_tuple:
									if item.strip() != "":
										well_names.append(item.strip())
							else:
								if well_names == []:
									well_names = range(len(line_tuple))
								head_row = False
	
						if head_row == False:
							row = []
							for item in line_tuple:
								row.append(item.strip())
	
							values.append(row)


			measures_count = len(values)
			for well in well_names:
				self.wells.append(Bioscreen_Well(matrix_size=measures_count, name=well))

			well_count = len(self.wells)
			for well in range(well_count) :
				for measure in range(measures_count):
					try:
						self.wells[well].values[measure] = float(values[measure][well])
					except:
						measures_count = measure			
			measurements = range(len(self.wells[0].values))
			self.good_measurements =  array(measurements) >= 0

	def process_data(self):
		self.useful_range()
		self.smoothen(exclude=[self.wells[0].name])
		self.log(exclude=[self.wells[0].name])
		self.wells[0].time_2_hours()

	def useful_range(self):
		measurements = range(len(self.wells[0].values))
		for well in self.wells:
			if well.name != "TenthSec.":
				for value_pos in measurements:
					if value_pos != -1:
						if well.values[value_pos] != 0:
							measurements[value_pos] = -1
		self.good_measurements =  array(measurements) == -1
					
	def smoothen(self, exclude=[]):
		for well in self.wells:
			if not(well.name in exclude):
				well.smoothen(good_measurements = self.good_measurements)

	def log(self, exclude=[], force_from_raw=False):
		for well in self.wells:
			if not(well.name in exclude):
				well.log(force_from_raw=force_from_raw, good_measurements = self.good_measurements)

	def plot(self, wells=[], logged=True, smoothened=True, plot_cfg_strings=['b.'], legend=None, normalize_start=None, title=None, xaxis=None, yaxis=None):
		x = self.wells[0].values[self.good_measurements]
		y_dimensions = (x.shape[0], len(wells))
		y = zeros(y_dimensions)
		y_pos = 0
		fig = mp.figure(self.plot_figure)
		for well in wells:
			if logged == True and self.wells[well].log2 != None:
				y[:,y_pos] = self.wells[well].log2			
			elif smoothened == True and self.wells[well].smoothened != None:
				y[:,y_pos] = self.wells[well].smoothened
			else:
				y[:,y_pos] = self.wells[well].values

			if normalize_start != None:
				y_diff = y[0,y_pos] - normalize_start
#				print y_diff, y[0, y_pos], y[0, y_pos]-
				y[:,y_pos] -= y_diff

			y_pos += 1
	
		mp.semilogy(x, y[:, 0], plot_cfg_strings[0], basey=2)
		for curve in range(1, len(wells)):
			mp.semilogy(x, y[:, curve], plot_cfg_strings[curve], basey=2)
		if legend != None:
			mp.legend(legend)
		if title != None:
			mp.title(title)
		if xaxis != None:
			mp.xlabel(xaxis)
		if yaxis != None:
			mp.ylabel(yaxis)
		self.plot_figure += 1
		fig.show()

class Bioscreen_Well():
	def __init__(self, matrix_size=None, values=None, name=None, media=None):
		'''
			The Bioscreen Well contains all the measurments from one well.
			It has the functions generally applied to the wells

			A well is initiated with
			@values		A tuple of values
			@@name		A name for the well
			@@media		A media description
		'''

		self.name = name
		self.media = media
		if values != None:
			self.values = array(values, dtype=float64)
		elif matrix_size != None:
			self.values = zeros((matrix_size,), dtype=float64)
		else:
			self.values = None

		self.log2 = None
		self.smoothened = None
		
	def smoothen(self, good_measurements=None):
		no_collapse = zeros(self.values[good_measurements == True].shape, dtype=float64)
		no_collapse[0] = self.values[0]
		for pos in range(no_collapse.shape[0]-1):
			if self.values[pos+1] > no_collapse[pos]:
				no_collapse[pos+1] = self.values[pos+1]
			else:
				no_collapse[pos+1] = no_collapse[pos]

		self.smoothened = zeros(self.values[good_measurements == True].shape, dtype=float64)
		self.smoothened[0] = self.values[0]

		for pos in range(no_collapse.shape[0]-2):
			self.smoothened[pos+1] = mean(no_collapse[pos:pos+3])

		self.smoothened[-1] = self.values[-1]

	def log(self, force_from_raw = False, good_measurements=None):
		if self.smoothened == None or force_from_raw == True:
			self.log2 = log2(self.values[good_measurements])
		else:
			self.log2 = log2(self.smoothened)

	def time_2_hours(self, divisor=36000):
		self.values /= divisor

class Prophecy_Run(Data_File):
	def __init__(self, bioscreen_run=None, duplicate_plates=False):
		self.duplicate_plates = duplicate_plates
		self.wellpattern = {}
		self.phenotype = {'lag': 0, 'rate': 1, 'gt': 1, 'generation time': 1, 'yield': 2, 'efficiency': 2}
		self.data = None

	def log(self):
		self.data = log2(self.data)

	def averages(self):
		if len(self.wellpattern) == 0:
			self.set_well_pattern()

		avgs = zeros((len(self.wellpattern),3))
		i = 0
		for key in self.wellpattern.keys():
			avgs[i,:] = self.well(key, avg=True)
			i += 1
		return avgs

	def well(self, well, avg=False):
		if len(self.wellpattern) == 0:
			self.set_well_pattern()

		try:
			if avg == True:
				return mean(self.data[self.wellpattern[well],:], axis=0)
			else:
				return self.data[self.wellpattern[well],:]
		except:
			return None

	def phenotypes(self, phenotype):
		try:
			return self.data[:,self.phenotype[phenotype.lower()]]
		except:
			return None

	def set_well_pattern(self, duplicate_plates=None, well_pattern=None):
		self.wellpattern = {}
		if duplicate_plates != None:
			self.set_duplicate_plates(value=duplicate_plates)
		if well_pattern == None:
			fs = self.file_loader(location=None)
			i = 0
			for line in fs:
				if line[0] != "#":
					tmp = line.split('\t')
					if len(tmp) > 1:
						try:
							if self.duplicate_plates == True:
								self.wellpattern[tmp[1]] = [int(tmp[0]), int(tmp[0])+100]
							else:
								self.wellpattern[tmp[1]] = [int(tmp[0])]
						except:
							if self.duplicate_plates == True:
								self.wellpattern[tmp[1]] = [i, i+100]
							else:
								self.wellpattern[tmp[1]] = [i]
					else:
						self.wellpattern[tmp[pos]] = [i] 
					i+=1	

			fs.close()
		else:
			i = 0
			for well in well_pattern:
				if self.duplicate_plates == True:
					self.wellpattern[well] = [i, i+100]
				else:
					self.wellpattern[well] = [i]
				i += 1

	def set_duplicate_plates(self, value=True):
		self.duplicate_plates = value

	def get_duplicate_plates(self):
		return self.duplicate_plates

	def load_from_file(self, location=None):
		fs = self.file_loader(location=location)
		data = []
		for line in fs:
			tmp = line.split('\t')
			heading_row = False
			for i in range(len(tmp)):
				if i == 0:
					try:
						tmp[i] = int(tmp[i].strip())
					except:
						print "Assuming this i a heading row:\n" , tmp
						heading_row = True
				elif heading_row == False:
					try:
						tmp[i] = float(tmp[i].strip())
					except:
						print "Unexpected data: " + tmp[i]
						halt()

			if not(heading_row):
				data.append(tmp[1:len(tmp)])

		self.data = array(data, dtype = float64)
		fs.close()


#
# SIMPLE PROPHECY FILE READING, MAKING AVERAGES, LOG2 TRANSFORMATIONS, ETC
#

#prophecy = Prophecy_Run()
#prophecy.load_from_file(location='2011 Deadaptation/Prophecy phenotypes/MZCS4001.txt')
#prophecy.set_well_pattern(duplicate_plates=True, well_pattern=range(1,101)) #Call experiments 1-100 and say wells 101-200 are dupes
#prophecy.log()
#print prophecy.averages()

#
# SIMPLE READING BIOSCREEN FILE, STANDARD SMOOTHING AND LOG2, AND FINALLY PLOTTING WELLS 2 and 102
#

#bioscreen = Bioscreen_Run()
#bioscreen.load_from_file(location='2011 Salt tolerance of diff spices/MYSC/MZGS9929.XL~') #location='/home/bambiraptor/Documents/PhD/Data/2011 Deadaptation/Bioscreen files/MZCS4000.XL~')
#bioscreen.process_data()
#bioscreen.plot(wells=[25,125,33,133,82,182], logged=False, plot_cfg_strings=['b.','b.','r.','r.','g.','g.'], normalize_start=0.11, legend=('MYSC25 #1','MYSC25 #2','MYSC32 #1', 'MYSC32 #2','MYSC82 #1','MYSC82 #2'), xaxis='hours', yaxis='OD', title='Debaryomyces hansenii: slow-GT examples')
#bioscreen.plot(wells=[26,126,83,183,31,131], logged=False, plot_cfg_strings=['b.','b.','r.','r.','g.','g.'], normalize_start=0.11, legend=('MYSC26 #1','MYSC26 #2','MYSC83 #1', 'MYSC83 #2','MYSC30 #1','MYSC30 #2'), xaxis='hours', yaxis='OD', title='Debaryomyces hansenii: fast-GT examples')
#bioscreen.plot(wells=[24,124,84,184], logged=False, plot_cfg_strings=['b.','b.','r.','r.'], normalize_start=0.11, legend=('MYSC24 #1','MYSC24 #2','MYSC84 #1', 'MYSC84 #2'), xaxis='hours', yaxis='OD', title='Debaryomyces hansenii: mid-GT examples')
pass


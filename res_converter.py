#!/usr/bin/env python

import sys, os, re

class Output_File():
	def __init__(self, input_file=None, encoding="utf-8"):

		self.__input_file__ = input_file
		self.__file_contents__ = None
		self.__cuvettes__ = {}
		self.__verified__ = False
		self.__imported__ = False

		if not self.__input_file__ == None:
			try:
				fs = open(self.__input_file__, 'r') #, encoding)
			except:
				print "Unable to open file, ", input_file
				sys.exit(-2)

			self.__file_contents__ = fs.read()

			fs.close()
		else:
			print "To load a file, the file must be specified!"
			sys.exit(-1)

		if self.__file_contents__ != None:
			if self.populate_cuvettes():
				self.__imported__ = True
			else:
				print "File is not a .res file!"
				sys.exit(-4)
		else:
			print "File was empty..."
			sys.exit(-3)


	def verify_type(self):
		if not self.__file_contents__ == None:
			cuv_split = re.split("[Cc][Uu][Vv]\=[0-9][0-9][0-9]", self.__file_contets__)
			if len(cuv_spilt) != len(self.__file_contents__) and len(cuv_split) > 1:
				return True
			else:
				return False
		else:
			return False

	def populate_cuvettes(self):
		if not self.__verified__ or self.verify_type():
			cuv_split = re.split("[Cc][Uu][Vv]\=[0-9][0-9][0-9]", self.__file_contents__)
			for c_index, cuvette in enumerate(cuv_split):
				cuvette = cuvette.replace("\n", "")
				cuvette = cuvette.replace("\n", "")
				for m_index, measurement in enumerate(cuvette.split(" ")):
					if len(measurement) > 4:
						measurement = measurement[0:4]
					if len(measurement) == 4:
						try:
							self.__cuvettes__[c_index].append(float(measurement))
						except KeyError:
							self.__cuvettes__[c_index] = [float(measurement)]
			return True
		else:
			return False

	def make_output(self, filename=None, file_ending=".csv"):
		if self.__imported__:	
			if filename == None:
				filename = str(self.__input_file__.split(".",1)[0]) + str(file_ending)

			try:
				fs = open(filename, "w")
			except:
				print "Unable to output to file:", filename
				sys.exit(-10)

			well_keys = self.__cuvettes__.keys()
			well_keys.sort()

			time = 0
			time_step = 12000
			eol = "\n"
			sep = chr(9)

			fs.write("READER:    Bioscreen C (200 wells)" + eol)
			fs.write("TEST NAME: unknown" + eol)
 			fs.write(eol)
                        fs.write("                               W E L L S   1 - 200" + eol)
			fs.write("                                     OD  values" + eol)
 			fs.write(eol)

			fs.write("TenthSec." + sep)
			for well in well_keys:
				fs.write(str(well))
				if well != well_keys[-1]:
					fs.write(sep)
			fs.write("\n")

			for row_index in xrange(len(self.__cuvettes__[well_keys[0]])):
				fs.write(str(time) + sep)
				for well in well_keys:
					fs.write(str(self.__cuvettes__[well][row_index]))
					if well != well_keys[-1]:
						fs.write(sep)

				fs.write(eol)
				time += time_step

			fs.close()

if len(sys.argv) > 1 and sys.argv[1] != "-h":
	output_file = Output_File(input_file=sys.argv[1])
	if len(sys.argv) > 2:
		output_file.make_output(filename=sys.argv[2])
	else:
		output_file.make_output()
else:
	print "You need to run this script with the file you want to convert as argument"
	print "COMMAND: res_converter [FILE]"
	sys.exit(0)

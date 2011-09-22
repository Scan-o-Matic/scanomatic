import matplotlib.pyplot as plt
import numpy as np


class Data_Object():
    def __init__(self, filename="MYSC0006_cleaned.XL~", skip_header=6):
        self.filename = filename
        try:
            self.data = np.genfromtxt(self.filename, skip_header)
        except IOError:
            print "Invalid filename: " + self.filename
            print "Reload using Data_object.load(filename) or create new object."
        self.msize = 9
        self.lwidth = (np.sqrt(5)+1)/2

    def load(self, filename="MYSC0006_cleaned.XL~"):
        self.filename = filename
        try:
            self.data = np.genfromtxt(self.filename, skip_header=6)
        except IOError:
            print "Invalid filename: " + self.filename

    def plot_time_series(self, x_well=0, y_well=1, x_label='t', y_label='OD',
            name="N/A", fign=1, clearit=True, window=0): 
        # Preliminaries:
        plt.figure(fign) 
        if clearit:
            plt.clf() 
    
        # Prepare data:
        x_data = self.data[:, x_well]
        y_data = self.data[:, y_well]    
        if window > 1:
            x_data = self.smoothme(x_data, window)
            y_data = self.smoothme(y_data, window)

        # Plot:
        plt.plot(x_data, y_data, '-b', label=name,
            ms=self.msize, lw=self.lwidth, mew=self.lwidth)

        plt.legend(numpoints=1, loc=4, ncol=1)
        plt.grid('on')
        plt.axis('auto')
        plt.xlabel(x_label, fontsize='xx-large')
        plt.ylabel(y_label, fontsize='xx-large')
        plt.title('Time series', fontsize='xx-large')

    def plot_autodifference(self, x_well=0, y_well=1, x_label='t', y_label='OD',
            name="N/A", fign=1, clearit=True, window=10, plotmax=True, plotmins=True): 
        # Preliminaries:
        plt.figure(fign) 
        if clearit:
            plt.clf() 
         
        # Prepare data:
        x_data = self.data[:, x_well]
        y_data = self.data[:, y_well]    
        if window > 1:
            x_data = self.smoothme(x_data, window)
            y_data = self.smoothme(y_data, window)
        # Calculate simple autodifference:
        autodiff  = y_data[1:] - y_data[:-1]
        # Compensate for uneven $\Delta t$:
        autodiff /= x_data[1:] - x_data[:-1]
        # Find point of maximum slope: 
        max_i = np.argmax(autodiff[1:]) + 1#np.argmax(autodiff[window:]) + window
        max_x = x_data[max_i]
        max_y = autodiff[max_i]
	left_min_i = np.argmin(autodiff[1:max_i]) +1
        left_min_x = x_data[left_min_i]
        left_min_y = autodiff[left_min_i]
	right_min_i = np.argmin(autodiff[max_i:]) + max_i
        right_min_x = x_data[right_min_i]
        right_min_y = autodiff[right_min_i]

        # Plot:
        plt.plot(x_data[0:-1], autodiff, '-r', label=name,
            ms=self.msize, lw=self.lwidth, mew=self.lwidth)
        if plotmax:
            plt.plot(max_x, max_y, 'kx', label="max("+name+")",
                ms=self.msize, lw=self.lwidth, mew=self.lwidth)

        if plotmins:
            plt.plot(left_min_x, left_min_y, 'kx', label="max("+name+")",
                ms=self.msize, lw=self.lwidth, mew=self.lwidth)
            plt.plot(right_min_x, right_min_y, 'kx', label="max("+name+")",
                ms=self.msize, lw=self.lwidth, mew=self.lwidth)

        plt.legend(numpoints=1, loc=1, ncol=1)
        plt.grid('on')
        plt.axis('auto')
        plt.xlabel(x_label, fontsize='xx-large')
        plt.ylabel(y_label, fontsize='xx-large')
        plt.title('Auto-difference', fontsize='xx-large')

        return max_x, max_y, max_i

    def smoothme(self, data, window):
        new_data = []
        for i in xrange(len(data[:-window])):
            new_data.append(np.mean(data[i:i + window]))

        return np.array(new_data)

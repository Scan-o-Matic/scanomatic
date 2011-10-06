import matplotlib.pyplot as plt
import numpy as np


class Data_Object():
    def __init__(self, filename="MYSC0006_cleaned.XL~", skip_header=6, delimiter=None):
        self.filename = filename
        try:
            self.data = np.genfromtxt(self.filename, skip_header=skip_header, delimiter=delimiter)
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

    def make_figure_normal(self, y_well, clearit=True, fign=1, window=20):
        self.plot_time_series(y_well=y_well, clearit=clearit, fign=fign, do_plot=True)
        max_x, max_y, max_i = self.plot_autodifference(y_well=y_well, window=window, do_plot=False, plotmins=False, clearit=False)
        self.add_slope_line_from_pt(int(max_x/(self.data[2,0]-self.data[1,0])), y_col=y_well, slope=max_y)

    def plot_time_series(self, x_well=0, y_well=1, x_label='t', y_label='OD',
            name="N/A", fign=1, clearit=True, window=0, char='-b', do_plot=False): 
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

        if do_plot:
               # Plot:
               plt.plot(x_data, y_data, char, label=name,
                   ms=self.msize, lw=self.lwidth, mew=self.lwidth)

               plt.legend(numpoints=1, loc=4, ncol=1)
               plt.grid('on')
               plt.axis('auto')
               plt.xlabel(x_label, fontsize='xx-large')
               plt.ylabel(y_label, fontsize='xx-large')
               plt.title('Time series', fontsize='xx-large')

    def plot_autodifference(self, x_well=0, y_well=1, x_label='t', y_label='OD',
            name="N/A", fign=1, clearit=True, window=10, plotmax=True, plotmins=True, \
            do_plot=True): 

        if do_plot:
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
        if plotmins:
            left_min_i = np.argmin(autodiff[1:max_i]) +1
            left_min_x = x_data[left_min_i]
            left_min_y = autodiff[left_min_i]
            right_min_i = np.argmin(autodiff[max_i:]) + max_i
            right_min_x = x_data[right_min_i]
            right_min_y = autodiff[right_min_i]

        if do_plot:
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

    def add_slope_line_from_pt(self, measurement, y_col, x_col=0, slope=None):
        if slope == None:
              slope = (self.data[measurement+1, y_col] - self.data[measurement-1, y_col])\
                    / (self.data[measurement+1, x_col] - self.data[measurement-1, x_col])

        # y = ax + b    ->  b = y - ax
        intercept = self.data[measurement, y_col] - slope * self.data[measurement, x_col]
         

        self.add_slope_line(self.data[measurement, x_col], slope, intercept)

    def add_slope_line(self, x, a, b, x_window=None, line_width=1.2, char="--r"):
        if x_window == None:
            #y = ax + b, y = 0 -> x = -b/a
            x_window = x - (-b/a)
        xs = np.array([x-x_window, x+x_window])
        print xs, a, b, x, a*x+b
        plt.plot(xs, a*xs+b, char, lw=line_width)
       

    def smoothme(self, data, window):
        new_data = []
        for i in xrange(len(data[:-window])):
            new_data.append(np.mean(data[i:i + window]))

        return np.array(new_data)

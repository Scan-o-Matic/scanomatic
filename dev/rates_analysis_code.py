#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt_lab
import logging
import scipy.interpolate as scint

def make_linear_interpolation(data):
    y, x = np.where(data != 0)
    Y, X = np.arange(data.shape[0]), np.arange(data.shape[1])
    logging.warning("ready for linear")
    f = scint.interp2d(x, y, data[y, x], 'linear')
    logging.warning("done linear interpol")
    return f(X,Y)

def make_linear_interpolation2(data):
    empty_poses = np.where(data == 0)
    f = data.copy()
    for i in xrange(len(empty_poses[0])):
        pos = [p[i] for p in empty_poses]
        a_pos = map(lambda x: x-1, pos)    
        a_pos = np.array(a_pos)
        a_pos[np.where(a_pos < 0)] = 0    
        b_pos = map(lambda x: x+1, pos)
        b_pos = np.array(b_pos)
        logging.warning("Pos {0}, a_pos {1}, b_pos {2}, data.shape {3}"\
            .format(pos, a_pos, b_pos, data.shape))
        b_pos[np.where(b_pos > np.array(data.shape))] -= 1
        cell = data[a_pos[0]:b_pos[0], a_pos[1]:b_pos[1]]
        f[tuple(pos)] = cell[np.where(cell != 0)].mean()
        logging.warning("Corrected position {0} to {1}".format(\
            pos, f[tuple(pos)]))
    return f

def make_cubic_interpolation(data):
    y, x = np.where(data !=0)
    Y, X = np.mgrid[0: data.shape[0], 0:data.shape[1]]
    f = scint.griddata((x,y), data[y,x], (X, Y), 'cubic')
    logging.warning('Done some shit')
    nans = np.where(np.isnan(f))
    logging.warning('Got {0} nans after cubic'.format(len(nans[0])))
    f[nans] = 0
    f2 = make_linear_interpolation2(f)
    logging.warning('Linear done')
    f[nans] = f2[nans]
    return f

#NEWEST PRECOG RATES
def get_empty_data_structure(plate, max_row, max_column):
    d = []
    for p in range(plate+1):
        d.append(np.ones((max_row[p]+1, max_column[p]+1))*48)
    data = np.array(d)
    return data


#USING 1/4th AS NORMALISING GRID
def make_norm_surface(data, sigma=3, only_linear=False):
    norm_surfaces = []
    for p in xrange(data.shape[0]):
        exp_pp = map(lambda x: x/2, data[p].shape)
        X, Y = np.mgrid[0:exp_pp[0],0:exp_pp[1]]*2
        norm_surface = np.zeros(data[p].shape)
        norm_surface[X, Y] = data[p][X, Y]
        m = norm_surface[X, Y].mean()#.mean(1)
        sd = norm_surface[X, Y].std()#.std(1)
        logging.warning('High-bound outliers ({0}) on plate {1}'.format(\
            len(np.where(m + sigma*sd < norm_surface[X,Y])[0]), p))
        norm_surface[np.where(m + sigma*sd <\
            norm_surface[X,Y])] = 0
        logging.warning('Low-bound outliers ({0}) on plate {1}'.format(\
            len(np.where(m - sigma*sd > norm_surface[X,Y])[0]), p))
        norm_surface[np.where(m - sigma*sd >
            norm_surface[X,Y])] = 0
        if only_linear:
            norm_surface = \
                make_linear_interpolation(norm_surface)
        else:
            norm_surface = \
                make_cubic_interpolation(norm_surface)
        norm_surfaces.append(norm_surface)
        logging.warning('Going for next plate')
    return np.array(norm_surfaces)


#NEW Luciano-format (And newest)
fs = open("analysis_GT.csv", 'r')

old_row = -1
max_row = [-1,-1,-1,-1]
max_column = [-1,-1,-1,-1]
plate = 0
data_store = {}
meta_data_store = {}

for line in fs:
    try:
        position, value, data = line.split("\t",2)
        plate, row_column = map(lambda x: x.strip(":"), 
            position.split(" "))
        row, column = row_column.split("-")
        plate, row, column = map(int, (plate, row, column))
        if max_row[plate] < row:
            max_row[plate] = row
        if max_column[plate] < column:
            max_column[plate] = column
        value = float(value.strip())
        data = data.strip()[1:-1]
        data = data.split(",")
        data = map(lambda x: x.split("-"), data)
        data_rates = map(float, [d[0] for d in data])
        data_times = map(float, [d[1] for d in data])
        data_store[(plate, row, column)] = value
        meta_data_store[(plate, row, column)] = (data_rates, data_times)
    except:
        try:
            position, data = line.split(" ", 1)
            row, column = map(int, position.split("-"))
            if row < old_row:
                plate += 1
            data = eval(data)
            if len(data) == 5:
                data_store[(plate, row, column)] = data
                if row > max_row:
                    max_row = row
                if column > max_column:
                    max_column = column
            old_row = row
        except:
            logging.warning("BAD data row: '%s'" % line)
        


fs.close()    

#NEW WAY
data = np.ones((plate+1, max_row+1, max_column+1, 5)) * 48

for loc, measures in data_store.items():

    data[loc] = np.array(measures)

data = get_empty_data_structure(plate, max_row, max_column)

for loc, measures in data_store.items():

    data[loc] = np.array(measures)

#NEWEST TESTS
data = get_empty_data_structure(plate, max_row, max_column)
dubious = 0
dubious2 = 0
st_pt = 0
ln_pts = 7
side_buff = 1
for loc, measures in meta_data_store.items():

    times = np.array(measures[0])
    OD = np.array(measures[1])
    f_OD = OD[st_pt:ln_pts][times[st_pt:ln_pts].argsort()[-1 - side_buff]] / \
        OD[st_pt:ln_pts][times[st_pt:ln_pts].argsort()[side_buff]]
    d_t = times[times[st_pt:ln_pts].argsort()[-1 - side_buff]] -\
        times[times[st_pt:ln_pts].argsort()[side_buff]]
    data[loc[0]][loc[1:]] = d_t * np.log(2) / np.log2(f_OD)
    if times[st_pt:ln_pts].min() in times[st_pt:ln_pts][-2:] and\
        times[st_pt:ln_pts].max() in times[st_pt:ln_pts][-2:]:

        logging.warning("Encountered dubious thing".format(d_t, f_OD))
        dubious += 1
    if times[st_pt:ln_pts].min() in times[st_pt:ln_pts][:2] or\
        times[st_pt:ln_pts].max() in times[st_pt:ln_pts][:2]:

        dubious2 += 1

plate_texts = {0: '30mL', 1: '50mL', 2: '70mL', 3: '80mL'}
plate_texts = {0: 'Plate A', 1: 'Plate B', 2: 'Plate C', 3: 'Plate D'}


#PLOT A SIMPLE HEAT MAP
fig = plt.figure()
for p in xrange(data.shape[0]):
    ax = fig.add_subplot(2,2,p+1, title="Rates from edge points (5 v2) Plate {0}".format( plate_texts[p]))
    plt.imshow(data[p], vmin=0, vmax=5)
    plt.colorbar(ax = ax, orientation='horizontal')

fig.show()
#PLOT DONE





#CREATING NOROM TARGET ARRAY
data2 = np.zeros(data.shape, dtype=np.float64)

for p in range( data.shape[0] ):
    for x in range( exp_pp[0] ):
        for y in range( exp_pp[1] ):

            data2[p, x*2, y*2] = data[p, x*2, y*2]

data3 = data2.copy()
#PASS 1
for p in range( data.shape[0]):
    for x in range( exp_pp[0]):
        for y in range(exp_pp[1]):
            try:
                cell = data2[p, x*2: x*2+3, y*2: y*2+3]            
            except:
                try:
                    cell = data2[p, x*2: x*2+2, y*2: y*2+3] 
                except:
                    cell = data2[p, x*2: x*2+2, y*2: y*2+2]                       
            data3[p, x*2+1, y*2+1] = cell[np.where(cell != 0)].mean()
#PASS 2

#DO NORM

data5 = data - data4

norm_surface = make_norm_surface(data)
data5 = data - norm_surface

exp_filter = np.ones((2,2), dtype=bool)
exp_filter[0,0] = 0
norm_grid = (exp_filter == False)
exp_pp = map(lambda x: map(lambda y: y/2, x.shape), data)
e_mean = []
e_sd = []
e_max = 0
e_min = 0
e_sd_max = 0
e_sd_min = 0
#CALC EXPERIMENTS
for p in range( data.shape[0] ):
    exp_mean = np.zeros(exp_pp[p], dtype=np.float64)
    exp_sd = np.zeros(exp_pp[p], dtype=np.float64)
    for x in range( exp_pp[p][0]):
        for y in range( exp_pp[p][1]):
            cell = data5[p][x*2:x*2+2, y*2:y*2+2]
            exp_mean[x,y] = cell[np.where(exp_filter)].mean()
            exp_sd[x,y] = cell[np.where(exp_filter)].std()
        logging.warning("Plate {0}, row {1}".format(p, x))
    if e_max < exp_mean.max():
        e_max = exp_mean.max()
    if e_min > exp_mean.min():
        e_min = exp_mean.min()
    if e_sd_max < exp_sd.max():
        e_sd_max = exp_sd.max()
    if e_sd_min > exp_sd.min():
        e_sd_min = exp_sd.min()

    e_mean.append(exp_mean)
    e_sd.append(exp_sd)


#PLOT A SIMPLE HEAT MAP
fig = plt.figure()
for p in xrange(data.shape[0]):
    ax = fig.add_subplot(2,2, p+1, title="Rate {0}".format( plate_texts[p]))
    plt.imshow(e_mean[p], vmin=e_min, 
        vmax=e_max)
    plt.colorbar(ax = ax, orientation='horizontal')

fig.savefig("./exp_norm.png")



fig = plt.figure()
for p in xrange(data.shape[0]):
    ax = fig.add_subplot(4,2,2*p+1, title="Rate {0}".format( plate_texts[p]))
    plt.imshow(e_mean[p], vmin=e_min, 
        vmax=e_max)
    plt.colorbar(ax = ax, orientation='horizontal')
    ax = fig.add_subplot(4,2,2*p+2, title="Standard dev {0}".format( plate_texts[p]))
    plt.imshow(e_sd[p], vmin=e_sd_min, vmax=e_sd_max)
    plt.colorbar(ax = ax, orientation='horizontal')

fig.savefig("./exp_norm_mean_sd.png")
#PLOT DONE

for p in xrange(data.shape[0]):

    print "{0} has mean {1} ({2}) and std {3}".format(\
        plate_texts[p], e_mean[p].mean(), 
            e_mean[p].mean() + norm_surface[p].mean(),
            e_mean[p].std())




#OLD STUFF
#

#LOADING DATA
X = plt_lab.csv2rec("analysis_GT.csv", skiprows=1, delimiter="\t")

X_list = []

for x in X:
    X_list.append([x[1],x[3],x[5],x[7]])


Y = np.asarray(X_list)
Y2 = np.reshape(Y, (16, 24,4))
#DATA LOADED

#CLEAR OUT OUT-LIERS
Y2[np.where(Y2[:,:,:]>10)] = np.nan

#ALTERNATIVE DATA
#X = plt_lab.csv2rec("analysis_locked_baseline.csv", skiprows=1, delimiter="\t")
#X_list = []
#for x in X:
#    X_list.append([x[2],x[5],x[8],x[11]])

#Z = np.asarray(X_list)
#Z2 = np.reshape(Y, (16, 24,4))
#ALTERNATIVE DATA LOADED

#PLOT A SIMPLE HEAT MAP
fig = plt.figure()
for p in xrange(Y2.shape[2]):
    fig.add_subplot(2,2,p+1, title="Plate %d" % p)
    plt.imshow(Y2[:,:,p])

fig.show()
#PLOT DONE

#PLOT HISTOGRAM
fig = plt.figure()
plt.clf()
plt.hist(Y2[np.where(Y2[:,:,0]>0)].ravel(), bins=25)
plt.show()
#PLOT DONE

#PLOTTING DATA VS ALTERNATIVE DATA...
plt.clf()
for i in xrange(Y2.shape[2]):
    plt.plot(Y2[:,:,i].ravel(), Z2[:,:,i].ravel(), '.', label="Plate %d" % i)


plt.xlabel('Rate')
plt.ylabel('Lowest value')
plt.legend()
plt.show()
#PLOT DONE

#NEIGHBOUR COUNT ARRAY
N = Y2.copy()
N[:,:,:] = 4 + 4*(1/np.sqrt(2))
N[0,:,:] = 3 + 2*(1/np.sqrt(2))
N[-1,:,:] = 3 + 2*(1/np.sqrt(2))
N[:,0,:] = 3 + 2*(1/np.sqrt(2))
N[:,-1,:] = 3 + 2*(1/np.sqrt(2))
N[0,0,:] = 2 + (1/np.sqrt(2))
N[-1,0,:] = 2 + (1/np.sqrt(2))
N[-1,-1,:] = 2 + (1/np.sqrt(2))
N[0,-1,:] = 2 + (1/np.sqrt(2))

#THE NUMBER OF NEIGHBOURS PER TYPE
neighbours = [2 + (1/np.sqrt(2)), 3 + 2*(1/np.sqrt(2)), 4 + 4*(1/np.sqrt(2))]

#MAKE NEIGHBOURDEPENDENT PLOT
plt.clf()
fig = plt.figure()

for i in xrange(N.shape[2]):
    fig.add_subplot(2,2,i+1, title="Plate %d" % i)
    plt.boxplot([Y2[np.where(N[:,:,i] == neighbours[0])].ravel(), 
        Y2[np.where(N[:,:,i] == neighbours[1])].ravel(),
        Y2[np.where(N[:,:,i] == neighbours[2])].ravel()],
        positions=neighbours)
    plt.ylabel('Rate')
    plt.xlabel('Neighbours')


fig.show()
#PLOT END

#POSITIONAL EFFECT

Y2[np.where(np.isnan(Y2[:,:,:]))] = 0

#STENCIL A
#n = 4 + 4 / np.sqrt(2)
#kernel = np.asarray([[-1/np.sqrt(2), -1.0,-1/np.sqrt(2)],[-1, n,-1],[-1/np.sqrt(2), -1, -1/np.sqrt(2)]])
#STENCIL B
n=6
kernel = np.asarray([[-1/2, -1.0,-1/2],[-1, n,-1],[-1/2, -1, -1/2]])
N2 = []
for i in xrange(Y2.shape[2]):
    N2.append(fftconvolve(kernel, Y2[:,:,i], mode='full'))


#PLOT POSITIONAL EFFECT
fig = plt.figure()
for p in xrange(Y2.shape[2]):
    fig.add_subplot(2,2,p+1, title="Plate %d" % p)
    plt.imshow(N2[p])
    plt.colorbar()

fig.show()


Y3 = 1 / Y2

#STENCILS
kernel = np.asarray([[0.5, 1.0, 0.5],
                    [1.0, 0, 1.0],
                    [0.5, 1.0, 0.5]])

kernel = np.asarray([[1/np.sqrt(2), 1.0,1/np.sqrt(2)],
                    [1.0, 0, 1.0],
                    [1/np.sqrt(2), 1.0, 1/np.sqrt(2)]])

def vector_orth_dist(x, y, p1):
    """
    @param x: is a vector of X-measures
    @param y: is a vector of corresponding Y-measures
    @param p1: is a polynomial returned from numpy.poly1d
    """
    #
    #get the point where ax + b = 0
    x_off = -p1[0]/p1[1]
    #
    #p's unit vector creation
    p_unit = np.asarray((1-x_off,p1(1)))
    p_unit = p_unit / np.sqrt(np.sum(p_unit**2))
    #
    #its orthogonal vector
    p_u_orth = np.asarray((-p_unit[1], p_unit[0]))
    #
    #distances:
    dists = np.zeros(x.shape)
    for d in xrange(dists.size):
        dists[d] = np.sum(np.asarray((x[d]-x_off,y[d]))* p_u_orth)
    #
    return dists

#PLOT THE POSITIONAL EFFECT
plt.clf()
fig = plt.figure()
Ps = []
Ns = []

for p in xrange(Y2.shape[2]):
    C = c2(Y2[:,:,p], kernel, mode='nearest').ravel() / np.sum(kernel)
    z1  = np.polyfit(C, Y2[:,:,p].ravel(),1)
    Ps.append(z1)
    p1 = np.poly1d(z1)
    l = np.linspace(C.min(), C.max(), num=100)
    Ns.append( vector_orth_dist(C, Y2[:,:,p].ravel(),p1) + np.mean(Y2[:,:,p]))
    fig.add_subplot(2,2,p+1, title="Plate %d" % p)
    plt.plot(C, Y2[:,:,p].ravel(), 'b.')
    plt.plot(l, p1(l), 'r-')
    plt.gca().annotate(s=str("%.3f" % z1[0]) + "x + " + str("%.3f" % z1[1]), 
        xy=(l[5]+0.6,p1(l[5])+0.3))
    plt.xlim(1.4, 4)
    plt.ylim(1, 4.5)
    plt.ylabel('Colony Generation Time')
    plt.xlabel('How cool your neighbours are (Very <-> Not)')

fig.show()

#PLOT SECONDARY POSITIONAL EFFECTS
fig = plt.figure()
Ns2 = []

for p in xrange(Y2.shape[2]):
    C = c2(np.reshape(Ns[p], (16,24)), kernel, mode='nearest').ravel() / np.sum(kernel)
    z1  = np.polyfit(C, Ns[p],1)
    p1 = np.poly1d(z1)
    l = np.linspace(C.min(), C.max(), num=100)
    Ns2.append( vector_orth_dist(C, Ns[p], p1) + np.mean(Ns[p]))
    fig.add_subplot(2,2,p+1, title="Plate %d" % p)
    plt.plot(C, Ns[p], 'b.')
    plt.plot(l, p1(l), 'r-')
    plt.gca().annotate(s=str("%.3f" % z1[0]) + "x + " + str("%.3f" % z1[1]),
        xy=(l[5],p1(l[5])+0.6))
    plt.xlim(2.0, 3.5)
    plt.ylim(1.5, 4.0)
    plt.ylabel('Colony Generation Time')
    plt.xlabel('How cool your neighbours are (Very <-> Not)')

plt.show()

#PLOT THE PROCESSED VS RAW HEAT MAPS
plt.clf()
fig = plt.figure()

for i in xrange(N.shape[2]):
    fig.add_subplot(2,2,i+1, title="Plate %d" % i)
    plt.boxplot([Y2[:,:,i].ravel(),
        Ns[i]])
    plt.annotate(s="std: %.3f" % np.std(Y2[:,:,i].ravel()), xy=(1,4) )
    plt.annotate(s="std: %.3f" % np.std(Ns[i]), xy=(2,4) )
    plt.ylabel('Generation Time')
    plt.xlabel('Untreated vs Normalized')
    plt.ylim(1.0,4.5)

fig.show()

#UNUSED
fig = plt.figure()

for p in xrange(Y2.shape[2]):
    fig.add_subplot(2,2,p+1, title="Plate %d" % p)
    plt.plot(c2(Y3[:,:,p], kernel, mode='constant', cval=np.mean(Y3[:,:,p]))\
        .ravel(), Y2[:,:,p].ravel(), 'b.')
    plt.ylabel('Colony rate')
    plt.xlabel('How cool your neighbours are (Not <-> Very)')

fig.show()

#PLOT START EFFECT
fig = plt.figure()

for p in xrange(Y2.shape[2]):
    fig.add_subplot(2,2,p+1, title="Plate %d" % p)
    plt.plot(Z2[:,:,p].ravel(), Ns[p], 'b.')
    plt.ylabel('Colony rate')
    plt.xlabel('Start-value')
    plt.xlim(0,200000)
    plt.ylim(0,4)

plt.show()

#
#
#
fs = open('slow_fast_test.csv', 'w')
y_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
yeasts = ['wt','Hog1']
fig = plt.figure()
for p in xrange(4):
    fs.write("\nPlate " + str(p+1) + "\n")
    fs.write("Row\tCol\tType\n")
    p_layout = np.random.random((8,12))>0.8
    for x in xrange(p_layout.shape[0]):
        for y in xrange(p_layout.shape[1]):
            fs.write(y_labels[x] + "\t" + str(y+1) + "\t" + yeasts[p_layout[x,y]] + "\n")
    fig.add_subplot(2,2,p+1)
    plt.imshow(p_layout, aspect='equal', interpolation='nearest')
    plt.xticks(np.arange(12), np.arange(12)+1)
    plt.yticks(np.arange(8), y_labels)
    plt.ylabel('Columns')
    plt.xlabel('Rows')

fs.close()
plt.show()

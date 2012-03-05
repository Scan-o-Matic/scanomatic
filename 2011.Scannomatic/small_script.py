#LOADING DATA
X = plt_lab.csv2rec("analysis_locked_GT.csv", skiprows=1, delimiter="\Z2 t")

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

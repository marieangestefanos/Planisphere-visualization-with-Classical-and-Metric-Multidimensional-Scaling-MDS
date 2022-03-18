import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import sqrtm, eig
from sklearn.manifold import MDS

## Disconnected graph
"""
from scipy.spatial.distance import pdist, squareform
from matplotlib.collections import LineCollection

np.random.seed(10)
N = 10
X = np.random.rand(N,2)
k = 3

# matrix of pairwise Euclidean distances
distmat = squareform(pdist(X, 'euclidean'))

# select the kNN for each datapoint
neighbors = np.sort(np.argsort(distmat, axis=1)[:, 0:k])

# get edge coordinates
coordinates = np.zeros((N, k, 2, 2))
for i in np.arange(N):
    for j in np.arange(k):
        coordinates[i, j, :, 0] = np.array([X[i,:][0], X[neighbors[i, j], :][0]])
        coordinates[i, j, :, 1] = np.array([X[i,:][1], X[neighbors[i, j], :][1]])

# create line artists
lines = LineCollection(coordinates.reshape((N*k, 2, 2)), color='black')

fig, ax = plt.subplots(1,1,figsize = (8, 8))
ax.scatter(X[:,0], X[:,1], c = 'black')
ax.add_artist(lines)
plt.show()
"""

## Load the data given the path of the dataset (.csv file)
def load_data(path):
    data = pd.read_csv(path, keep_default_na=False)
    return data


## Compute geodesic distance (haversin formula)
def haversin(phi1, phi2, lbda1, lbda2):
    r = 6371 #earth radius (km)
    t1 = (np.sin((phi2-phi1)/2))**2
    t2 = np.cos(phi1)*np.cos(phi2)*(np.sin((lbda2-lbda1)/2))**2
    arg = t1 + t2
    arcsin = np.arcsin(np.sqrt(arg))
    return 2*r*arcsin


## Extract 2 most populated cities of each country
def preprocess(rawdf):
    dfVar = rawdf[['city', 'lat', 'lng', 'country', 'iso3', 'population']] #useful variables
    sortdf = dfVar.sort_values(['country', 'population'], ascending=[True, False])
    groupdf = sortdf.groupby('country').head(2).reset_index(drop=True)
    df = groupdf[['city', 'lat', 'lng', 'country', 'iso3']]
    return df


## Compute the distance matrix
def dist_matrix(df):
    nbCities = df.shape[0]
    D = np.zeros((nbCities, nbCities))
    for i in range(len(D)):
        for j in range(len(D)):
            phi1 = df.iloc[i].lat
            phi2 = df.iloc[j].lat
            lbda1 = df.iloc[i].lng
            lbda2 = df.iloc[j].lng
            D[i, j] = haversin(phi1*np.pi/180, phi2*np.pi/180, lbda1*np.pi/180, lbda2*np.pi/180)
    return D


# Associate each country to its continent
def find_continents(df):
    country_continent = load_data("country_continent.csv")
    continents = []
    for country_idx in range(n):
        country = df['iso3'][country_idx]
        country_found = False
        continent_cpt = 0
        while (not (country_found) and continent_cpt < 262):
            current_country = country_continent['Three_Letter_Country_Code'][continent_cpt]
            if current_country == country:
                continents.append(country_continent['Continent_Code'][continent_cpt])
                country_found = True
            else:
                continent_cpt += 1
        if continent_cpt == 262:
            continents.append('Not found')
    return continents


## Classical MDS Algorithm
def classical_mds(d):
    n = d.shape[0]
    # double centering trick
    one = np.ones((n, 1))
    # s = -0.5*(d - (1.0/n)*d@one@one.T - (1.0/n)*one@one.T@d + (1.0/n**2) * one@one.T@d@one@one.T)

    C = np.tile(np.mean(d, axis=0), (n, 1))
    R = np.tile(np.mean(d, axis=1), (n, 1))
    M = np.tile(np.mean(d), (n, n))
    s = -0.5 * (d - R - C + M)

    # eigen-decomposition of s
    eigVal, eigVect = np.linalg.eig(s)
    # sorting eigenVal and eigenVect to have eigenVal in the descending order
    idx = eigVal.argsort()[::-1]
    eigVal = eigVal[idx]
    eigVect = eigVect[:, idx]

    # data embedding
    k = 2
    ikn = np.zeros((n, n))
    for i in range(k):
        ikn[i, i] = 1
    eigVal[-1] *= -1
    x = ikn @ (np.diag(np.sqrt(eigVal))) @ eigVect.T

    # embedded data dataframe
    x = x.real
    x1 = np.reshape(x[0, :], (1, n))
    x2 = np.reshape(x[1, :], (1, n))
    X = np.concatenate((x1.T, x2.T), axis=1)
    dfX = pd.DataFrame(X, columns=['x1', 'x2'])
    # adding countries and iso3
    dfX = dfX.set_index(df.index)
    dfX.insert(2, 'country', df['country'])
    dfX.insert(3, 'iso3', df['iso3'])

    # Adding continents to the dataframe
    continents = find_continents(df)
    dfCont = pd.DataFrame(np.array(continents), columns=['continent'])
    dfCont = dfCont.set_index(df.index)
    dfX.insert(3, 'continent', dfCont['continent'])

    return dfX


## Metric MDS Algorithm
def metric_mds(max_iter, eps):
    mds = MDS(n_components=2, max_iter=max_iter, eps=eps, dissimilarity="precomputed")
    pos = mds.fit(d).embedding_

    continents = find_continents(df)
    dfPos = pd.DataFrame(list(zip(pos[:, 0], pos[:, 1], np.array(continents))), columns=['pos1', 'pos2', 'continent'])
    dfPos.insert(3, 'iso3', df['iso3'])

    title = "max_iter=" + str(max_iter) + ", eps=" + str(eps)

    return dfPos, title


## Compute the error between computed and true distances
def compute_error(d, dfRes):
    x1 = dfRes.iloc[:, 0]
    x2 = dfRes.iloc[:, 1]
    nbCities = dfRes.shape[0]

    error = 0
    # Compute the distance matrix of the embedded data and the error
    embedded_d = np.zeros((nbCities, nbCities))
    for i in range(len(embedded_d)):
        for j in range(i+1):
            phi1 = x1[i]
            phi2 = x1[j]
            lbda1 = x2[i]
            lbda2 = x2[j]
            new_dist = haversin(phi1*np.pi/180, phi2*np.pi/180, lbda1*np.pi/180, lbda2*np.pi/180)
            embedded_d[i, j] = new_dist
            error += (new_dist-d[i, j])**2
    return np.sqrt(error)/(nbCities*(nbCities+1)/2)

path = "worldcities.csv"
rawdf = load_data(path)
df = preprocess(rawdf) # Extracting 2 most populated cities of each country


## Testing haversin formula
"""
prs = 127
tky = 187
phi1 = df.iloc[prs].lat
phi2 = df.iloc[tky].lat
lbda1 = df.iloc[prs].lng
lbda2 = df.iloc[tky].lng
print(haversin(phi1, phi2, lbda1, lbda2))
"""

## Computing the distance matrix
# (done once and saved in dist.npy file)
"""
D = dist_matrix(df)
new_name = "dist.npy"
np.save(new_name, D)
"""

## Loading the distance matrix D (saved before)
saved_d = "dist.npy"
d = np.load(saved_d)
n = d.shape[0]
# sns.heatmap(d) # plotting d heatmap
# plt.show()

## 2 methods of data displaying:
"""
# using matplotlib
plt.scatter(x=rawdf['lng'], y=rawdf['lat'])
plt.show()
# using seaborn
sns.scatterplot(data=rawdf, x='lng', y='lat', hue='capital')
"""

## Double_centering illustration
"""
    ## Preprocessing raw data
reducedf = rawdf[['city', 'lat', 'lng', 'country', 'iso3', 'population']]  # useful variables
sortdf = reducedf.sort_values(['population'], ascending=False)
subdf = sortdf.iloc[:20]
subD = dist_matrix(subdf)
# np.save("dist_subdf.npy", D)
n = subD.shape[0]
one = np.ones((n, n))
subS = -0.5*(subD - (1.0/n)*subD@one@one.T - (1.0/n)*one@one.T@subD + \
          (1.0/n**2) * one@one.T@subD@one@one.T)
## Displaying heatmaps
mask = np.triu(np.ones_like(subS, dtype=bool))
# cmap = sns.diverging_palette(0, 225, s=80, l=65, as_cmap=True)
cmap = sns.color_palette("mako", as_cmap=True)
# sns.heatmap(subD, mask=mask, cmap=cmap, square=True, linewidths=.5)
# sns.heatmap(subS, mask=mask, cmap=cmap, square=True, linewidths=.5)

subS1 = np.ones((n, n))
for i in range(n):
    for j in range(n):
        subS1[i, j] = -0.5*(subD[i, j]**2 - subD[0, i]**2 - subD[0, j]**2)
# sns.heatmap(subS1, mask=mask, cmap=cmap, square=True, linewidths=.5)
"""

## Classical MDS
dfX = classical_mds(d)
sns.scatterplot(data=dfX, x='x1', y='x2', hue='continent')
for i in range(n):
    plt.annotate(dfX['iso3'][i], (dfX['x1'][i], dfX['x2'][i]))
plt.show()
err = compute_error(d, dfX)


## Metric MDS

max_iter_list = [1, 10, 20, 30, 40, 50, 100, 1000, 3000, 6000, 10000]
eps_list = [100000, 10000, 1000, 100, 10, 1, 1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13]
for max_iter in max_iter_list:
    for eps in eps_list:
        dfPos, title = metric_mds(max_iter, eps)
        err = compute_error(d, dfPos)
        plt.figure()
        plt.title(title+", err="+str(err)+"km")
        sns.scatterplot(data=dfPos, x='pos1', y='pos2', hue="continent")
        plt.savefig(title + ".png")

        # for i in range(n):
        #     plt.annotate(dfPos['iso3'][i], (dfPos['pos1'][i], dfPos['pos2'][i]))
        # plt.show()
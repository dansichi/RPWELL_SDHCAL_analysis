import numpy as np
from scipy.spatial.distance import cdist 

class cluster:
    def __init__(self, x, y):
        self._points = np.dstack([x,y])[0]
        self._connections = np.where(cdist(self._points, self._points) == 1)
    

    def append(self, p):
        self._points = np.append(self._points, np.array([p]), axis=0)
        self._connections = np.where(cdist(self._points, self._points) == 1)
    

    def isLinked(self, a, b, history=np.array([])):
        children = self._connections[1][self._connections[0] == a]
        if children.shape[0] == 0:
            return False
        elif b in children:
            return True
        else:
            children = np.setdiff1d(children,history)
            ans = False

            new_hist =  np.concatenate((children, history))
            for child in children:
                ans = ans or self.isLinked(child, b, new_hist)
                if ans:
                    return ans
            return ans
    

    def seeding(self, p, radius=2):
        # print("radius {}".format(radius))
        seed = np.where(cdist(np.array([p]), self._points) < radius)[1]
        # if len(seed) > 0:
        #     print("seed found at {}".format(seed))
        # else:
        #     print("no seed found for point {} among hits: {}".format(p, self._points))
        return seed
    

    def cluster(self, p, radius=2):
        seed = self.seeding(p, radius)
        members = []
        if len(seed) == 0:
            print("no seed found!")
            return []
        for i in range(self._points.shape[1]):
            if i == seed[0]:
                members.append(self._points[i])
            if self.isLinked(seed[0],i):
                members.append(self._points[i])
        return members

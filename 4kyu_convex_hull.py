import math
import numpy as np

def calc_centroid(points):
    centroid = [0]*len(points[0])
    for point in points:
        for i,coord in enumerate(point):
            centroid[i] += coord
    return list(map(lambda coord: coord/len(points), centroid))

def L2(svec,dvec):
    return math.sqrt(sum((x-y)**2 for x,y in zip(svec,dvec)))

def norm(vec):
    svec = [0] * len(vec)
    return L2(svec,vec)

def angle(p, q, u=None):
    if u is None:
        u = p # u points in direction from origin
    
    if p is q:
        return math.inf

    v = [q_i - p_i for q_i, p_i in zip(q,p)]
    
    num = sum(u*v for u,v in zip(u,v))
    den = norm(u) * norm(v)

    if den == 0:
        den = 0.00001

    # Avoid rounding errors when num == den    
    if num/den < -1:
        return math.acos(-1)

    return math.acos(num/den)

def argmin(arr):
    min_idx = 0
    min_val = arr[min_idx]
    for i,el in enumerate(arr[1:], 1):
        if el < min_val:
            min_idx = i
            min_val = arr[min_idx]
    return min_idx, min_val

def argmax(arr):
    max_idx = 0
    max_val = arr[max_idx]
    for i,el in enumerate(arr[1:], 1):
        if el > max_val:
            max_idx = i
            max_val = arr[max_idx]
    return max_idx

def drop_duplicate_points(points):
    hashes = list(map(lambda p: hash(tuple(p)), points))
    idxs = set(hashes.index(hsh) for hsh in hashes)
    return [points[i] for i in idxs]

def convex_hull(pointlist, verbose=False):
    pointlist = drop_duplicate_points(pointlist)

    # Find the center of mass of all points and select the furthest point
    centroid = calc_centroid(pointlist)
    
    distances = [ L2(centroid, p) for p in pointlist ]
    idx = argmax(distances)
    
    # Starting point and vector
    p = pointlist[idx]
    u = [p[i] - centroid[i] for i in range(len(p))]

    if verbose:
        print(centroid)
        print(" -> ", p)

    # Start building the hull
    hull = [p]

    while True:
        # Calculate angles from point p and select the point with the lowest angle from p
        angles = list(map(lambda q: angle(p, q, u), pointlist))
        idx, _ = argmin(angles)
        q = pointlist[idx]
        u = [qi-pi for qi,pi in zip(q,p)]
        p = q
        hull.append(p)
        
        if verbose:
            print(" -> ", p)

        if hull[0] == hull[-1]: break
        
        if len(hull) > len(pointlist):
            raise ValueError('There was an error calculating hull')

    return hull[:-1]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    # np.random.seed(0)
    # points = np.random.rand(1000,2).tolist()
    
    # Three points
    points = [[0, 0], [5, 3], [0, 5]]
    assert sorted(convex_hull(points)) == [[0, 0], [0, 5], [5, 3]]
    
    # As first case with central point
    points = [[0, 0], [5, 3], [0, 5], [2, 3]]
    assert sorted(convex_hull(points)) == [[0, 0], [0, 5], [5, 3]]

    # As first case with colinear point
    points = [[0, 0], [5, 3], [0, 5], [0, 3]]
    assert sorted(convex_hull(points)) == [[0, 0], [0, 5], [5, 3]]

    # # As first case with central point
    # points = [[0, 0], [5, 3], [0, 5], [2, 3]]
    # assert sorted(convex_hull(points)) == [[0, 0], [0, 5], [5, 3]]

    # # With duplicated point
    # points = [[0, 0], [5, 3], [0, 5], [5, 3]]
    # assert sorted(convex_hull(points)) == [[0, 0], [0, 5], [5, 3]]

    # # Central point, colinear poitn and duplicated point
    # points = [[0, 0], [5, 3], [0, 5], [0, 3], [2, 3], [5, 3]]
    # assert sorted(convex_hull(points)) == [[0, 0], [0, 5], [5, 3]]

    centroid = calc_centroid(points)
    distances = [L2(centroid,p) for p in points]
    p = points[argmax(distances)]
    
    fig, ax = plt.subplots()
    ax.axhline(0, color='black', alpha=0.25)
    ax.axvline(0, color='black', alpha=0.25)
    ax.scatter(*zip(*points), color='lightblue', label='Data')
    ax.scatter(*centroid, color='blue', label='Center Of Mass')
    ax.scatter(*p, color='red', label='Starting Hull Point')

    hull = convex_hull(points)
    print(hull)
    ax.plot(*zip(*hull+[hull[0]]), 'r--', label='hull')
    ax.axis('equal')

    plt.show()
import math

def calc_centroid(points):
    centroid = [0]*len(points[0])
    for point in points:
        for i,coord in enumerate(point):
            centroid[i] += coord
    return list(map(lambda coord: coord/len(points), centroid))

def L2(svec,dvec):
    return math.sqrt(sum((x-y)**2 for x,y in zip(svec,dvec)))

def angle(p, q, offset=0):
    """Calculate the angle between points p and q.

    Returns a float between 0 and 2 * pi. 
    """
    
    delta_x = q[0] - p[0]
    delta_y = q[1] - p[1]

    if delta_x == 0:
        if   delta_y > 0: return 1 * math.pi / 2
        elif delta_y < 0: return 2 * math.pi / 2
        else:             return 4 * math.pi / 2

    angle = math.atan(delta_y/delta_x) - offset

    if delta_y == 0:
        return math.pi

    if   delta_x < 0 and delta_y > 0: angle += 2 * math.pi / 2
    elif delta_x < 0 and delta_y < 0: angle += 2 * math.pi / 2
    elif delta_x > 0 and delta_y < 0: angle =  2 * math.pi + angle
    
    return angle % (2*math.pi)

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

def convex_hull(pointlist):
    # Find the center of mass of all points and select the furthest point
    centroid = calc_centroid(pointlist)
    
    print(centroid)

    distances = [ L2(centroid, p) for p in pointlist ]
    idx = argmax(distances)
    
    # Starting point
    p = pointlist[idx]

    # Angle offset from centroid to starting point, in case starting point is in quadrant 3
    offset = angle(centroid, p, offset=0)

    print(" -> ", p, offset)

    # Start building the hull
    hull = [p]

    while True:
        # Calculate angles from point p and select the point with the lowest angle from p
        angles = list(map(lambda q: angle(p,q, offset), pointlist))
        idx, new_offset = argmin(angles)
        offset += new_offset
        p = pointlist[idx]

        print(" -> ", p, offset)

        hull.append(p)

        # TODO Remove once terminal condition established
        if hull[0] == hull[-1]: break
        if len(hull) > len(pointlist):
            raise ValueError('There was an error calculating hull')

    return hull[:-1]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    # np.random.seed()
    points = np.random.randn(10000,2).tolist()
    # points = [[1,0],[0,1],[-1,0],[0.5,1],[-0.5,0.25],[0.25,0.4]]
    # points = [[0,1],[-0.5,0],[0.5,-1],[2,0.25],[2.5,0]]
    # points = [[0, 0], [5, 3], [.25, 5], [0, 3], [2, 3], [5, 3]]
    centroid = calc_centroid(points)
    hull = convex_hull(points)
    # print(hull)
    
    fig, ax = plt.subplots()
    ax.axhline(0, color='black', alpha=0.25)
    ax.axvline(0, color='black', alpha=0.25)
    ax.scatter(*zip(*points), color='lightblue', label='Data')
    ax.scatter(*centroid, color='blue', label='Center Of Mass')
    ax.plot(*zip(*hull+[hull[0]]), 'r--', label='hull')
    ax.axis('equal')
    ax.legend()
    plt.show()
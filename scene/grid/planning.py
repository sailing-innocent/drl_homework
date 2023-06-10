import numpy as np 

def bfs(maze, start, end):
    height = maze.shape[0]
    width = maze.shape[1]
    visited = np.zeros((height, width), dtype=int)
    visited[start[0]][start[1]] = 1
    queue = []
    queue.append(start)
    parent = {}
    while len(queue) > 0:
        item = queue.pop(0)
        x = item[0]
        y = item[1]
        neighbours = []
        if x > 0 and maze[x][y] & 0b1000 == 0:
            neighbours.append((x-1, y))
        if x < height-1 and maze[x][y] & 0b0010 == 0:
            neighbours.append((x+1, y))
        if y > 0 and maze[x][y] & 0b0001 == 0:
            neighbours.append((x, y-1))
        if y < width-1 and maze[x][y] & 0b0100 == 0:
            neighbours.append((x, y+1))
        unvisited_neighbours = []
        for neighbour in neighbours:
            if visited[neighbour[0]][neighbour[1]] == 0:
                unvisited_neighbours.append(neighbour)
        for neighbour in unvisited_neighbours:
            queue.append(neighbour)
            visited[neighbour[0]][neighbour[1]] = 1
            parent[neighbour] = item
            if neighbour == end:
                queue = []
                break
    traj = []
    traj.append(end)
    while traj[-1] != start:
        traj.append(parent[traj[-1]])
    traj.reverse()
    return traj

def bfs_policy(maze, start, end):
    height = maze.shape[0]
    width = maze.shape[1]
    visited = np.zeros((height, width), dtype=int)
    visited[start[0]][start[1]] = 1
    queue = []
    queue.append(start)
    parent = {}
    while len(queue) > 0:
        item = queue.pop(0)
        x = item[0]
        y = item[1]
        neighbours = []
        if x > 0 and maze[x][y] & 0b1000 == 0:
            neighbours.append((x-1, y))
        if x < height-1 and maze[x][y] & 0b0010 == 0:
            neighbours.append((x+1, y))
        if y > 0 and maze[x][y] & 0b0001 == 0:
            neighbours.append((x, y-1))
        if y < width-1 and maze[x][y] & 0b0100 == 0:
            neighbours.append((x, y+1))
        unvisited_neighbours = []
        for neighbour in neighbours:
            if visited[neighbour[0]][neighbour[1]] == 0:
                unvisited_neighbours.append(neighbour)
        for neighbour in unvisited_neighbours:
            queue.append(neighbour)
            visited[neighbour[0]][neighbour[1]] = 1
            parent[neighbour] = item
            if neighbour == end:
                queue = []
                break
    
    policy = []
    traj = []
    traj.append(end)
    while traj[-1] != start:
        pr = parent[traj[-1]]
        st = traj[-1]
        if st != None:
            d = np.array([st[0] - pr[0],st[1]-pr[1]])
            # up, right, down, left
            if d[0] == -1 and d[1] == 0:
                policy.append(0)
            if d[0] == 0 and d[1] == 1:
                policy.append(1)
            if d[0] == 1 and d[1] == 0:
                policy.append(2)
            if d[0] == 0 and d[1] == -1:
                policy.append(3)
        traj.append(pr)
    traj.reverse()
    policy.reverse()
    return policy


def astar(maze, start, end):
    height = maze.shape[0]
    width = maze.shape[1]

    # TODO 1: initialize the open list
    # TODO 2: initialize the closed list
    # TODO 3: put the starting node on the open list (you can leave its f at zero)
    # TODO 4: while the open list is not empty
        # TODO 5: find the node with the least f on the open list, call it "q"
        # TODO 6: pop q off the open list
        # TODO 7: generate q's 8 successors and set their parents to q
        # TODO 8: for each successor
            # TODO 9: if successor is the goal, stop the search
            # TODO 10: successor.g = q.g + distance between successor and q
            # TODO 11: successor.h = distance from goal to successor
            # TODO 12: successor.f = successor.g + successor.h
            # TODO 13: if a node with the same position as successor is in the OPEN list \
            #           which has a lower f than successor, skip this successor
            # TODO 14: if a node with the same position as successor is in the CLOSED list \
            #           which has a lower f than successor, skip this successor
            # TODO 15: otherwise, add the node to the open list
            # TODO 16: push q on the closed list
        # TODO 17: end (for loop)
    # TODO 18: end (while loop)
    # TODO 19: save the path from goal to start
    # TODO 20: return the path

    return traj
import numpy as np 

def get_maze_debug(maze):
    [h, w] = maze.shape
    print()
    print("A maze with ", w, " cols and ", h, " rows.")
    fmt_strs = []
    for i in range(2*h+1):
        fmt_strs.append("")
    
    fmt_strs[0] += "#"
    for i in range(h):
        for j in range(w):
            if i == 0:
                # for the first row, consider the upper bound
                if maze[i][j] & 0b1000 == 0:
                    fmt_strs[0] += "    #"
                else:
                    fmt_strs[0] += " -- #"
            if j == 0:
                # for the first column, consider the left bound
                if maze[i][j] & 0b0001 == 0:
                    fmt_strs[2*i+1] += " "
                    fmt_strs[2*i+2] += "#"
                else:
                    fmt_strs[2*i+1] += "|"
                    fmt_strs[2*i+2] += "#"
            # consider the right and lower bound
            if maze[i][j] & 0b0100 == 0:
                fmt_strs[2*i+1] += "     "
            else:
                fmt_strs[2*i+1] += "    |"
            
            # lower bound
            if maze[i][j] & 0b0010 == 0:
                fmt_strs[2*i+2] += "    #"
            else:
                fmt_strs[2*i+2] += " -- #"

    return "\n".join(fmt_strs)

def print_maze(maze):
    print(get_maze_debug(maze))

def generate_maze(mode = 'test', width = 3, height = 3, startnode = (0, 0), endnode = (2, 2)):
    maze = np.ones((height, width), dtype=int) * 0b1111
    
    if mode == 'test':
        assert(height==3)
        assert(width==3)
        #  THE DEFAULT MAZE FOR TEST
        #                ^ 
        #   # -- # -- #  | #
        # ->     |    |    |
        #   #    #    #    #
        #   |         |    |
        #   #    # -- #    #
        #   |              |  
        #   # -- # -- # -- #
        #
        # 1 for wall
        # 0 for path
        
        for i in range(height):
            for j in range(width):
                maze[0][0] = 0b1101 # up | right | down | left
                maze[0][1] = 0b1101
                maze[0][2] = 0b1101
                maze[1][0] = 0b0001
                maze[1][1] = 0b0110
                maze[1][2] = 0b0101
                maze[2][0] = 0b0011
                maze[2][1] = 0b1010
                maze[2][2] = 0b0110
    
    elif mode == 'dfs':
        # given the current cell as a parameter
        # mark the current cell as visited
        # while the current cell has any unvisited neighbour cells
        #   choose one of the unvisited neighbours
        #   remove the wall between the current cell and the chosen cell
        #   invoke the routine recursively for a chosen cell
        start = startnode 
        visited = np.zeros((height, width), dtype=int)
        visited[start[0]][start[1]] = 1
        stack = []
        stack.append(start)
        while len(stack) > 0:
            item = stack[-1]
            x = item[0]
            y = item[1]
            neighbours = []
            if x > 0:
                neighbours.append((x-1, y))
            if x < height-1:
                neighbours.append((x+1, y))
            if y > 0:
                neighbours.append((x, y-1))
            if y < width-1:
                neighbours.append((x, y+1))
            unvisited_neighbours = []
            for neighbour in neighbours:
                if visited[neighbour[0]][neighbour[1]] == 0:
                    unvisited_neighbours.append(neighbour)
            if len(unvisited_neighbours) > 0:
                sel = np.random.randint(0, len(unvisited_neighbours))
                neighbour = unvisited_neighbours[sel]
                
                # change wall to path
                if neighbour[0] == x-1:
                    maze[x][y] = maze[x][y] & 0b0111
                    maze[neighbour[0]][neighbour[1]] = maze[neighbour[0]][neighbour[1]] & 0b1101
                elif neighbour[0] == x+1:
                    maze[x][y] = maze[x][y] & 0b1101
                    maze[neighbour[0]][neighbour[1]] = maze[neighbour[0]][neighbour[1]] & 0b0111
                elif neighbour[1] == y-1:
                    maze[x][y] = maze[x][y] & 0b1110
                    maze[neighbour[0]][neighbour[1]] = maze[neighbour[0]][neighbour[1]] & 0b1011
                elif neighbour[1] == y+1:
                    maze[x][y] = maze[x][y] & 0b1011
                    maze[neighbour[0]][neighbour[1]] = maze[neighbour[0]][neighbour[1]] & 0b1110
                    
                stack.append(neighbour)
                visited[neighbour[0]][neighbour[1]] = 1
            else:
                stack.pop()

    assert endnode != startnode
  
    assert startnode[0] == 0 or startnode[0] == height-1 or startnode[1] == 0 or startnode[1] == width-1
    if startnode[0] == 0:
        maze[startnode[0]][startnode[1]] = maze[startnode[0]][startnode[1]] & 0b0111
    elif startnode[0] == height-1:
        maze[startnode[0]][startnode[1]] = maze[startnode[0]][startnode[1]] & 0b1101
    elif startnode[1] == 0:
        maze[startnode[0]][startnode[1]] = maze[startnode[0]][startnode[1]] & 0b1110
    elif startnode[1] == width-1:
        maze[startnode[0]][startnode[1]] = maze[startnode[0]][startnode[1]] & 0b1011
     
    assert endnode[0] == 0 or endnode[0] == height-1 or endnode[1] == 0 or endnode[1] == width-1
    if endnode[0] == 0:
        maze[endnode[0]][endnode[1]] = maze[endnode[0]][endnode[1]] & 0b0111
    elif endnode[0] == height-1:
        maze[endnode[0]][endnode[1]] = maze[endnode[0]][endnode[1]] & 0b1101
    elif endnode[1] == 0:
        maze[endnode[0]][endnode[1]] = maze[endnode[0]][endnode[1]] & 0b1110
    elif endnode[1] == width-1:
        maze[endnode[0]][endnode[1]] = maze[endnode[0]][endnode[1]] & 0b1011
        
    return maze 

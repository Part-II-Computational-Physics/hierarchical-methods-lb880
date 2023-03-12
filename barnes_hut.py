import numpy as np

class Body():
    def __init__(self, mass, pos, vel) -> None:
        self.mass = mass
        self.pos = pos
        self.vel = vel

class Quadnode():
    """Quadtree that subdivides the space
    
    With children quadtrees, and stores contained bodies"""

    def __init__(self, parent, coord, side) -> None:
        """
        Parameters
        ----------
        parent : Quadnode
            the parent of the quadnode
        coord
            coordinates of most negative corner of the box
        side
            one side length of the box
        """
        self.parent = parent

        self.bodies = set()

        self.coord = coord
        self.side = side

        self.CoM = np.zeros(2)
        self.mass = 0

    def quad_insert(self, body:Body, min_size):
        """Try to insert body into this quadnode of the quadtree"""
        if len(self.bodies) ==0 or self.side <= min_size: # is empty leaf or min box size
            self.bodies.add(body)
        elif len(self.bodies) == 1: # is non-empty leaf
            self.bodies.add(body)
            # create four children
            coords = [self.coord, 
                      self.coord + np.array([0, self.side/2]),
                      self.coord + np.array([self.side/2, 0]),
                      self.coord + np.array([self.side/2, self.side/2])]
            self.children = [Quadnode(self, coord, self.side/2) for coord in coords]

            # add bodies
            for current_body in self.bodies:
                # True for more negative, False for more positive
                x = ((current_body.pos[0] - self.coord[0]) < (self.side / 2))
                y = ((current_body.pos[1] - self.coord[1]) < (self.side / 2))
                # select child
                if x and y: child=self.children[0]
                elif x and not y: child=self.children[1]
                elif not x and y: child=self.children[2]
                elif not x and not y: child=self.children[3]
                child.quad_insert(current_body, min_size)
        else: # is node
            self.bodies.add(body)
            # determine which child the body lies in
            # True for more negative, False for more positive
            x = ((body.pos[0] - self.coord[0]) < (self.side / 2))
            y = ((body.pos[1] - self.coord[1]) < (self.side / 2))
            # select child
            if x and y: child=self.children[0]
            elif x and not y: child=self.children[1]
            elif not x and y: child=self.children[2]
            elif not x and not y: child=self.children[3]
            # and insert there
            child.quad_insert(body, min_size)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == "__main__":
    body_list = []
    total_size = 1
    num_bodies = 100
    for _ in range(num_bodies):
        body_list.append(Body(1, np.array([np.random.uniform(0, total_size), np.random.uniform(0, total_size)]), np.zeros(2)))

    quadtree = Quadnode(None, np.zeros(2), total_size)
    min_size = total_size / 2**8
    for body in body_list:
        quadtree.quad_insert(body, min_size)

    X = [body.pos[0] for body in body_list]
    Y = [body.pos[1] for body in body_list]

    fig, ax = plt.subplots()
    ax.plot(X, Y, 'o')
    ax.set_xlim(0, total_size)
    ax.set_ylim(0, total_size)
    ax.set_aspect('equal')

    def create_rectangle(quadnode:Quadnode):
        """Create rectangle patches for all leaf nodes in quadtree"""
        if not hasattr(quadnode, 'children'):
            rect = patches.Rectangle(quadnode.coord, quadnode.side, quadnode.side,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        else:
            for child in quadnode.children:
                create_rectangle(child)

    create_rectangle(quadtree)

    plt.show()


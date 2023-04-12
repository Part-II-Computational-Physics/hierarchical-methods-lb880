import numpy as np
import matplotlib.patches as patches

import numpy.typing as npt

class Body():
    def __init__(self, mass, pos) -> None:
        self.mass = mass
        self.pos = pos
        self.acc = np.zeros(2)

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

    def __repr__(self) -> str:
        return f"CoM: {self.CoM}, Mass: {self.mass}, Children: {hasattr(self, 'children')}"
    
    def print_tree(self, indent=0) -> None:
        print('\t'*indent, self)
        if hasattr(self, 'children'):
            indent += 1
            for child in self.children:
                child.print_tree(indent)

    def quad_insert(self, body:Body, min_size) -> None:
        """Try to insert body into this quadnode of the quadtree"""
        if len(self.bodies) == 0 or self.side <= min_size: # is empty leaf or min box size
            self.bodies.add(body)
        elif len(self.bodies) == 1: # is non-empty leaf
            self.bodies.add(body)
            # create four children
            coords = [self.coord, 
                      self.coord + np.array([self.side/2, 0]),
                      self.coord + np.array([0, self.side/2]),
                      self.coord + np.array([self.side/2, self.side/2])]
            self.children = [Quadnode(self, coord, self.side/2) for coord in coords]

            # add bodies
            for current_body in self.bodies:
                # True for more negative, False for more positive
                x = ((current_body.pos[0] - self.coord[0]) < (self.side / 2))
                y = ((current_body.pos[1] - self.coord[1]) < (self.side / 2))
                # select child
                #  --- ---
                # | 2 | 3 |
                #  --- --- 
                # | 0 | 1 |
                #  --- --- 
                if x and y: child=self.children[0]
                elif not x and y: child=self.children[1]
                elif x and not y: child=self.children[2]
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
            elif not x and y: child=self.children[1]
            elif x and not y: child=self.children[2]
            elif not x and not y: child=self.children[3]
            # and insert there
            child.quad_insert(body, min_size)

    def calculate_com(self) -> None:
        """Recursively explore the quadtree and calculate children's CoM, \
            then return up to calculate all nodes CoM and mass
        """
        if not hasattr(self, 'children'): # leaf node
            if len(self.bodies) > 0 : # has bodies, but no children
                weighted_positions = np.zeros((len(self.bodies), 2))
                for i, body in enumerate(self.bodies):
                    weighted_positions[i] = body.mass * body.pos
                    self.mass += body.mass
                self.CoM = np.sum(weighted_positions, axis=0) / self.mass

        else: # has children
            weighted_positions = np.zeros((4,2)) # has 4 children
            for i, child in enumerate(self.children):
                # calculate com of all children to then use
                child.calculate_com()
                weighted_positions[i] = child.mass * child.CoM
                self.mass += child.mass
            self.CoM = np.sum(weighted_positions, axis=0) / self.mass

def _accelerations_through_tree(
        properties:dict,
        body:Body,
        node:Quadnode
    ) -> None:
    """Recursive function to go through tree until theta condition satisfied
    
    Considers possible acceleration from given node"""

    if node.mass == 0:
        return
    
    if body in node.bodies:
        if hasattr(node, 'children'):
            for child in node.children:
                _accelerations_through_tree(properties, body, child)
        return
    
    r = body.pos - node.CoM # points at body 
    mag_r = np.linalg.norm(r)

    # theta = side / distance to CoM
    if hasattr(node, 'children') and (node.side / mag_r > properties['theta']): # not allowed theta
        # further traverse tree
        for child in node.children:
            _accelerations_through_tree(properties, body, child)
    else:
        dir_r = r / mag_r
        body.acc -= node.mass * (properties['G'] * dir_r) / (mag_r**2 + properties['softening']**2)

def calculate_accelerations(
        properties:dict,
        masses:npt.NDArray,
        positions:npt.NDArray
    ) -> npt.NDArray:
    """Barnes Hut acceleration calculation"""

    # create body objects
    bodies = [Body(mass, position) for mass, position in zip(masses, positions)]

    quadtree = Quadnode(None, np.zeros(2), properties['size'])
    min_size = properties['size'] / 2**8

    # fill in quadtree
    for body in bodies:
        quadtree.quad_insert(body, min_size)
    
    quadtree.calculate_com()

    # calculate the accelerations
    for body in bodies:
        _accelerations_through_tree(properties, body, quadtree)

    accelerations = [body.acc for body in bodies]
    return np.array(accelerations)


def draw_rectangles(properties:dict, positions, ax) -> list:
    """Create list of patch artists to draw rectangles
    
    Not efficient as remakes quadtree each call
    
    Parameters
    ----------
    properties : dict
        dictionary of various universe parameters
    positions : NDArray
        numpy array of all body positions
    ax
        plot axis to draw onto
    """

    # create body objects
    bodies = [Body(1, position) for position in positions]

    quadtree = Quadnode(None, np.zeros(2), properties['size'])
    min_size = properties['size'] / 2**8

    # fill in quadtree
    for body in bodies:
        quadtree.quad_insert(body, min_size)
    
    patch_artists = []
    _create_rectangle(quadtree, patch_artists, ax)
    return patch_artists

def _create_rectangle(quadnode:Quadnode, patch_artists:list, ax):
    """Create rectangle patches for all leaf nodes in quadtree"""

    if not hasattr(quadnode, 'children'):
        rect = patches.Rectangle(quadnode.coord, quadnode.side, quadnode.side,
                                    linewidth=1, edgecolor='r', facecolor='none')
        patch_artists.append(ax.add_patch(rect))
    else:
        for child in quadnode.children:
            _create_rectangle(child, patch_artists, ax)

def create_rectangle(quadnode:Quadnode, ax):
    """Create rectangle patches for all leaf nodes in quadtree"""
    if not hasattr(quadnode, 'children'):
        rect = patches.Rectangle(quadnode.coord, quadnode.side, quadnode.side,
                                    linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    else:
        for child in quadnode.children:
            create_rectangle(child, ax)

def main():
    import matplotlib.pyplot as plt

    body_list = []
    total_size = 1
    num_bodies = 100
    for _ in range(num_bodies):
        body_list.append(Body(1, np.array([np.random.uniform(0, total_size), np.random.uniform(0, total_size)]), np.zeros(2)))

    quadtree = Quadnode(None, np.zeros(2), total_size)
    min_size = total_size / 2**8
    for body in body_list:
        quadtree.quad_insert(body, min_size)
    
    quadtree.calculate_com()
    quadtree.print_tree()

    X = [body.pos[0] for body in body_list]
    Y = [body.pos[1] for body in body_list]

    fig, ax = plt.subplots()
    ax.plot(X, Y, 'o')
    ax.set_xlim(0, total_size)
    ax.set_ylim(0, total_size)
    ax.set_aspect('equal')

    create_rectangle(quadtree, ax)

    plt.show()
    
if __name__ == "__main__":
    main()


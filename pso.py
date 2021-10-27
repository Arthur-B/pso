"""
Particle swarm optimization implementation in Python.
"""

from dataclasses import dataclass, replace

def fom(x) -> float:
    """
    Figure of merit function
    """
    return - (x[0]**2 + x[1]**2)**0.5


@dataclass
class DataPoint:
    """
    A point in space.
    Defined by its position, x, and performance, y.
    """
    x: tuple            # Position
    y: float = None     # Value (FOM)

    def updateFom(self, fom=fom):
        self.y = fom(self.x)
    
    def __post_init__(self):
        if self.y is None:
            self.updateFom()


@dataclass
class Particle:
    """
    One particle of the the optimizations.
    Defined by its current position, a DataPoint, and its personal best, another DataPoint.
    """
    currentPosition: DataPoint
    # pBest: DataPoint = None

    def updatePBest(self):
        if self.currentPosition.y < self.pBest.y:
            # dataclasses.replace to make a copy of the object
            self.pBest = replace(self.currentPosition)

    def __post_init__(self):
        self.pBest = replace(self.currentPosition)  # Personal best, initialized as first position


@dataclass
class Swarm:
    paramSpace: list    # List of boundaries of parameter space (min, max)
    swarmSize: int # Size of the swarm
    epochs: int

    # swarm: list = None   # List of particles
    # gBest: DataPoint = None

    c1: float = 1.49
    c2: float = 1.49
    w: float = 0.9
    w_decrease: float = 0.5
    wallType = 1

    def initPosition(self):
        self.swarm = [] # List of particles
        self.gBest = DataPoint((0, 0), -1) # Global best, initialized at an impossible value
        for i in range(self.swarmSize):
            position = DataPoint((1, 1)) # Should be a random or LHS
            self.swarm.append(Particle(position))
    
    def updatePosition(self):
        for particle in self.swarm:
            position = (2, 2) # Should follow rule (c1, c2, velocity, etc.)
            particle.currentPosition.x = position   # Assign new x
            particle.currentPosition.updateFom()    # Compute new y
            particle.updatePBest()                  # Check if new y better than personal best

    def updateGBest(self):
        for particle in self.swarm:
            if particle.currentPosition.y < self.gBest.y:
                self.gBest = replace(particle.currentPosition)
 

    def __post_init__(self):
        self.initPosition()
        self.updateGBest()
        self.wDifference = self.w_decrease / self.epochs



# class Optimization:
#     def __init__(self, swarmSize, epochs,
#                  c1=1.49, c2=1.49, wallType=1) -> None:
        
#         self.swarmSize = swarmSize
#         self.epochs = epochs
        
#         self.c1 = c1
#         self.c2 = c2
#         self.wallType = wallType

#         pass


# %% Test a point

# test_point = DataPoint((1, 2))
# print(test_point)
# test_point.x = (5, 5)
# print(test_point)
# test_point.updateFom()
# print(test_point)

# %% Test a particle

# test_particle = Particle(currentPosition=DataPoint((1, 1)))
# print(test_particle)

# test_particle.currentPosition.x = (10, 10)
# test_particle.currentPosition.updateFom()
# test_particle.updatePBest()

# print(test_particle)

# test_particle.currentPosition.x = (3, 3)
# test_particle.currentPosition.updateFom()
# test_particle.updatePBest()

# print(test_particle)

# %% Test a swarm

paramSpace = [[-1, 1],
              [-5, 5]]
swarm = Swarm(paramSpace, 3, 10)
print(swarm)

print('\n\n\n')
swarm.updatePosition()
swarm.updateGBest()
print(swarm)
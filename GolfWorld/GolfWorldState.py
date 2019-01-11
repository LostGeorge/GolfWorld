from simple_rl.mdp.StateClass import State

class GolfWorldState(State):
    ''' Class for Golf World States '''

    def __init__(self, x, y):
        State.__init__(self, data=[x, y], is_terminal=False)
        self.x = round(x, 5)
        self.y = round(y, 5)
        
    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, GolfWorldState) and self.x == other.x and self.y == other.y
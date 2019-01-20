import random, math
from numpy.random import normal
from simple_rl.mdp.MDPClass import MDP
from .GolfWorldState import GolfWorldState

class GolfWorldMDP(MDP):

    ACTIONS = ["driver_left", "driver_up", "driver_right", "driver_down",
               "iron_left", "iron_up", "iron_right", "iron_down",
               "putt_left", "putt_up", "putt_right", "putt_down"]
    '''
    Driver shoots foward (up) by 10 units, with relative inaccuracy
    Iron is 5 units, higher accuracy
    Putt is 1 unit, accurate
    '''

    # weird storage things needed

    def __init__(self, width, height,
                tee_loc, hole_loc, rough_locs, hazard_locs, wind_dir, wind_spd,
                gamma, step_cost, name):
        
        '''
        In rough, all driver shots converted to iron and failure chance exists
        In hazard, only putt is allowed and greater failure chance exists
        wind_dir is either up, down, left, right
        wind_spd shifts driver shots by .2*speed and iron shots by .1*speed on average
        '''
        self.width = width
        self.height = height
        self.tee_loc = tee_loc
        self.hole_loc = hole_loc
        self.rough_locs = rough_locs
        self.hazard_locs = hazard_locs
        self.wind_dir = wind_dir
        self.wind_spd = wind_spd
        self.gamma = gamma
        self.step_cost = step_cost
        self.name = name

        self.new_x = 0
        self.new_y = 0

        self.init_state = GolfWorldState(self.tee_loc[0], self.tee_loc[1])
        
        MDP.__init__(self, GolfWorldMDP.ACTIONS, self._transition_func, self._reward_func, self.init_state, self.gamma, self.step_cost)
    
    def _is_goal_state(self, state):
        return (int(round(state.x)), int(round(state.y))) == self.hole_loc
    
    def _is_rough_state(self, state):
        return (int(round(state.x)), int(round(state.y))) in self.rough_locs
    
    def _is_hazard_state(self, state):
        return (int(round(state.x)), int(round(state.y))) in self.hazard_locs

    def _is_driver_action(self, action):
        return action == "driver_left" or action == "driver_right" or action == "driver_up" or action == "driver_down"
    
    def _is_iron_action(self, action):
        return action == "iron_left" or action == "iron_right" or action == "iron_up" or action == "iron_down"

    def _wind_movement(self, dir, spd, action):
        wind_x = 0.0
        wind_y = 0.0
        if dir == "up":
            if self._is_driver_action(action=action):
                wind_y += .2*spd
            elif self._is_iron_action(action=action):
                wind_y += .1*spd
        elif dir == "down":
            if self._is_driver_action(action=action):
                wind_y -= .2*spd
            elif self._is_iron_action(action=action):
                wind_y -= .1*spd
        elif dir == "left":
            if self._is_driver_action(action=action):
                wind_x -= .2*spd
            elif self._is_iron_action(action=action):
                wind_x -= .1*spd
        elif dir == "right":
            if self._is_driver_action(action=action):
                wind_x += .2*spd
            elif self._is_iron_action(action=action):
                wind_x += .1*spd
        return (wind_x,wind_y)
    
    def _rand_movement(self, action):
        rand_r = 0.0
        rand_deg = random.random()*math.pi
        if self._is_driver_action(action=action):
            rand_r += normal(0.0,2.0,None)
        elif self._is_iron_action(action=action):
            rand_r += normal(0.0,1.0,None)
        return (rand_r*math.cos(rand_deg),rand_r*math.sin(rand_deg))

    def _transition_pos(self, state, action):
        if state.is_terminal():
            return (state.x, state.y)
        
        d_x = 0
        d_y = 0
        fail = random.random()

        if self._is_hazard_state(state) and fail < 0.5:
            if action == "driver_left" or action == "iron_left":
                self._transition_pos(state, "putt_left")
            elif action == "driver_up" or action == "iron_up":
                self._transition_pos(state, "putt_up")
            elif action == "driver_right" or action == "iron_right":
                self._transition_pos(state, "putt_right")
            elif action == "driver_down" or action == "iron_down":
                self._transition_pos(state, "putt_down")

            elif action == "putt_left":
                d_x += 1
            elif action == "putt_up":
                d_y += 1
            elif action == "putt_right":
                d_x += 1
            elif action == "putt_down":
                d_y -= 1
            
        elif self._is_rough_state(state) and fail < 0.9:
            if action == "driver_left":
                self._transition_pos(state, "iron_left")
            elif action == "driver_up":
                self._transition_pos(state, "iron_up")
            elif action == "driver_right":
                self._transition_pos(state, "iron_right")
            elif action == "driver_left":
                self._transition_pos(state, "iron_left")
            
            elif action == "putt_left":
                d_x += 1
            elif action == "putt_up":
                d_y += 1
            elif action == "putt_right":
                d_x += 1
            elif action == "putt_down":
                d_y -= 1
            
            else:
                (w_x,w_y) = self._wind_movement(self.wind_dir, self.wind_spd, action)
                (r_x,r_y) = self._rand_movement(action)

                if action == "iron_up":
                    d_x += int(round(w_x + r_x))
                    d_y += int(round(w_y + r_y + 5.0))
                elif action == "iron_down":
                    d_x += int(round(w_x + r_x))
                    d_y += int(round(w_y + r_y - 5.0))
                elif action == "iron_left":
                    d_x += int(round(w_x + r_x - 5.0))
                    d_y += int(round(w_y + r_y))
                elif action == "iron_right":
                    d_x += int(round(w_x + r_x + 5.0))
                    d_y += int(round(w_y + r_y))
        
        elif (not self._is_rough_state(state)) and (not self._is_hazard_state(state)):
            if action == "putt_left":
                d_x += 1
            elif action == "putt_up":
                d_y += 1
            elif action == "putt_right":
                d_x += 1
            elif action == "putt_down":
                d_y -= 1
            else:
                (w_x,w_y) = self._wind_movement(self.wind_dir, self.wind_spd, action)
                (r_x,r_y) = self._rand_movement(action)

                if action == "iron_up":
                    d_x += int(round(w_x + r_x))
                    d_y += int(round(w_y + r_y + 5.0))
                elif action == "iron_down":
                    d_x += int(round(w_x + r_x))
                    d_y += int(round(w_y + r_y - 5.0))
                elif action == "iron_left":
                    d_x += int(round(w_x + r_x - 5.0))
                    d_y += int(round(w_y + r_y))
                elif action == "iron_right":
                    d_x += int(round(w_x + r_x + 5.0))
                    d_y += int(round(w_y + r_y))
                
                elif action == "driver_up":
                    d_x += int(round(w_x + r_x))
                    d_y += int(round(w_y + r_y + 10.0))
                elif action == "driver_down":
                    d_x += int(round(w_x + r_x))
                    d_y += int(round(w_y + r_y - 10.0))
                elif action == "driver_left":
                    d_x += int(round(w_x + r_x - 10.0))
                    d_y += int(round(w_y + r_y))
                elif action == "driver_right":
                    d_x += int(round(w_x + r_x + 10.0))
                    d_y += int(round(w_y + r_y))

        if 0 < state.x + d_x <= self.width + 1 and 0 <= state.y + d_y <= self.height + 1:
            self.new_x = state.x + d_x
            self.new_y = state.y + d_y
            return (self.new_x, self.new_y)
        self.new_x = state.x
        self.new_y = state.y
        return (self.new_x, self.new_y)
    
    def _transition_func(self, state, action):
        coords = (self.new_x, self.new_y)
        next_state = GolfWorldState(coords[0], coords[1])
        if coords == self.hole_loc:
            next_state.set_terminal(True)
        return next_state
    
    def _is_goal_state_action(self, state, action):
        return self._transition_pos(state,action) == self.hole_loc

    def _reward_func(self, state, action):
        if self._is_goal_state_action(state, action):
            return 10.0 - self.step_cost
        return 0 - self.step_cost
    
    def __str__(self):
        return self.name + "_h-" + str(self.height) + "_w-" + str(self.width)

    def __repr__(self):
        return self.__str__()
    

import random
"""

This file implements strategies for a game-playing agent for an item pickup game built in Minecraft. 
The game is represented on a 3x5 board. Each square can be occupied by an agent, a cookie, or nothing. 
Agents collect cookies by moving onto a square which contains a cookie. Their goal is to collect more 
than their opponent before time runs out. Only one item can occupy a square at a time.
@author Ashka Stephen 
@version 2.22.2017
"""

class GamePlayer(object):
    '''Represents the logic for an individual player in the game'''

    def __init__(self, player_id, game):
        '''"player_id" indicates which player is represented (int)
        "game" is a game object with a get_successors function'''
        self.player_id = player_id
        self.game = game
        return

    def evaluate(self, state):
        '''Evaluates a given state for the specified agent
        "state" is a game state object'''
        pass

    def minimax_move(self, state):
        '''Returns a string action representing a move for the agent to make'''
        pass

    def alpha_beta_move(self, state):
        '''Same as minimax_move with alpha-beta pruning'''
        pass


class BasicPlayer(GamePlayer):
    '''A basic agent which takes random (valid) actions'''

    def __init__(self, player_id, game):
        GamePlayer.__init__(self, player_id, game)

    def evaluate(self, state):
        '''This agent doesn't evaluate states, so just return 0'''
        return 0

    def minimax_move(self, state):
        '''Don't perform any game-tree expansions, just pick a random move
            that's available in the list of successors'''
        assert state.player == self.player_id
        successors, actions = self.game.get_successors(state)
        # Take a random successor's action
        return random.choice(actions)

    def alpha_beta_move(self, state, agent):
        '''Just calls minimax_move'''
        return self.minimax_move(state, agentd)


def minimax_dfs(game, state, depth, horizon, eval_fn):
    def maxValue(game, state, depth, horizon, eval_fn):
        prob = 0.5
        if depth == horizon:
            return eval_fn(state)  
        maxval = -9999
        succ, act = game.get_successors(state)
        for (s, a) in zip(succ, act):
            curr = minValue(game, s, depth+1, horizon, eval_fn)
            if curr == maxval:
                rand = random.random()
                if rand > prob:
                    maxval = curr
            if curr > maxval:
                maxval = curr
        return maxval
    def minValue(game, state, depth, horizon, eval_fn):
        prob = 0.5
        if depth == horizon:
            return eval_fn(state)        
        minval = 9999
        succ, act = game.get_successors(state)        
        for (s, a) in zip(succ,act):
            curr = maxValue(game, s, depth+1, horizon, eval_fn)
            if curr == minval:
                rand = random.random()
                if rand > prob:
                    minval = curr
            if curr < minval:
                minval = curr
        return minval
    depth = 0
    succ, act = game.get_successors(state)
    top = -9999
    action = 'z'
    for (s, a) in zip(succ,act):
        prob = 0.5
        other = minValue(game, s, depth+1, horizon, eval_fn)
        if other == top:
            rand = random.random()
            if rand > prob:
                top = other
                action = a
        if other > top:
            top = other
            action = a
    return top, action

def alpha_beta_move_helper(game, state, depth, horizon, eval_fn, alpha, beta):
    if depth == horizon:
        return eval_fn(state)
        #not sure about abobe part???

    def maxValue(game, state, depth, horizon, eval_fn, alpha, beta):
        prob = 0.5
        if depth == horizon:
            return eval_fn(state)  
        succ, act = game.get_successors(state)
        for (s, a) in zip(succ, act):
            curr = minValue(game, s, depth+1, horizon, eval_fn, alpha, beta)
            alpha = max(alpha, curr)
            if beta <= alpha:
                return alpha
        return alpha
    def minValue(game, state, depth, horizon, eval_fn, alpha, beta):
        prob = 0.5
        if depth == horizon:
            return eval_fn(state)        
        succ, act = game.get_successors(state)        
        for (s, a) in zip(succ,act):
            curr = maxValue(game, s, depth+1, horizon, eval_fn, alpha, beta)
            beta = min(beta, curr)
            if beta <= alpha:
                break
                #or return beta here not break
        return beta
        
    depth = 0
    succ, act = game.get_successors(state)
    top = beta
    action = 'z'
    for (s, a) in zip(succ,act):
        prob = 0.5
        other = maxValue(game, s, depth+1, horizon, eval_fn, alpha, beta)
        if other > top:
            top = other
            action = a
    return top, action 


class StudentPlayer(GamePlayer):
    def __init__(self, player_id, game):
        GamePlayer.__init__(self, player_id, game)

    def evaluate(self, state):
        #METHOD 1: CURRENT NUM OF COOKIES
        numCookiesAdvantage = (state.cookiecounts[self.player_id] - state.cookiecounts[1-self.player_id]) + 5
        cookiesBorderCount = 0
        cookiesBorderCount2 = 0
        for i in range(len(state.grid)):
            for j in range(len(state.grid[i])):
                if ((state.grid[i][j] == str(self.player_id))):
                    xlocationPlayer = i
                    ylocationPlayer = j
                    originalxlocationPlayer = xlocationPlayer
                    originalylocationPlayer = ylocationPlayer
                if ((state.grid[i][j] == str(1-self.player_id))):
                    x2locationPlayer = i
                    y2locationPlayer = j
                    originalxlocationPlayer1 = x2locationPlayer
                    originalylocationPlayer1 = y2locationPlayer

        #METHOD 2: ADJACENT COOKIES
        xoffset=[-1, -1, -1, 0, 0, 1, 1, 1]
        yoffset=[-1, 0, 1, 1, -1, 1, 0, -1]
        for index in range(len(xoffset)):
            xlocationPlayer += xoffset[index]
            ylocationPlayer += yoffset[index]
            x2locationPlayer += xoffset[index]
            y2locationPlayer += yoffset[index]

            if ((xlocationPlayer>=0) and (ylocationPlayer>=0) and (xlocationPlayer<len(state.grid)) and (ylocationPlayer<len(state.grid[i]))):
                if(state.grid[xlocationPlayer][ylocationPlayer] == 'c'):
                    cookiesBorderCount = cookiesBorderCount + 1

            if ((x2locationPlayer>=0) and (y2locationPlayer>=0) and (x2locationPlayer<len(state.grid)) and (y2locationPlayer<len(state.grid[i]))):
                if(state.grid[x2locationPlayer][y2locationPlayer] == 'c'):
                    cookiesBorderCount2 = cookiesBorderCount2 + 1
        
            xlocationPlayer = originalxlocationPlayer
            ylocationPlayer = originalylocationPlayer
            x2locationPlayer = originalxlocationPlayer1
            y2locationPlayer = originalylocationPlayer1

        #METHOD 3 -> avg manhattan distance
        distance = 0
        distance2 = 0        
        cookiecountforDist = 1
        for i in range(len(state.grid)):
            for j in range(len(state.grid[i])):
                if ((state.grid[i][j] == 'c')):
                    cookiecountforDist = cookiecountforDist + 1
                    distance += abs(originalxlocationPlayer - i) + abs(originalylocationPlayer - j)
                    distance2 += abs(originalxlocationPlayer1 - i) + abs(originalylocationPlayer1 - j)
        avgDistDiff = distance2 - distance
        diffCookieBorder = cookiesBorderCount - cookiesBorderCount2
        return (1.5*numCookiesAdvantage) + diffCookieBorder  + (.2*avgDistDiff)


    def minimax_move(self, state):
        assert state.player == self.player_id
        horizon = 6
        val, action = minimax_dfs(self.game, state, 0, horizon, self.evaluate)
        return action

    def alpha_beta_move(self, state):
        assert state.player == self.player_id
        alpha = 9999
        beta = -9999
        horizon = 6
        val, action = alpha_beta_move_helper(self.game, state, 0, horizon, self.evaluate, alpha, beta)
        return action

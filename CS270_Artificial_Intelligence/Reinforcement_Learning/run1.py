from cliff_walking import CliffWalkingMDP
from game import CliffWalkingGame
from value_iteration_agents import ValueIterationAgent

if __name__ == "__main__":
    grid = [[-100.0, -100.0, -100.0, -100.0, -100.0],
            [-100.0, ' ', ' ', 'S', -100.0],
            [-100.0, ' ', ' ', ' ', -100.0],
            [-100.0, ' ', ' ', ' ', -100.0],
            [-100.0, ' ', ' ', ' ', -100.0],
            [-100.0, ' ', ' ', ' ', -100.0],
            [-100.0, ' ', ' ', ' ', -100.0],
            [-100.0, ' ', ' ', ' ', +100.0],
            [-100.0, ' ', ' ', ' ', -100.0],
            [-100.0, ' ', ' ', ' ', -100.0],
            [-100.0, ' ', ' ', ' ', -100.0],
            [-100.0, ' ', ' ', ' ', -100.0],
            [-100.0, ' ', ' ', ' ', -100.0],
            [-100.0, -100.0, -100.0, -100.0, -100.0]]

    noise = 0.2
    num_missions = 1

    mdp = CliffWalkingMDP(grid)
    mdp.noise = noise

    agent = ValueIterationAgent(mdp)

    game = CliffWalkingGame(agent)
    game.noise = noise

    game.run_game(num_missions)

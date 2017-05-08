from cliff_walking import CliffWalkingMDP
from game import CliffWalkingGame
from q_learning_agents import QLearningAgent

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

    noise = 0.0
    num_missions = 150

    mdp = CliffWalkingMDP(grid)
    mdp.noise = noise

    agent = QLearningAgent(action_fn=mdp.get_possible_actions, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1)

    game = CliffWalkingGame(agent)
    game.noise = noise

    game.run_game_q(num_missions)

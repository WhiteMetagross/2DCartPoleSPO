#This program evaluates the performance of a trained SPO agent on the CartPole-v1 environment.
#This program uses the centralized configuration from config.py.

import os
import gymnasium as gym
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_default_config

#Import from the SPOinPyTorch package.
try:
    from SPOinPyTorch import SPOAgent, Config
except ImportError:
    #Fallback for development/local testing.
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from SPOinPyTorch import SPOAgent, Config

#This function evaluates the agent's performance.
def evaluate_agent(agent, env_name, num_episodes=100, render=False, device='cuda', create_plots=False):
    device = device if torch.cuda.is_available() else 'cpu'

    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    episode_rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(state_tensor)

            #Convert action appropriately for the environment.
            is_discrete_env = isinstance(env.action_space, gym.spaces.Discrete)
            if is_discrete_env:
                #CartPole expects a Python int (0 or 1).
                action_item = int(action.squeeze().cpu().numpy().item())
            else:
                #Continuous envs expect a 1D numpy array.
                action_item = action.squeeze(0).cpu().numpy()

            state, reward, terminated, truncated, _ = env.step(action_item)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_reward >= 475:
            successes += 1

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {np.mean(episode_rewards[-10:]):.2f}")

    env.close()

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = successes / num_episodes

    print("\n" + "="*50)
    print("EVALUATION RESULTS:")
    print("="*50)
    print(f"Number of episodes: {num_episodes}")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average episode length: {avg_length:.2f}")
    print(f"Success rate (reward >= 475): {success_rate:.2%}")
    print(f"Min reward: {min(episode_rewards):.2f}")
    print(f"Max reward: {max(episode_rewards):.2f}")

    if create_plots:
        create_evaluation_plots(episode_rewards, episode_lengths)

    return {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
        "success_rate": success_rate,
        "min_reward": min(episode_rewards),
        "max_reward": max(episode_rewards),
        "all_rewards": episode_rewards
    }

#This function creates evaluation plots.
def create_evaluation_plots(episode_rewards, episode_lengths):
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    #Plot episode rewards.
    ax1.plot(episode_rewards, alpha=0.7, color='blue')
    ax1.axhline(y=475, color='red', linestyle='--', alpha=0.8, label='Success Threshold')
    rolling_mean = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
    ax1.plot(range(9, len(episode_rewards)), rolling_mean, color='orange', linewidth=2, label='10-Episode Rolling Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    #Plot reward distribution.
    ax2.hist(episode_rewards, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=np.mean(episode_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax2.axvline(x=475, color='orange', linestyle='--', label='Success Threshold')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    #Plot episode lengths.
    ax3.plot(episode_lengths, alpha=0.7, color='purple')
    rolling_mean_length = np.convolve(episode_lengths, np.ones(10)/10, mode='valid')
    ax3.plot(range(9, len(episode_lengths)), rolling_mean_length, color='orange', linewidth=2, label='10-Episode Rolling Average')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Episode Length')
    ax3.set_title('Episode Lengths Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    #Plot reward vs episode length.
    reward_vs_length = np.corrcoef(episode_rewards, episode_lengths)[0, 1]
    ax4.scatter(episode_lengths, episode_rewards, alpha=0.6, color='red')
    ax4.set_xlabel('Episode Length')
    ax4.set_ylabel('Episode Reward')
    ax4.set_title(f'Reward vs Episode Length (Correlation: {reward_vs_length:.3f})')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 6))

    bins = np.arange(0, len(episode_rewards) + 10, 10)
    binned_rewards = []
    for i in range(len(bins) - 1):
        start_idx = bins[i]
        end_idx = min(bins[i + 1], len(episode_rewards))
        if start_idx < len(episode_rewards):
            binned_rewards.append(np.mean(episode_rewards[start_idx:end_idx]))

    ax5.bar(range(len(binned_rewards)), binned_rewards, alpha=0.7, color='teal')
    ax5.axhline(y=475, color='red', linestyle='--', alpha=0.8, label='Success Threshold')
    ax5.set_xlabel('Episode Batch (10 episodes each)')
    ax5.set_ylabel('Average Reward')
    ax5.set_title('Average Reward per 10-Episode Batch')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    success_episodes = [i for i, reward in enumerate(episode_rewards) if reward >= 475]
    success_bins = np.arange(0, len(episode_rewards) + 10, 10)
    success_counts = []
    for i in range(len(success_bins) - 1):
        start_idx = success_bins[i]
        end_idx = min(success_bins[i + 1], len(episode_rewards))
        count = sum(1 for ep in success_episodes if start_idx <= ep < end_idx)
        success_counts.append(count)

    ax6.bar(range(len(success_counts)), success_counts, alpha=0.7, color='gold')
    ax6.set_xlabel('Episode Batch (10 episodes each)')
    ax6.set_ylabel('Number of Successful Episodes')
    ax6.set_title('Successful Episodes per 10 Episode Batch')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

#This function loads a trained agent from checkpoint.
def load_trained_agent(model_path, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Normalize config to a Config object.
    if isinstance(config, dict):
        cfg = Config()
        cfg.update(config)
        config = cfg

    #Load checkpoint and update config if it contains saved configuration.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'config' in checkpoint:
        print("Loading configuration from checkpoint...")
        config.update(checkpoint['config'])
    else:
        print("Warning: Checkpoint does not contain saved config. Using provided/default configuration.")

    cfg_dict = config.get_dict()

    #Build environment based on (possibly updated) configurations.
    env = gym.make(cfg_dict["env_name"])
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_low = env.action_space.low if not is_discrete else None
    action_high = env.action_space.high if not is_discrete else None

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if is_discrete else env.action_space.shape[0]

    agent = SPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=cfg_dict,
        is_discrete=is_discrete,
        action_low=action_low,
        action_high=action_high,
        device=device
    )

    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])

    env.close()
    return agent

#This is the main function.
def main():
    #Centralized configuration.
    config = get_default_config()

    #Look for model checkpoints in common locations.
    model_paths = [
        'checkpoints/best_model.pth',
        'checkpoints/final_model.pth',
        '../checkpoints/best_model.pth',
        '../checkpoints/final_model.pth'
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("No trained model found. Please train a model first using train.py.")
        print("Expected model locations:")
        for path in model_paths:
            print(f"  - {path}")
        return

    print(f"Loading trained model from: {model_path}")
    agent = load_trained_agent(model_path, config)
    print("Loaded trained model successfully")

    #Evaluate the agent.
    print("\nEvaluating agent performance...")
    results = evaluate_agent(agent, config.env_name, num_episodes=100, render=False, create_plots=True)

    #Save results.
    with open("evaluation_results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "all_rewards"}, f, indent=4)

    print("\nDetailed results saved to evaluation_results.json")
    print("Evaluation plots saved as evaluation_plots.png and evaluation_analysis.png")

    #Render a few episodes for visual evaluation.
    render_episodes = input("\nRender 5 episodes for visual evaluation? (y/n): ").lower().strip()
    if render_episodes == 'y':
        print("Rendering 5 episodes for visual evaluation...")
        evaluate_agent(agent, config.env_name, num_episodes=5, render=True, create_plots=False)


if __name__ == "__main__":
    main()

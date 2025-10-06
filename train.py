#This program trains the SPO agent on the CartPole-v1 environment.
#This program uses the centralized configuration from config.py.

import os
import json
import time
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm

#Import from the SPOinPyTorch package.
try:
    from SPOinPyTorch import SPOAgent, Config
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from SPOinPyTorch import SPOAgent, Config

#This function saves the model checkpoint with the configuration.
def save_checkpoint(agent, epoch, reward, filepath, config):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'reward': reward,
        'config': config.get_dict(),
    }, filepath)

#This function evaluates the agent's performance.
def evaluate_agent(agent, env_name, num_episodes=10, device='cuda'):
    device = device if torch.cuda.is_available() else 'cpu'
    env = gym.make(env_name)
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
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
        total_reward += episode_reward
    env.close()
    return total_reward / num_episodes

#This is the main function.
def main():
    config = Config()
    #Centralized configuration.
    from config import get_default_config
    config = get_default_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #Create output directories.
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print(f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Starting the SPO algorithm for training on {config.env_name}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    
    #Number of parallel environments.
    num_envs = 16
    print(f"Using {num_envs} parallel environments")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    #Create vectorized environments.
    envs = gym.vector.SyncVectorEnv([
        lambda: gym.make(config.env_name) for _ in range(num_envs)
    ])
    
    #Check if the action space is discrete or continuous.
    is_discrete = isinstance(envs.single_action_space, gym.spaces.Discrete)
    action_low = envs.single_action_space.low if not is_discrete else None
    action_high = envs.single_action_space.high if not is_discrete else None
        
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n if is_discrete else envs.single_action_space.shape[0]
    
    #Create the SPO agent.
    agent = SPOAgent(
        state_dim=state_dim, action_dim=action_dim, config=config.get_dict(),
        is_discrete=is_discrete, action_low=action_low, action_high=action_high, device=device
    )
    
    best_avg_reward = -float("inf")
    global_step = 0
    update = 0
    start_time = time.time()
    
    reward_history = []
    loss_history = []
    eval_rewards = []
    no_improvement_count = 0
    
    log_file = open('logs/training_log.txt', 'w')
    log_file.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Environment: {config.env_name}\n")
    log_file.write(f"Device: {device}\n")
    log_file.write(f"Num environments: {num_envs}\n")
    log_file.write(f"Config: {config.get_dict()}\n\n")
    
    #Training loop.
    try:
        states, _ = envs.reset()
        episode_rewards = np.zeros(num_envs)
        episode_reward_history = []
        
        num_steps_per_batch = config.steps_per_batch // num_envs
        
        progress_bar = tqdm(total=config.total_timesteps, desc="Training Progress", unit="steps")
        
        #Main training loop.
        while global_step < config.total_timesteps:
            batch_states = torch.zeros((num_steps_per_batch, num_envs, state_dim), device=device)
            batch_actions = torch.zeros((num_steps_per_batch, num_envs), device=device, dtype=torch.long)
            batch_log_probs = torch.zeros((num_steps_per_batch, num_envs), device=device)
            batch_rewards = torch.zeros((num_steps_per_batch, num_envs), device=device)
            batch_dones = torch.zeros((num_steps_per_batch, num_envs), device=device)
            batch_values = torch.zeros((num_steps_per_batch, num_envs), device=device)
            
            for step in range(num_steps_per_batch):
                states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    actions, log_probs, _, values = agent.get_action_and_value(states_tensor)

                #For discrete actions, ensure proper shape for vectorized env.
                if is_discrete:
                    actions_np = actions.cpu().numpy().flatten()
                else:
                    actions_np = actions.cpu().numpy()
                
                batch_states[step] = states_tensor
                batch_actions[step] = actions
                batch_log_probs[step] = log_probs
                batch_values[step] = values.flatten()

                next_states, rewards, terminateds, truncateds, _ = envs.step(actions_np)
                dones = terminateds | truncateds
                
                batch_rewards[step] = torch.tensor(rewards, device=device).view(-1)
                batch_dones[step] = torch.tensor(dones, device=device).view(-1)
                
                episode_rewards += rewards
                
                for env_idx, done in enumerate(dones):
                    if done:
                        episode_reward_history.append(episode_rewards[env_idx])
                        episode_rewards[env_idx] = 0
                
                states = next_states
                global_step += num_envs
                progress_bar.update(num_envs)

            if episode_reward_history:
                recent_rewards = episode_reward_history[-min(100, len(episode_reward_history)):]
                avg_episode_reward = np.mean(recent_rewards)
                reward_history.append(avg_episode_reward)

            with torch.no_grad():
                next_states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
                next_value = agent.get_value(next_states_tensor).reshape(1, -1)
                advantages, returns = agent.compute_gae(batch_rewards, batch_dones, batch_values, next_value.flatten())

            b_states = batch_states.reshape((-1, state_dim))
            b_actions = batch_actions.reshape(-1)
            b_log_probs = batch_log_probs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

            for param_group in agent.optimizer.param_groups:
                frac = 1.0 - (global_step / config.total_timesteps)
                param_group["lr"] = config.learning_rate * max(frac, 0.0)
                
            batch_size = config.steps_per_batch
            minibatch_size = batch_size // config.num_minibatches
            epoch_losses = []
            
            indices = np.random.permutation(batch_size)
            for epoch in range(config.update_epochs):
                for start in range(0, batch_size, minibatch_size):
                    mb_idx = indices[start:start + minibatch_size]
                    mb_advantages = b_advantages[mb_idx]
                    if config.normalize_advantages:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    loss_info = agent.update(
                        b_states[mb_idx], b_actions[mb_idx], 
                        b_log_probs[mb_idx], mb_advantages, b_returns[mb_idx]
                    )
                    epoch_losses.append(loss_info)
            
            #Loss history for logging and monitoring.
            if epoch_losses:
                avg_losses = {
                    'policy_loss': np.mean([l['policy_loss'] for l in epoch_losses]),
                    'value_loss': np.mean([l['value_loss'] for l in epoch_losses]),
                    'entropy_loss': np.mean([l['entropy_loss'] for l in epoch_losses]),
                    'total_loss': np.mean([l['total_loss'] for l in epoch_losses])
                }
                loss_history.append(avg_losses)
            
            update += 1
            
            if update % config.log_interval == 0:
                elapsed_time = time.time() - start_time
                fps = int(global_step / elapsed_time) if elapsed_time > 0 else 0
                current_lr = agent.optimizer.param_groups[0]['lr']
                
                log_msg = f"Update {update:4d} | Step {global_step:7d} | "
                if reward_history:
                    log_msg += f"Reward {reward_history[-1]:7.2f} | "
                if loss_history:
                    log_msg += f"PL {loss_history[-1]['policy_loss']:6.3f} | VL {loss_history[-1]['value_loss']:6.3f} | "
                log_msg += f"LR {current_lr:.2e} | FPS {fps}"
                
                progress_bar.set_description(f"Training Progress - {log_msg}")
                print(log_msg)
                log_file.write(log_msg + "\n")
                log_file.flush()
            
            if update % config.eval_interval == 0:
                avg_reward = evaluate_agent(agent, config.env_name, device=device)
                eval_rewards.append(avg_reward)
                
                eval_msg = f"Evaluation | Update {update:4d} | Avg Reward: {avg_reward:7.2f}"
                print(eval_msg)
                log_file.write(eval_msg + "\n")
                log_file.flush()
                
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    save_checkpoint(agent, update, avg_reward, 'checkpoints/best_model.pth', config)
                    no_improvement_count = 0
                    print(f"New best model saved! Reward: {best_avg_reward:.2f}")
                else:
                    no_improvement_count += 1
                
                if avg_reward >= config.target_reward:
                    print(f"Target reward {config.target_reward} achieved. Stopping training.")
                    break
                    
                if no_improvement_count >= config.early_stopping_patience:
                    print(f"No improvement for {config.early_stopping_patience} evaluations. Early stopping.")
                    break

            if update % config.save_interval == 0:
                save_checkpoint(agent, update, best_avg_reward, f'checkpoints/checkpoint_update_{update}.pth', config)
        
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\nTraining interrupted: {e}")
        save_checkpoint(agent, update, best_avg_reward, 'checkpoints/interrupted_model.pth', config)
        print("Model saved to checkpoints/interrupted_model.pth")
    
    finally:
        envs.close()
        log_file.close()

        save_checkpoint(agent, update, best_avg_reward, 'checkpoints/final_model.pth', config)
        
        with open('logs/training_history.json', 'w') as f:
            json.dump({
                'reward_history': reward_history,
                'eval_rewards': eval_rewards,
                'final_reward': best_avg_reward,
                'total_updates': update,
                'total_timesteps': global_step
            }, f, indent=4)
        
        #Print final statistics.
        print(f"\nTraining completed.")
        print(f"Best average reward: {best_avg_reward:.2f}")
        print(f"Total updates: {update}")
        print(f"Total timesteps: {global_step:,}")
        print(f"Final model saved to: checkpoints/final_model.pth")
        print(f"Best model saved to: checkpoints/best_model.pth")
        print(f"Training history saved to: logs/training_history.json")
        print(f"Use 'python visualization.py' to visualize results.")


if __name__ == "__main__":
    main()

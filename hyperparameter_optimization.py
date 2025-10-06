#This program optimizes the hyperparameters for the SPO agent on the CartPole-v1 environment.
#This program uses the centralized configuration from config.py.

import os
import optuna
import gymnasium as gym
import torch
import numpy as np
import json
from tqdm import tqdm

#Import from the SPOinPyTorch package.
try:
    from SPOinPyTorch import SPOAgent, Config
except ImportError:
    #Fallback for development/local testing.
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from SPOinPyTorch import SPOAgent, Config

#This function evaluates the agent's performance.
def evaluate_agent(agent, env_name, num_episodes=5, device='cuda'):
    device = device if torch.cuda.is_available() else 'cpu'
    env = gym.make(env_name)
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        while not done and steps < 1000:
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
            steps += 1
        total_reward += episode_reward
    env.close()
    return total_reward / num_episodes

#This function saves the agent's checkpoint.
def save_checkpoint(agent, trial_number, update, best_reward, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"trial_{trial_number}_update_{update}.pt")
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'update': update,
        'best_reward': best_reward,
        'trial_number': trial_number
    }, checkpoint_path)
    return checkpoint_path

#This function defines the Optuna objective.
def objective(trial):
    #Suggested hyperparameters ranges.
    num_layers = trial.suggest_int('num_layers', 2, 4)
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    steps_per_batch = trial.suggest_categorical('steps_per_batch', [2048, 4096, 8192])
    update_epochs = trial.suggest_int('update_epochs', 8, 15)
    num_minibatches = trial.suggest_categorical('num_minibatches', [16, 32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.98, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.98)
    epsilon = trial.suggest_float('epsilon', 0.1, 0.3)
    entropy_coeff = trial.suggest_float('entropy_coeff', 0.001, 0.05, log=True)
    value_loss_coeff = trial.suggest_float('value_loss_coeff', 0.4, 0.8)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 2.0)
    normalize_advantages = trial.suggest_categorical('normalize_advantages', [True, False])
    
    #Create configuration.
    config = {
        "env_name": "CartPole-v1",
        "seed": 17,
        "total_timesteps": 100_000,  #Reduced for faster optimization.
        "steps_per_batch": steps_per_batch,
        "update_epochs": update_epochs,
        "num_minibatches": num_minibatches,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "epsilon": epsilon,
        "entropy_coeff": entropy_coeff,
        "value_loss_coeff": value_loss_coeff,
        "max_grad_norm": max_grad_norm,
        "actor_hidden_dims": [hidden_size] * num_layers,
        "critic_hidden_dims": [hidden_size] * num_layers,
        "normalize_advantages": normalize_advantages,
        "eval_interval": 10,
        "target_reward": 475.0,  #Official solved threshold for CartPole-v1.
        "early_stopping_patience": 20
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_envs = 8  #Reduced for faster optimization.
    checkpoint_dir = f"optuna_checkpoints/trial_{trial.number}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    try:
        envs = gym.vector.SyncVectorEnv([
            lambda: gym.make(config["env_name"]) for _ in range(num_envs)
        ])
    except Exception as e:
        print(f"Trial {trial.number} failed to create envs: {e}")
        raise optuna.exceptions.TrialPruned()
    
    is_discrete = isinstance(envs.single_action_space, gym.spaces.Discrete)
    action_low = envs.single_action_space.low if not is_discrete else None
    action_high = envs.single_action_space.high if not is_discrete else None
    
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n if is_discrete else envs.single_action_space.shape[0]
    
    agent = SPOAgent(
        state_dim=state_dim, action_dim=action_dim, config=config,
        is_discrete=is_discrete, action_low=action_low, action_high=action_high, device=device
    )
    
    best_reward = -float("inf")
    global_step = 0
    update = 0
    no_improvement_count = 0
    
    try:
        states, _ = envs.reset()
        episode_rewards = np.zeros(num_envs)
        
        num_steps_per_batch = config["steps_per_batch"] // num_envs
        
        #Training loop.
        while global_step < config["total_timesteps"]:
            batch_states = torch.zeros((num_steps_per_batch, num_envs, state_dim), device=device)
            batch_actions = torch.zeros((num_steps_per_batch, num_envs), device=device, dtype=torch.long)
            batch_log_probs = torch.zeros((num_steps_per_batch, num_envs), device=device)
            batch_rewards = torch.zeros((num_steps_per_batch, num_envs), device=device)
            batch_dones = torch.zeros((num_steps_per_batch, num_envs), device=device)
            batch_values = torch.zeros((num_steps_per_batch, num_envs), device=device)
            
            #Collect experience.
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
                
                states = next_states
                global_step += num_envs

            #Compute advantages and returns.
            with torch.no_grad():
                next_states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
                next_value = agent.get_value(next_states_tensor).reshape(1, -1)
                advantages, returns = agent.compute_gae(batch_rewards, batch_dones, batch_values, next_value.flatten())

            #Prepare batch data.
            b_states = batch_states.reshape((-1, state_dim))
            b_actions = batch_actions.reshape(-1)
            b_log_probs = batch_log_probs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

            #Update learning rate.
            for param_group in agent.optimizer.param_groups:
                frac = 1.0 - (global_step / config["total_timesteps"])
                param_group["lr"] = config["learning_rate"] * max(frac, 0.0)
                
            #Update policy.
            batch_size = config["steps_per_batch"]
            minibatch_size = batch_size // config["num_minibatches"]
            
            indices = np.random.permutation(batch_size)
            for epoch in range(config["update_epochs"]):
                for start in range(0, batch_size, minibatch_size):
                    mb_idx = indices[start:start + minibatch_size]
                    mb_advantages = b_advantages[mb_idx]
                    if config["normalize_advantages"]:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    agent.update(
                        b_states[mb_idx], b_actions[mb_idx], 
                        b_log_probs[mb_idx], mb_advantages, b_returns[mb_idx]
                    )
            
            update += 1
            
            #Evaluate periodically.
            if update % config["eval_interval"] == 0:
                avg_reward = evaluate_agent(agent, config["env_name"], device=device)
                
                #Report intermediate value for pruning.
                trial.report(avg_reward, update)
                
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    save_checkpoint(agent, trial.number, update, best_reward, checkpoint_dir)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if avg_reward >= config["target_reward"]:
                    break
                    
                if no_improvement_count >= config["early_stopping_patience"]:
                    break
        
    except optuna.exceptions.TrialPruned:
        #Normal pruning by Optuna - not an error
        print(f"Trial {trial.number} pruned by Optuna (underperforming).")
        raise
    except Exception as e:
        #Actual runtime error - log full details
        import traceback
        print(f"Trial {trial.number} failed due to runtime error: {repr(e)}")
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned()
    
    finally:
        envs.close()
    
    return best_reward

#This is the main function.
def main():
    print("Starting hyperparameter optimization for SPO agent.")
    print("This may take several minutes to hours depending on the number of trials.")
    
    #Create study.
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    #Run optimization.
    n_trials = 50  #Adjust based on available time/resources.
    study.optimize(objective, n_trials=n_trials)
    
    #Save results.
    print("\nOptimization completed.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.2f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    #Save best hyperparameters.
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    print("\nBest hyperparameters saved to best_hyperparameters.json")
    print("You can now use these hyperparameters for training with train.py")


if __name__ == "__main__":
    main()

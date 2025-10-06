#This program visualizes the training progress of the SPO agent.
#This program uses the centralized configuration from config.py.

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

#This class visualizes the training progress of the SPO agent.
class SPOTrainingVisualizer:
    def __init__(self, log_dir="logs/"):
        self.log_dir = log_dir
        self.data = {}
        plt.style.use('seaborn-v0_8')
    
    #This function loads the training data from the JSON file or parses the log file.
    def load_training_data(self):
        try:
            with open(os.path.join(self.log_dir, 'training_history.json'), 'r') as f:
                self.data = json.load(f)
            print("Loaded training history from training_history.json")
            return True
        except FileNotFoundError:
            print("training_history.json not found. Trying to load from log file...")
            return self.parse_log_file()
    
    #This function parses the log file to extract the training metrics.
    def parse_log_file(self):
        log_file_path = os.path.join(self.log_dir, 'training_log.txt')
        if not os.path.exists(log_file_path):
            print(f"No log file found at {log_file_path}")
            return False
        
        reward_history = []
        eval_rewards = []
        policy_losses = []
        value_losses = []
        steps = []
        eval_steps = []
        
        with open(log_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if 'Update' in line and 'Reward' in line and 'PL' in line:
                    parts = line.split('|')
                    try:
                        step_part = [p for p in parts if 'Step' in p][0]
                        step = int(step_part.split()[-1])
                        
                        reward_part = [p for p in parts if 'Reward' in p][0]
                        reward = float(reward_part.split()[-1])
                        
                        pl_part = [p for p in parts if 'PL' in p][0]
                        pl = float(pl_part.split()[-1])
                        
                        vl_part = [p for p in parts if 'VL' in p][0]
                        vl = float(vl_part.split()[-1])
                        
                        steps.append(step)
                        reward_history.append(reward)
                        policy_losses.append(pl)
                        value_losses.append(vl)
                    except (IndexError, ValueError):
                        continue
                
                elif 'Evaluation' in line and 'Avg Reward' in line:
                    parts = line.split('|')
                    try:
                        update_part = [p for p in parts if 'Update' in p][0]
                        update = int(update_part.split()[-1])
                        
                        reward_part = [p for p in parts if 'Avg Reward' in p][0]
                        reward = float(reward_part.split()[-1])
                        
                        eval_steps.append(update)
                        eval_rewards.append(reward)
                    except (IndexError, ValueError):
                        continue
        
        if reward_history:
            self.data = {
                'reward_history': reward_history,
                'eval_rewards': eval_rewards,
                'policy_losses': policy_losses,
                'value_losses': value_losses,
                'steps': steps,
                'eval_steps': eval_steps
            }
            print(f"Parsed {len(reward_history)} training records and {len(eval_rewards)} evaluation records")
            return True
        
        return False
    
    #This function creates comprehensive training progress plots.
    def plot_training_progress(self, save_plots=True, show_plots=True):
        if not self.data:
            print("No data loaded. Please run load_training_data() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SPO Training Progress', fontsize=16, fontweight='bold')
        
        #Plot 1: Training Rewards.
        ax1 = axes[0, 0]
        if 'reward_history' in self.data and self.data['reward_history']:
            rewards = self.data['reward_history']
            steps = self.data.get('steps', range(len(rewards)))
            ax1.plot(steps, rewards, alpha=0.7, color='blue', label='Training Reward')
            
            #Add rolling average.
            if len(rewards) > 10:
                window = min(50, len(rewards) // 10)
                rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(steps[window-1:], rolling_avg, color='red', linewidth=2, label=f'{window}-Step Average')
            
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Episode Reward')
            ax1.set_title('Training Reward Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        #Plot 2: Evaluation Rewards.
        ax2 = axes[0, 1]
        if 'eval_rewards' in self.data and self.data['eval_rewards']:
            eval_rewards = self.data['eval_rewards']
            eval_steps = self.data.get('eval_steps', range(len(eval_rewards)))
            ax2.plot(eval_steps, eval_rewards, 'o-', color='green', linewidth=2, markersize=6)
            ax2.axhline(y=475, color='red', linestyle='--', alpha=0.8, label='Success Threshold')
            ax2.set_xlabel('Update Number')
            ax2.set_ylabel('Average Evaluation Reward')
            ax2.set_title('Evaluation Performance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        #Plot 3: Policy Loss.
        ax3 = axes[1, 0]
        if 'policy_losses' in self.data and self.data['policy_losses']:
            policy_losses = self.data['policy_losses']
            steps = self.data.get('steps', range(len(policy_losses)))
            ax3.plot(steps, policy_losses, alpha=0.7, color='orange')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Policy Loss')
            ax3.set_title('Policy Loss Over Time')
            ax3.grid(True, alpha=0.3)
        
        #Plot 4: Value Loss.
        ax4 = axes[1, 1]
        if 'value_losses' in self.data and self.data['value_losses']:
            value_losses = self.data['value_losses']
            steps = self.data.get('steps', range(len(value_losses)))
            ax4.plot(steps, value_losses, alpha=0.7, color='purple')
            ax4.set_xlabel('Training Steps')
            ax4.set_ylabel('Value Loss')
            ax4.set_title('Value Loss Over Time')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('spo_training_progress.png', dpi=300, bbox_inches='tight')
            print("Training progress plot saved as 'spo_training_progress.png'")
        
        if show_plots:
            plt.show()
        
        return fig
    
    #This function creates detailed reward analysis plots.
    def plot_reward_analysis(self, save_plots=True, show_plots=True):
        if not self.data or 'eval_rewards' not in self.data:
            print("No evaluation data available for reward analysis.")
            return
        
        eval_rewards = self.data['eval_rewards']
        if not eval_rewards:
            print("No evaluation rewards found.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SPO Reward Analysis', fontsize=16, fontweight='bold')
        
        #Plot 1: Reward progression with variance.
        ax1 = axes[0, 0]
        eval_steps = self.data.get('eval_steps', range(len(eval_rewards)))
        ax1.plot(eval_steps, eval_rewards, 'o-', color='blue', linewidth=2, markersize=6)
        ax1.axhline(y=475, color='red', linestyle='--', alpha=0.8, label='Success Threshold')
        ax1.fill_between(eval_steps, eval_rewards, alpha=0.3, color='blue')
        ax1.set_xlabel('Update Number')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Reward Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        #Plot 2: Reward distribution.
        ax2 = axes[0, 1]
        ax2.hist(eval_rewards, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=np.mean(eval_rewards), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(eval_rewards):.2f}')
        ax2.axvline(x=475, color='orange', linestyle='--', label='Success Threshold')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Reward Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        #Plot 3: Learning curve with confidence intervals.
        ax3 = axes[1, 0]
        if len(eval_rewards) > 5:
            window = max(3, len(eval_rewards) // 10)
            rolling_mean = np.convolve(eval_rewards, np.ones(window)/window, mode='valid')
            rolling_std = np.array([np.std(eval_rewards[max(0, i-window+1):i+1]) 
                                  for i in range(window-1, len(eval_rewards))])
            
            x_vals = eval_steps[window-1:]
            ax3.plot(x_vals, rolling_mean, color='blue', linewidth=2, label='Rolling Mean')
            ax3.fill_between(x_vals, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                           alpha=0.3, color='blue', label='±1 Std Dev')
            ax3.axhline(y=475, color='red', linestyle='--', alpha=0.8, label='Success Threshold')
            ax3.set_xlabel('Update Number')
            ax3.set_ylabel('Reward')
            ax3.set_title(f'Learning Curve (Window: {window})')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        #Plot 4: Performance metrics.
        ax4 = axes[1, 1]
        success_rate = sum(1 for r in eval_rewards if r >= 475) / len(eval_rewards)
        final_performance = np.mean(eval_rewards[-5:]) if len(eval_rewards) >= 5 else np.mean(eval_rewards)
        
        metrics = ['Success Rate', 'Final Performance', 'Max Reward', 'Mean Reward']
        values = [success_rate, final_performance/500, max(eval_rewards)/500, np.mean(eval_rewards)/500]
        colors = ['green', 'blue', 'orange', 'purple']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_ylabel('Normalized Value')
        ax4.set_title('Performance Summary')
        ax4.set_ylim(0, 1)
        
        #Add value labels on bars.
        for bar, value, original in zip(bars, values, [success_rate, final_performance, max(eval_rewards), np.mean(eval_rewards)]):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{original:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('spo_reward_analysis.png', dpi=300, bbox_inches='tight')
            print("Reward analysis plot saved as 'spo_reward_analysis.png'")
        
        if show_plots:
            plt.show()
        
        return fig

#This is the main function.
def main():
    #Parse command line arguments.
    parser = argparse.ArgumentParser(description='Visualize SPO training progress')
    parser.add_argument('--log-dir', default='logs/', help='Directory containing log files')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display plots')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save plots')
    
    args = parser.parse_args()
    
    visualizer = SPOTrainingVisualizer(log_dir=args.log_dir)
    
    if not visualizer.load_training_data():
        print("Failed to load training data. Make sure you have training logs or history files.")
        return
    
    print("Creating training progress plots...")
    visualizer.plot_training_progress(
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    print("Creating reward analysis plots...")
    visualizer.plot_reward_analysis(
        save_plots=not args.no_save,
        show_plots=not args.no_show
    )
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()

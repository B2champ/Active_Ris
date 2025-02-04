import argparse
import numpy as np
import torch
import os
import csv
import re
import env_GC as env  # Import your custom environment
import DDPG as DDPG  # Import the DDPG agent

def whiten(state):
    return (state - np.mean(state)) / np.std(state)

def evaluate_model(args, agent, env):
    """Evaluate the model on the environment."""
    eval_rewards, eval_sumrates, eval_probingpowers = [], [], []
    csv_path = f"./Evaluation_Results/evaluation_results.csv"
    
    if not os.path.exists("./Evaluation_Results"):
        os.makedirs("./Evaluation_Results")
    
    # Open CSV file to save evaluation metrics
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Episode", "Step", "Reward", "SumRate", "ProbingPower"])

    for eps in range(args.num_eval_eps):
        state, done = env.reset(), False
        episode_reward, episode_sumrate, episode_probingpower = 0, 0, 0
        state = whiten(state)

        for t in range(args.num_time_steps_per_eps):
            action = agent.select_action(np.array(state))  # Use deterministic policy
            next_state, reward, done, sum_rate, prob = env.step(action)
            
            episode_reward += reward
            episode_sumrate += sum_rate
            episode_probingpower += prob

            state = whiten(next_state)

            # Write metrics to CSV
            with open(csv_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([eps + 1, t + 1, reward, sum_rate, prob])
            
            if done:
                break

        eval_rewards.append(episode_reward / args.num_time_steps_per_eps)
        eval_sumrates.append(episode_sumrate / args.num_time_steps_per_eps)
        eval_probingpowers.append(episode_probingpower / args.num_time_steps_per_eps)
        
        print(f"Episode {eps + 1}: Reward = {episode_reward:.3f}, Avg SumRate = {episode_sumrate:.3f}, Avg ProbingPower = {episode_probingpower:.3f}")
    
    return eval_rewards, eval_sumrates, eval_probingpowers

def find_model_files(models_dir):
    """Find actor and critic files in the models directory."""
    actor_file, critic_file = None, None
    for file_name in os.listdir(models_dir):
        if re.search(r'_actor$', file_name):
            actor_file = os.path.join(models_dir, file_name)
        elif re.search(r'_critic$', file_name):
            critic_file = os.path.join(models_dir, file_name)
    if not actor_file or not critic_file:
        raise FileNotFoundError("Actor or Critic file not found in the models directory.")
    return actor_file, critic_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models_dir", nargs='?', default="./models/", help="Directory containing the trained models")
    parser.add_argument("--num_antennas", default=8, type=int, help="Number of antennas in the BS")
    parser.add_argument("--num_RIS_elements", default=16, type=int, help="Number of RIS elements")
    parser.add_argument("--num_users", default=5, type=int, help="Number of users")
    parser.add_argument("--num_groups", default=4, type=int, help="Number of groups")
    parser.add_argument("--num_time_steps_per_eps", default=10, type=int, help="Number of steps per episode")
    parser.add_argument("--num_eval_eps", default=10, type=int, help="Number of evaluation episodes")
    parser.add_argument("--awgn_var", default=1e-3, type=float, help="AWGN variance")
    parser.add_argument("--gpu", default="0", type=int, help="GPU ordinal")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    env = env.RIS_MISO(
        num_antennas=args.num_antennas,
        num_RIS_elements=args.num_RIS_elements,
        num_groups=args.num_groups,
        num_users=args.num_users,
        AWGN_var=args.awgn_var,
        carrfreq=10**9,
        alpha_t=2.2,
        alpha_r=2.2,
        Rican_BR=10,
        Rican_RU=10,
        Rican_BU=10,
        h_ris=20,
        h_base=10,
        x_bs=0, y_bs=0,
        x_ris=0, y_ris=150, 
        max_steps=100
    )


    # Initialize DDPG agent
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1

    agent = DDPG.DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        power_t=10.0,  # Example transmission power
        max_action=max_action,
        M=args.num_antennas,
        N=args.num_RIS_elements,
        K=args.num_users,
        group=args.num_groups,
        actor_lr=1e-3,  # Match training parameters
        critic_lr=1e-3,
        actor_decay=1e-5,
        critic_decay=1e-5,
        device=device,
        discount=0.99,
        tau=1e-3
    )

    # Find model files
    actor_path, critic_path = find_model_files(args.models_dir)

    print("Loading models...")
    agent.actor.load_state_dict(torch.load(actor_path))
    agent.critic.load_state_dict(torch.load(critic_path))
    agent.actor.eval()
    agent.critic.eval()
    print(f"Models loaded successfully:\nActor: {actor_path}\nCritic: {critic_path}")

    # Evaluate the model
    print("Starting evaluation...")
    rewards, sumrates, probingpowers = evaluate_model(args, agent, env)

    # Save evaluation results
    np.savetxt(f"./Evaluation_Results/avg_rewards.csv", rewards, delimiter=',')
    np.savetxt(f"./Evaluation_Results/avg_sumrates.csv", sumrates, delimiter=',')
    np.savetxt(f"./Evaluation_Results/avg_probingpowers.csv", probingpowers, delimiter=',')
    print("Evaluation complete. Results saved.")

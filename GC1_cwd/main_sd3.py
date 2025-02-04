import argparse
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import SD3 as SD3
import utils
import csv
import math as mt

import env_GC as env


def whiten(state):
    return (state - np.mean(state)) / np.std(state)

# Argument parser ___Info________________________________________________________________________________________________________________________________________________________________
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Choose the type of the experiment <This will reproduce the type of the experiment given in the paper>
    parser.add_argument('--experiment_type', default='power', choices=['custom', 'power', 'rsi_elements', 'learning_rate', 'decay', 'sum_rate_ris'],
                        help='Choose one of the experiment types to reproduce the learning curves given in the paper')
    # Training-specific parameters
    parser.add_argument("--policy", default="SD3", help='Algorithm (default: DDPG)')
    parser.add_argument("--env", default="RIS_MISO", help='OpenAI Gym environment_fullyConnected name')
    parser.add_argument("--seed", default=0, type=int, help='Seed number for PyTorch and NumPy (default: 0)')
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')
    parser.add_argument("--start_time_steps", default=0, type=int, metavar='N', help='Number of exploration time steps sampling random actions (default: 0)')
    parser.add_argument("--buffer_size", default=100000, type=int, help='Size of the experience replay buffer (default: 100000)')
    parser.add_argument("--batch_size", default=32, metavar='N', help='Batch size (default: 16)')
    parser.add_argument("--save_model", action="store_true", help='Save model and optimizer parameters')
    parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')
    # environment_fullyConnected-specific parameters
    parser.add_argument("--num_antennas", default=8, type=int, metavar='N', help='Number of antennas in the BS')
    parser.add_argument("--num_RIS_elements", default=16, type=int, metavar='N', help='Number of RIS elements')
    parser.add_argument("--num_groups", default=1, type=int, metavar='Groups', help='Number of Groups')
    parser.add_argument("--num_users", default=4, type=int, metavar='N', help='Number of users')
    parser.add_argument("--power_t", default=mt.pow(10, (50-30)/10), type=float, metavar='N', help='Transmission power for the constrained optimization in dB') #This needed to be changed to 20dBm
    parser.add_argument("--num_time_steps_per_eps", default=1000, type=int, metavar='N', help='Maximum number of steps per episode (default: 20000)')
    parser.add_argument("--num_eps", default=200, type=int, metavar='N', help='Maximum number of episodes (default: 1000)')
    parser.add_argument("--awgn_var", default=mt.pow(10, -((140-30)/10)), type=float, metavar='G', help='Variance of the additive white Gaussian noise (default: 0.01)')
    parser.add_argument("--channel_est_error", default=False, type=bool, help='Noisy channel estimate? (default: False)')
    # Algorithm-specific parameters
    parser.add_argument("--exploration_noise", default=0.01, metavar='G', help='Std of Gaussian exploration noise')
    parser.add_argument("--discount", default=0.99, metavar='G', help='Discount factor for reward (default: 0.99)')
    parser.add_argument("--tau", default=1e-3, type=float, metavar='G',  help='Learning rate in soft/hard updates of the target networks (default: 0.001)')
    parser.add_argument("--lr", default=1e-4, type=float, metavar='G', help='Learning rate for the networks (default: 0.001)')
    parser.add_argument("--decay", default=1e-6, type=float, metavar='G', help='Decay rate for the networks (default: 0.00001)')
    args = parser.parse_args()

    # Argument parser Over ___________________________________________________________________________________________________________________________________________________________________

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")



    file_name = f"{args.num_antennas}_{args.num_RIS_elements}_{args.num_users}_{args.power_t}_{args.lr}_{args.decay}"

        # Folder Creation Specific Parts___________________________________________________________________________________________________________________________________________________

    if not os.path.exists(f"./Training Curves/{args.experiment_type}"):
        os.makedirs(f"./Training Curves/{args.experiment_type}")


    if not os.path.exists(f"./Training_Values_{args.power_t}_{args.num_RIS_elements}_{args.num_users}"):
        os.makedirs(f"./Training_Values_{args.power_t}_{args.num_RIS_elements}_{args.num_users}")


    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

        # Folder Creation Over ____________________________________________________________________________________________________________________________________________________________


    csv_path = f"./Training_Values_{args.power_t}_{args.num_RIS_elements}_{args.num_users}/{file_name}_metrics.csv"
    csv_path_s = f"./Training_Values_{args.power_t}_{args.num_RIS_elements}_{args.num_users}/{file_name}_metrics_sumrate.csv"
    csv_path_p = f"./Training_Values_{args.power_t}_{args.num_RIS_elements}_{args.num_users}/{file_name}_metrics_probing.csv"

    # with open(csv_path, mode='w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(["Episode", "Step", "Instantaneous Reward"])

    # with open(csv_path_s, mode='w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(["Episode", "Step", "Instantaneous Sumrate"])

    # with open(csv_path_p, mode='w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(["Episode", "Step", "Instantaneous_probingPow"])


    env = env.RIS_MISO(args.num_antennas, args.num_RIS_elements,args.num_groups, args.num_time_steps_per_eps, args.num_users, AWGN_var=args.awgn_var,carrfreq=10**9,
                 alpha_t=2.2,
                 alpha_r=2.2,
                 Rican_BR=10,
                 Rican_RU=10,
                 Rican_BU=10,
                 h_ris=20,
                 h_base=10,
                 x_bs=0, y_bs=0,
                 x_ris=0, y_ris=150)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.state_dim


    action_dim = env.action_dim
    max_action = 1
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "power_t": args.power_t,
        "max_action": max_action,
        "M": args.num_antennas,
        "N": args.num_RIS_elements,
        "K": args.num_users,
        "group": args.num_groups,
        "actor_lr": args.lr,
        "critic_lr": args.lr,
        "actor_decay": args.decay,
        "critic_decay": args.decay,
        "device": device,
        "discount": args.discount,
        "tau": args.tau
    }
    # Initialize the algorithm
    agent = SD3.SD3(**kwargs)
    #____________________________________________________________Model Loading Specific Parts___________________________________________________________________________________________________
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        agent.load(f"./models/{policy_file}")
    #_____________________________________________________________Model Loading Over _____________________________________________________________________________________________________________________________________________________________
   
    replay_buffer = utils.ExperienceReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)

    writer = SummaryWriter(log_dir=f"./Tensorboard_DDPG/{args.experiment_type}/tensorboard_logs")  

    instant_rewards, instant_sumRate, instant_probingPower = [],[], []
    avg_rewards, avg_sumRate, avg_probingPower = [],[], []
    
    max_reward = 0

    for eps in range(int(args.num_eps)):
        state, done = env.reset(), False
        
        episode_reward, episode_sumrate, episode_probingpower = 0,0,0
        episode_num = eps
        episode_time_steps = 0

        state = whiten(state)

        eps_rewards = []
        eps_sumrate, eps_probingpower = [],[]

        for t in range(int(args.num_time_steps_per_eps)):
            # Choose action from the policy
            action = agent.select_action(np.array(state))
        
            # Take the selected action
            next_state, reward, done, sum_rate, prob = env.step(action)

            done = 1.0 if t == args.num_time_steps_per_eps - 1 else float(done) #Done condition already no need to add in Enviroment. Those are added for the ideal case.
           
            replay_buffer.add(state, action, next_state, reward, done)   # Store data in the experience replay buffer

            state = next_state

            #________________________________________________________________________ Summarize the reward over the episode_____________________________________________________________

            episode_reward += reward
            episode_sumrate += sum_rate
            episode_probingpower += prob

            state = whiten(state)

            if reward > max_reward:
                max_reward = reward

            
            agent.update_parameters(replay_buffer, args.batch_size) # Train the agent
            
            writer.add_scalar('Reward per Steps', reward, eps * args.num_time_steps_per_eps + t) #Log Reward to Tensorboard
            #________________________________________________________________ Write data in CSV File:__________________________________________________________________________________ 
            # with open(csv_path, mode='a', newline='') as csv_file:
            #     csv_writer = csv.writer(csv_file)
            #     csv_writer.writerow([eps + 1, t + 1, reward])

            # with open(csv_path_s, mode='a', newline='') as csv_file:
            #     csv_writer = csv.writer(csv_file)
            #     csv_writer.writerow([eps + 1, t + 1, sum_rate])

            # with open(csv_path_p, mode='a', newline='') as csv_file:
            #     csv_writer = csv.writer(csv_file)
            #     csv_writer.writerow([eps + 1, t + 1, prob])

            

            # #________________________________________________________________ Write data in CSV File:__________________________________________________________________________________

            print(f"Time step: {t + 1} Episode Num: {episode_num + 1} Reward: {reward:.3f}")
    
            eps_rewards.append(reward)
            eps_sumrate.append(sum_rate)
            eps_probingpower.append(prob)


            episode_time_steps += 1

            if done:
                print(f"\nTotal T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_time_steps} Max. Reward: {max_reward:.3f}\n")

                # Reset the environment_fullyConnected
                state, done = env.reset(), False
                episode_time_steps = 0
                episode_num += 1
                state = whiten(state)



        avg_rew = episode_reward/args.num_time_steps_per_eps        
        avg_sr = episode_sumrate/args.num_time_steps_per_eps
        avg_pr = episode_probingpower/args.num_time_steps_per_eps 

        avg_rewards.append(avg_rew)
        avg_sumRate.append(avg_sr)
        avg_probingPower.append(avg_pr)

        instant_rewards.append(eps_rewards)
        instant_sumRate.append(eps_sumrate)
        instant_probingPower.append(eps_probingpower)


        #________________________________________________________________ Write data in CSV File:__________________________________________________________________________________
    
        np.savetxt("Max_Reward.csv", [max_reward], delimiter=',')
        np.savetxt( "Average_Reward.csv", avg_rewards, delimiter=',' )
        np.savetxt( "Avg_SumRate.csv", avg_sumRate, delimiter=',' )
        np.savetxt( "Avg_PobingPower.csv", avg_probingPower, delimiter=',' )



        writer.add_scalar("Reward/Average per Episode", avg_rew, eps + 1)
        
   
    # if args.save_model:
    agent.save(f"./models/{file_name}")

    writer.close()
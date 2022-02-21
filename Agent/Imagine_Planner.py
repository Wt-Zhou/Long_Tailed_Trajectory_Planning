import argparse

import numpy as np
import math
import torch
import json
import os
import os.path as osp
import tensorflow as tf
import tempfile
import time
import random
import _thread
import baselines.common.tf_util as U
import random
import matplotlib.pyplot as plt
from rtree import index as rindex
from collections import deque
from scipy import stats
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from Agent.model import dqn_model, bootstrap_model
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.actions import LaneAction
from Agent.world_model.world_model import World_Model

class GMBRL(object):

    def __init__(self, world_model):
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.target_speed = 30/3.6 
        self.world_model = world_model

    def parse_args(self):
        parser = argparse.ArgumentParser("DQN experiments for Atari games")
        # Environment
        parser.add_argument("--env", type=str, default="Town03", help="name of the game")
        parser.add_argument("--seed", type=int, default=42, help="which seed to use")
        parser.add_argument("--decision_count", type=int, default=1, help="how many steps for a decision")
        # Core DQN parameters
        parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
        parser.add_argument("--train-buffer-size", type=int, default=int(1e8), help="train buffer size")
        parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for Adam optimizer")
        parser.add_argument("--num-steps", type=int, default=int(4e7), help="total number of steps to run the environment for")
        parser.add_argument("--batch-size", type=int, default=64, help="number of transitions to optimize at the same time")
        parser.add_argument("--learning-freq", type=int, default=20, help="number of iterations between every optimization step")
        parser.add_argument("--target-update-freq", type=int, default=50, help="number of iterations between every target network update") #10000
        parser.add_argument("--learning-starts", type=int, default=50, help="when to start learning") 
        parser.add_argument("--gamma", type=float, default=0.995, help="the gamma of q update") 
        parser.add_argument("--bootstrapped-data-sharing-probability", type=float, default=0.8, help="bootstrapped_data_sharing_probability") 
        parser.add_argument("--bootstrapped-heads-num", type=int, default=10, help="bootstrapped head num of networks") 
        parser.add_argument("--learning-repeat", type=int, default=10, help="learn how many times from one sample of RP") 
        # Bells and whistles
        boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
        boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
        boolean_flag(parser, "bootstrap", default=True, help="whether or not to use bootstrap model")
        boolean_flag(parser, "prioritized", default=True, help="whether or not to use prioritized replay buffer")
        parser.add_argument("--prioritized-alpha", type=float, default=0.9, help="alpha parameter for prioritized replay buffer")
        parser.add_argument("--prioritized-beta0", type=float, default=0.1, help="initial value of beta parameters for prioritized replay")
        parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
        # Checkpointing
        parser.add_argument("--save-dir", type=str, default="./logs", help="directory in which training state and model should be saved.")
        parser.add_argument("--save-azure-container", type=str, default=None,
                            help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
        parser.add_argument("--save-freq", type=int, default=5000, help="save model once every time this many iterations are completed")
        boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
        return parser.parse_args()

    def maybe_save_model(self, savedir, state):
        """This function checkpoints the model and state of the training algorithm."""
        if savedir is None:
            return
        start_time = time.time()
        model_dir = "model-{}".format(state["num_iters"])
        U.save_state(os.path.join(savedir, model_dir, "saved"))
        state_dir = "training_state.pkl-{}".format(state["num_iters"]) + ".zip"
        relatively_safe_pickle_dump(state, os.path.join(savedir, state_dir), compression=True)
        logger.log("Saved model in {} seconds\n".format(time.time() - start_time))

    def maybe_load_model(self, savedir, model_step):
        """Load model if present at the specified path."""
        if savedir is None:
            return
        model_dir = "training_state.pkl-{}".format(model_step) + ".zip"
        # state_path = os.path.join(os.path.join(savedir, 'training_state.pkl-100028.zip'))
        state_path = os.path.join(os.path.join(savedir, model_dir))
        found_model = os.path.exists(state_path)
        if found_model:
            state = pickle_load(state_path, compression=True)
            model_dir = "model-{}".format(state["num_iters"])
            U.load_state(os.path.join(savedir, model_dir, "saved"))
            logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
            return state

    def test(self, model_list, test_steps, env):
        # Init DRL
        args = self.parse_args()
        savedir = args.save_dir + "_" + args.env
        save_model = True

        if args.seed > 0:
            set_global_seeds(args.seed)
        
        with U.make_session(120) as sess:
            # Create training graph and replay buffer
            act_dqn, train_dqn, update_target_dqn, q_values_dqn = deepq.build_train_dqn(
                make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
                original_dqn=dqn_model,
                num_actions=env.action_space.n,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=args.gamma,
                grad_norm_clipping=10,
                double_q=args.double_q
            )
            
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(args.num_steps / 50, initial_p=args.prioritized_beta0, final_p=1.0) # Approximately Iters
            
            learning_rate = args.lr # Maybe Segmented

            U.initialize()
            num_iters = 0

            for model_step in model_list:
                # Load the model
                state = self.maybe_load_model(savedir, model_step)
                if state is not None:
                    num_iters, replay_buffer = state["num_iters"], state["replay_buffer"]
        
                start_time, start_steps = None, None
                test_iters = 0

                obs = env.reset()
                self.trajectory_planner.clear_buff()

                model_reward = 0

                # Test
                while test_iters < test_steps:
                    num_iters += 1
                    obs = np.array(obs)
                    # Rule-based Planner
                    self.dynamic_map.update_map_from_obs(obs, env)
                    rule_trajectory = self.trajectory_planner.trajectory_update(self.dynamic_map)

                    # Bootstapped Action
                    q_list = q_values_dqn(obs[None])
                    action = np.array(np.where(q_list[0]==np.max(q_list[0]))[0])
                
                    print("[DQN]: Obs",obs.tolist())
                    print("[DQN]: Q_List",q_list)
                    print("[DQN]: Action",action)

                    if len(action) > 1:
                        action = np.array(action[0])

                    # Control
                    trajectory = self.trajectory_planner.trajectory_update_CP(action, rule_trajectory)
                    for i in range(args.decision_count):
                        control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                        output_action = [control_action.acc, control_action.steering]
                        new_obs, rew, done, info = env.step(output_action)
                        model_reward += rew

                        if done:
                            break
                        self.dynamic_map.update_map_from_obs(new_obs, env)

                    obs = new_obs
                    if done:
                        # self.record_termianl_data(model_step, obs, action, rew, q_values, q_values_dqn, self.rtree) # before update obs
                        obs = env.reset()
                        self.trajectory_planner.clear_buff()
                        test_iters += 1

                    # Record Data    
                self.record_test_data(model_step, model_reward, env)
    
    def collect_corner_data(self, env):
        # Collect Corner Data
        args = self.parse_args()
        savedir = args.save_dir + "_" + args.env

        corner_buffer_size = 1000
        self.corner_buffer = ReplayBuffer(corner_buffer_size)

        obs = env.reset()
        self.trajectory_planner.clear_buff()
        decision_count = 0
        while True:
            obs = np.array(obs)

            print("Obs",obs.tolist())

            # Rule-based Planner
            self.dynamic_map.update_map_from_obs(obs, env)
            rule_trajectory, action = self.trajectory_planner.trajectory_update(self.dynamic_map)
            trajectory = self.trajectory_planner.trajectory_update_CP(action, rule_trajectory)
            # Control
            for i in range(args.decision_count):
                control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                output_action = [control_action.acc, control_action.steering]
                new_obs, rew, done, info = env.step(output_action)
                if done:
                    break
                self.dynamic_map.update_map_from_obs(new_obs, env)

            mask = np.random.binomial(1, args.bootstrapped_data_sharing_probability, args.bootstrapped_heads_num) # add mask for data
            self.corner_buffer.add(obs, action, rew, new_obs, float(done), mask)
            obs = new_obs

            if done:
                # self.maybe_save_model(savedir, {
                #     'corner_buffer': self.corner_buffer,
                # })
                print("Finish Corner Data Collection")
                break
        return self.corner_buffer
    
    def learn(self, total_timesteps, imagine_env, load_model_step):      
        # Init DRL
        args = self.parse_args()
        savedir = args.save_dir + "_" + args.env
        save_model = True


        if args.seed > 0:
            set_global_seeds(args.seed)
        
        with U.make_session(120) as sess:
        # Create training graph and replay buffer
            act_dqn, train_dqn, update_target_dqn, q_values_dqn = deepq.build_train_dqn(
                make_obs_ph=lambda name: U.CARLAInput(imagine_env.env.observation_space, name=name),
                original_dqn=dqn_model,
                num_actions=imagine_env.env.action_space.n,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=args.gamma,
                grad_norm_clipping=10,
                double_q=args.double_q
            )
            
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(args.num_steps / 50, initial_p=args.prioritized_beta0, final_p=1.0) # Approximately Iters

            learning_rate = args.lr # maybe Segmented

            U.initialize()
            update_target_dqn()
            num_iters = 0

            # Load the model
            state = self.maybe_load_model(savedir, load_model_step)
            if state is not None:
                num_iters, replay_buffer = state["num_iters"], state["replay_buffer"]
            
            start_time, start_steps = None, None
            obs = imagine_env.reset()

            fig, ax = plt.subplots()

            while num_iters < total_timesteps:
                num_iters += 1
                obs = np.array(obs)

                # DQN Action
                q_list = q_values_dqn(obs[None])
                optimal_action = np.array(np.where(q_list[0]==np.max(q_list[0]))[0])
                if len(optimal_action) > 1:
                    optimal_action = np.array([optimal_action[0]])

                random_action = np.array([random.randint(0,6)])

                if random.uniform(0,1) < 0.5: # epsilon-greddy
                    action = random_action
                else:
                    action = optimal_action

                print("[DQN]: Obs",obs.tolist())
                print("[DQN]: DQN Action",action)
                print("[DQN]: Q_List",q_list[0])

                
                new_obs, rew, done, _ = imagine_env.step(action)

                # Draw Plot
                ax.cla() 
                if abs(new_obs.tolist()[0] - obs.tolist()[0]) > 0:
                    angle = math.atan((-new_obs.tolist()[1]+obs.tolist()[1])/(new_obs.tolist()[0] - obs.tolist()[0]) )/math.pi*180 + 180
                else:
                    angle = obs.tolist()[4]/math.pi*180
                rect = plt.Rectangle((obs.tolist()[0],-obs.tolist()[1]),2.5,6,angle=angle+90)
                ax.add_patch(rect)
                if abs(new_obs.tolist()[5] - obs.tolist()[5]) > 0.1:
                    angle2 = math.atan((-new_obs.tolist()[6]+obs.tolist()[6])/(new_obs.tolist()[5] - obs.tolist()[5]) )/math.pi*180 + 180
                else:
                    angle2 = obs.tolist()[9]/math.pi*180 + 45

                # rect = plt.Rectangle((obs.tolist()[5],-obs.tolist()[6]),2,5,angle=obs.tolist()[9]/math.pi*180)
                rect = plt.Rectangle((obs.tolist()[5],-obs.tolist()[6]),2.5,6,angle=angle2+90, facecolor="red")
                ax.add_patch(rect)
                ax.axis([-92,-13,-199,-137])
                ax.legend()
                plt.pause(0.001)
                    
                mask = np.random.binomial(1, args.bootstrapped_data_sharing_probability, args.bootstrapped_heads_num) # add mask for data
                replay_buffer.add(obs, action[0], rew, new_obs, float(done), mask)
                obs = new_obs
                if done:
                    random_head = np.random.randint(args.bootstrapped_heads_num)
                    if save_model == True:
                        print("[DQN]: Save model")
                        self.maybe_save_model(savedir, {
                            'replay_buffer': replay_buffer,
                            'num_iters': num_iters,
                        })
                        save_model = False
                    obs = imagine_env.reset()
                    self.trajectory_planner.clear_buff()

                if (num_iters > args.learning_starts and
                        num_iters % args.learning_freq == 0):
                    # Sample a bunch of transitions from replay buffer
                    if args.prioritized:
                        # Update rl
                        if replay_buffer.__len__() > args.batch_size:
                            for i in range(args.learning_repeat):
                                print("[DQN]: Learning")
                                experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters), count_train=True)
                                (obses_t, actions, rewards, obses_tp1, dones, masks, train_time, weights, batch_idxes) = experience
                                # Minimize the error in Bellman's equation and compute TD-error
                                td_errors_dqn, q_t_selected_dqn, q_t_selected_target_dqn, qt_dqn = train_dqn(obses_t, actions, rewards, obses_tp1, dones, masks, weights, learning_rate)
                                # Update the priorities in the replay buffer
                                new_priorities = np.abs(td_errors_dqn) + args.prioritized_eps

                                replay_buffer.update_priorities(batch_idxes, new_priorities)
                    
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                        weights = np.ones_like(rewards)
                    
                    
                # Update target network.
                if num_iters % args.target_update_freq == 0:
                    print("[DQN]: Update target network")
                    update_target_dqn()
               
                start_time, start_steps = time.time(), 0

                # Save the model and training state.
                if num_iters >= 0 and num_iters % args.save_freq == 0:
                    save_model = True
   
    def learn_dqn(self, total_timesteps, env, load_model_step):      
        # Init DRL
        args = self.parse_args()
        savedir = args.save_dir + "_" + args.env
        save_model = True


        if args.seed > 0:
            set_global_seeds(args.seed)
        
        with U.make_session(120) as sess:
        # Create training graph and replay buffer
            act_dqn, train_dqn, update_target_dqn, q_values_dqn = deepq.build_train_dqn(
                make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
                original_dqn=dqn_model,
                num_actions=env.action_space.n,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=args.gamma,
                grad_norm_clipping=10,
                double_q=args.double_q
            )
            
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(args.num_steps / 50, initial_p=args.prioritized_beta0, final_p=1.0) # Approximately Iters

            learning_rate = args.lr # maybe Segmented

            U.initialize()
            update_target_dqn()
            num_iters = 0

            # Load the model
            state = self.maybe_load_model(savedir, load_model_step)
            if state is not None:
                num_iters, replay_buffer = state["num_iters"], state["replay_buffer"]
            
            start_time, start_steps = None, None
            obs = env.reset()

            episode_reward = 0

            while num_iters < total_timesteps:
                obs = np.array(obs)

                # # Rule-based Planner
                self.dynamic_map.update_map_from_obs(obs, env)
                rule_trajectory = self.trajectory_planner.trajectory_update(self.dynamic_map)

                # DQN Action
                q_list = q_values_dqn(obs[None])
                optimal_action = np.array(np.where(q_list[0]==np.max(q_list[0]))[0])
                if len(optimal_action) > 1:
                    optimal_action = np.array([optimal_action[0]])

                random_action = np.array([random.randint(0,6)])

                if random.uniform(0,1) < 0.5: # epsilon-greddy
                    action = random_action
                else:
                    action = optimal_action

                print("[DQN]: Obs",obs.tolist())
                print("[DQN]: DQN Action",action)
                print("[DQN]: Q_List",q_list[0])

                # Control
                trajectory = self.trajectory_planner.trajectory_update_CP(action[0], rule_trajectory)
                for i in range(args.decision_count):
                    control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                    output_action = [control_action.acc, control_action.steering]
                    new_obs, rew, done, info = env.step(output_action)

                    episode_reward += rew
                    if done:
                        break
                    self.dynamic_map.update_map_from_obs(new_obs, env)
                    
                mask = np.random.binomial(1, args.bootstrapped_data_sharing_probability, args.bootstrapped_heads_num) # add mask for data
                replay_buffer.add(obs, action[0], rew, new_obs, float(done), mask)
                obs = new_obs

                if done:
                    
                    self.trajectory_planner.clear_buff()
                    env.reset()
                    fw = open("reward.txt", 'a')   
                    # Write num

                    fw.write(str(episode_reward)) 
                    fw.write("\n")
                    fw.close()        
                    episode_reward = 0

                if (num_iters > args.learning_starts and
                        num_iters % args.learning_freq == 0):
                    # Sample a bunch of transitions from replay buffer
                    if args.prioritized:
                        # Update rl
                        if replay_buffer.__len__() > args.batch_size:
                            for i in range(args.learning_repeat):
                                print("[DQN]: Learning")
                                experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters), count_train=True)
                                (obses_t, actions, rewards, obses_tp1, dones, masks, train_time, weights, batch_idxes) = experience
                                # Minimize the error in Bellman's equation and compute TD-error
                                td_errors_dqn, q_t_selected_dqn, q_t_selected_target_dqn, qt_dqn = train_dqn(obses_t, actions, rewards, obses_tp1, dones, masks, weights, learning_rate)
                                # Update the priorities in the replay buffer
                                new_priorities = np.abs(td_errors_dqn) + args.prioritized_eps

                                replay_buffer.update_priorities(batch_idxes, new_priorities)
                    
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                        weights = np.ones_like(rewards)
                    
                    
                # Update target network.
                if num_iters % args.target_update_freq == 0:
                    print("[DQN]: Update target network")
                    update_target_dqn()
               
                start_time, start_steps = time.time(), 0

                # Save the model and training state.
                if num_iters >= 0 and num_iters % args.save_freq == 0:
                    print("[DQN]: Save model")
                    self.maybe_save_model(savedir, {
                        'replay_buffer': replay_buffer,
                        'num_iters': num_iters,
                    })
                num_iters += 1

class Log_Replay_Imagine_Model:

    def __init__(self, corner_buffer):
        
        # Imagine Model
        experience = corner_buffer._storage[0]
        obs_e, action_e, rew, new_obs, done, masks, train_times = experience
        self.corner_buffer = corner_buffer
        self.s_0 = obs_e # Start State
        print("Start State",self.s_0)
        self.current_s = self.s_0
        self.simulation_step = 0

        # Planner
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.target_speed = 30/3.6 

    def load_world_model(self, env, load_step):
        args = self.parse_args()

        self.env = env

        self.set_seed_everywhere(args.seed)
        self.make_dir(args.work_dir)
        model_dir = self.make_dir(os.path.join(args.work_dir, 'world_model'))
        buffer_dir = self.make_dir(os.path.join(args.work_dir, 'world_buffer'))

        with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        replay_buffer = World_Buffer(
            obs_shape=env.observation_space.shape,
            action_shape=[1], # discrete, 1 dimension!
            capacity= args.replay_buffer_capacity,
            batch_size= args.batch_size,
            device=device
        )
        self.world_model = World_Model(obs_shape=env.observation_space.shape,
            action_shape=[1], # discrete, 1 dimension!
            state_space_dim=env.state_dimension,
            device=device,
            transition_model_type = args.transition_model_type,
            env=env)

        try:
            self.world_model.load(model_dir, load_step)
            print("[World_Model] : Load learned model successful, step=",load_step)
        except:
            pass

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # FIXME:Should be the same with CP
        parser.add_argument("--decision_count", type=int, default=1, help="how many steps for a decision")

        # environment
        parser.add_argument('--domain_name', default='carla')
        parser.add_argument('--task_name', default='run')
        parser.add_argument('--image_size', default=84, type=int)
        parser.add_argument('--action_repeat', default=1, type=int)
        parser.add_argument('--frame_stack', default=1, type=int) #3
        parser.add_argument('--resource_files', type=str)
        parser.add_argument('--eval_resource_files', type=str)
        parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
        parser.add_argument('--total_frames', default=1000, type=int)
        # replay buffer
        parser.add_argument('--replay_buffer_capacity', default=10000, type=int)
        # train
        parser.add_argument('--agent', default='bisim', type=str, choices=['baseline', 'bisim', 'deepmdp'])
        parser.add_argument('--init_steps', default=10, type=int)
        parser.add_argument('--num_train_steps', default=1000000, type=int)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--hidden_dim', default=256, type=int)
        parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
        parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
        parser.add_argument('--load_encoder', default=None, type=str)
        # eval
        parser.add_argument('--eval_freq', default=20000, type=int)  # TODO: master had 10000
        parser.add_argument('--num_eval_episodes', default=20, type=int)
        # critic
        parser.add_argument('--critic_lr', default=1e-3, type=float)
        parser.add_argument('--critic_beta', default=0.9, type=float)
        parser.add_argument('--critic_tau', default=0.005, type=float)
        parser.add_argument('--critic_target_update_freq', default=2, type=int)
        # actor
        parser.add_argument('--actor_lr', default=1e-3, type=float)
        parser.add_argument('--actor_beta', default=0.9, type=float)
        parser.add_argument('--actor_log_std_min', default=-10, type=float)
        parser.add_argument('--actor_log_std_max', default=2, type=float)
        parser.add_argument('--actor_update_freq', default=2, type=int)
        # encoder/decoder
        parser.add_argument('--encoder_type', default='pixelCarla098', type=str, choices=['pixel', 'pixelCarla096', 'pixelCarla098', 'identity'])
        parser.add_argument('--encoder_feature_dim', default=50, type=int)
        parser.add_argument('--encoder_lr', default=1e-3, type=float)
        parser.add_argument('--encoder_tau', default=0.005, type=float)
        parser.add_argument('--encoder_stride', default=1, type=int)
        parser.add_argument('--decoder_type', default='pixel', type=str, choices=['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction'])
        parser.add_argument('--decoder_lr', default=1e-3, type=float)
        parser.add_argument('--decoder_update_freq', default=1, type=int)
        parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
        parser.add_argument('--num_layers', default=4, type=int)
        parser.add_argument('--num_filters', default=32, type=int)
        # sac
        parser.add_argument('--discount', default=0.99, type=float)
        parser.add_argument('--init_temperature', default=0.01, type=float)
        parser.add_argument('--alpha_lr', default=1e-3, type=float)
        parser.add_argument('--alpha_beta', default=0.9, type=float)
        # misc
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--work_dir', default='.', type=str)
        parser.add_argument('--save_tb', default=False, action='store_true')
        parser.add_argument('--save_model', default=True, action='store_true')
        parser.add_argument('--save_buffer', default=False, action='store_true')
        parser.add_argument('--save_video', default=False, action='store_true')
        parser.add_argument('--transition_model_type', default='probabilistic', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
        parser.add_argument('--render', default=False, action='store_true')
        parser.add_argument('--port', default=2000, type=int)
        args = parser.parse_args()
        return args

    def update_transition_and_reward(self, env):
        args = self.parse_args()

        self.set_seed_everywhere(args.seed)
        self.make_dir(args.work_dir)
        model_dir = self.make_dir(os.path.join(args.work_dir, 'world_model'))
        buffer_dir = self.make_dir(os.path.join(args.work_dir, 'world_buffer'))

        with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        replay_buffer = World_Buffer(
            obs_shape=env.observation_space.shape,
            action_shape=[1], # discrete, 1 dimension!
            capacity= args.replay_buffer_capacity,
            batch_size= args.batch_size,
            device=device
        )
        self.world_model = World_Model(obs_shape=env.observation_space.shape,
            action_shape=[1], # discrete, 1 dimension!
            state_space_dim=env.state_dimension,
            device=device,
            transition_model_type = args.transition_model_type,
            env=env)

        try:
            load_step = 360000
            self.world_model.load(model_dir, load_step)
            print("[World_Model] : Load learned model successful, step=",load_step)

        except:
            load_step = 0
            print("[World_Model] : No learned model, Creat new model")


        # L = Logger(args.work_dir, use_tb=args.save_tb)

        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        for step in range(args.num_train_steps):
            if done:
                if step > 0:
                    # L.log('train/duration', time.time() - start_time, step)
                    start_time = time.time()
                    # L.dump(step)

                # L.log('train/episode_reward', episode_reward, step)

                obs = env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                reward = 0
                # L.log('train/episode', episode, step)
    
            
            # evaluate agent periodically
            if step % args.eval_freq == 0:
                # L.log('eval/episode', episode, step)
                if args.save_model:
                    print("[World_Model] : Saved Model! Step:",step + load_step)
                    self.world_model.save(model_dir, step + load_step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)
                    print("[World_Model] : Saved Buffer!")

            # run training update
            if step >= args.init_steps:
                num_updates = args.init_steps if step == args.init_steps else 1
                for _ in range(num_updates):
                    self.world_model.update(replay_buffer, step) # Updated Transition and Reward Module


            obs = np.array(obs)
            curr_reward = reward

            # Rule-based Planner
            self.dynamic_map.update_map_from_obs(obs, env)
            rule_trajectory = self.trajectory_planner.trajectory_update(self.dynamic_map)
            action = np.array(random.randint(0,6)) #FIXME:Action space
            print("Action",action)
            # Control
            trajectory = self.trajectory_planner.trajectory_update_CP(action, rule_trajectory)
            for i in range(args.decision_count):
                control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                output_action = [control_action.acc, control_action.steering]
                new_obs, reward, done, info = env.step(output_action)
                if done:
                    break
                self.dynamic_map.update_map_from_obs(new_obs, env)

            print("Predicted Reward:",self.world_model.get_reward_prediction(obs, action))
            print("Actual Reward:",reward)
            print("Predicted State:",self.world_model.get_trans_prediction(obs, action)[0])
            print("Actual State:",(new_obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low))
            episode_reward += reward
            normal_new_obs = (new_obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            normal_obs = (obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            replay_buffer.add(normal_obs, action, curr_reward, reward, normal_new_obs, done)
            # np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)

            obs = new_obs
            episode_step += 1
        
    def reward_predict(self, state, action):
        # The reward can also done by directly calculate ego v
        return self.world_model.get_reward_prediction(state, action)

    def transition_predict(self, state, action):
        print("Simulation_step:",self.simulation_step)
        # Env dynamics (Directly Read From Memory)
        if self.simulation_step < len(self.corner_buffer._storage):
            experience = self.corner_buffer._storage[self.simulation_step]
            obs_e, action_e, rew, new_obs, done, masks, train_times = experience
            predict_state = new_obs.copy()
        else:
            experience_1 = self.corner_buffer._storage[-2]
            experience_2 = self.corner_buffer._storage[-1]
            obs_e, action_e, rew, new_obs, done, masks, train_times = experience_1
            obs_e_2, action_e_2, rew_2, new_obs_2, done_2, masks_2, train_times_2 = experience_2
            x = new_obs_2[5]
            y = new_obs_2[6]
            vx = (new_obs_2[7] - new_obs[7])/0.2 #FIXME:Simulation Time Step
            vy = (new_obs_2[8] - new_obs[8])/0.2
            predict_state = new_obs_2.copy()
            predict_state[5] = x + vx * 0.2 * (self.simulation_step - len(self.corner_buffer._storage) + 1)
            predict_state[6] = y + vy * 0.2 * (self.simulation_step - len(self.corner_buffer._storage) + 1)
            predict_state[7] = vx
            predict_state[8] = vy

        # Ego dynamics (Vehicle Dynamics)
        temp_state = state.copy()
        temp_state[5] = 0
        temp_state[6] = 0
        temp_state[7] = 0
        temp_state[8] = 0
        temp_state[9] = 0

        next_state = self.world_model.get_trans_prediction(temp_state, action)[0].cpu().numpy()
        next_state = next_state[0] * (self.env.observation_space.high - self.env.observation_space.low) + self.env.observation_space.low

        predict_state[0] = next_state[0]
        predict_state[1] = next_state[1]
        predict_state[2] = next_state[2]
        predict_state[3] = next_state[3]
        predict_state[4] = next_state[4]

        return predict_state
       
    def reset(self):    
        if not self.world_model:
            self.load_world_model()

        self.current_s = self.s_0
        self.simulation_step = 0
        return self.s_0

    def step(self, action):

        # Step reward
        reward = self.reward_predict(self.current_s, action)

        # State
        self.current_s = self.transition_predict(self.current_s, action)
        
        
        p1 = np.array([self.current_s[0] , self.current_s[1]]) # ego
        p2 = np.array([self.current_s[5] , self.current_s[6]]) # the env vehicle
        p3 = p2 - p1
        p4 = math.hypot(p3[0],p3[1])


        # If finish
        done = False
        print("p4",p4)
        if p4 < 5:# Collision check
            print("Imagine: Collision")
            done = True
        
        elif (self.current_s[0] + 73) ** 2 + (self.current_s[1] - 167) ** 2 < 100:
            print("Imagine: Pass")
            done = True

        elif self.current_s[0] > -20 or self.current_s[0] < -82 or self.current_s[1] > 200 or self.current_s[1] < 160:
            print("Imagine: Out of Area")
            done = True

        self.simulation_step += 1
        return self.current_s, reward, done, None

    def set_seed_everywhere(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def make_dir(self, dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            pass
        return dir_path


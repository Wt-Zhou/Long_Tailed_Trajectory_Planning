import glob
import os
import sys
try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass
try:
	sys.path.append(glob.glob("/home/icv/.local/lib/python3.6/site-packages/")[0])
except IndexError:
	pass

import carla
import time
import numpy as np
import math
import random
import gym
import matplotlib.pyplot as plt

from tqdm import tqdm
from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap

# from Agent.zzz.CP import CP, Imagine_Model
EPISODES=2642

if __name__ == '__main__':

    # Create environment
    
    env = CarEnv_02_Intersection_fixed()

    # Create Agent
    trajectory_planner = JunctionTrajectoryPlanner()
    controller = Controller()
    dynamic_map = DynamicMap()
    target_speed = 30/3.6 

    pass_time = 0
    task_time = 0
    
    fig, ax = plt.subplots()

    # Loop over episodes
    for episode in tqdm(range(1, EPISODES + 1), unit='episodes'):
        
        print('Restarting episode')

        # Reset environment and get initial state
        obs = env.reset()
        episode_reward = 0
        done = False
        decision_count = 0
        
        # Loop over steps
        while True:
            obs = np.array(obs)
            dynamic_map.update_map_from_obs(obs, env)
            rule_trajectory, action = trajectory_planner.trajectory_update(dynamic_map)

            # # action = random.randint(0,6)
            # print("action",action)
            rule_trajectory = trajectory_planner.trajectory_update_CP(action, rule_trajectory)
            # Control
            
            for i in range(1):
                control_action =  controller.get_control(dynamic_map,  rule_trajectory.trajectory, rule_trajectory.desired_speed)
                action = [control_action.acc, control_action.steering]
                new_obs, reward, done, _ = env.step(action)   
                dynamic_map.update_map_from_obs(new_obs, env)
                if done:
                    break
                # Set current step for next loop iteration
            obs = new_obs
            episode_reward += reward  
            
            # Draw Plot
            # ax.cla() 
            
            # # Real Time
            # angle = -obs.tolist()[4]/math.pi*180 - 90
            # rect = plt.Rectangle((obs.tolist()[0],-obs.tolist()[1]),2.2,5,angle=angle, facecolor="red")
            # ax.add_patch(rect)
            # angle2 = -obs.tolist()[9]/math.pi*180 - 90
            # rect = plt.Rectangle((obs.tolist()[5],-obs.tolist()[6]),2.2,5,angle=angle2, facecolor="blue")
            # ax.add_patch(rect)
            
            # # Predict
            
            # ax.axis([190,280,-120,-30])
            # plt.pause(0.0001)          

            if done:
                trajectory_planner.clear_buff(clean_csp=False)
                task_time += 1
                if reward > 0:
                    pass_time += 1
                break

        print("Episode Reward:",episode_reward)
        print("Success Rate:",pass_time/task_time)
        


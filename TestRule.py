import glob
import os
import os.path as osp
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

import math
import random
import time

import carla
import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from results import Results
# from Agent.zzz.JunctionTrajectoryPlanner_simple_predict import JunctionTrajectoryPlanner
from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed

# from Agent.zzz.CP import CP, Imagine_Model
EPISODES=62

if __name__ == '__main__':

    # Create environment
    
    env = CarEnv_02_Intersection_fixed()

    # Create Agent
    trajectory_planner = JunctionTrajectoryPlanner()
    controller = Controller()
    dynamic_map = DynamicMap()
    target_speed = 30/3.6 
    
    results = Results(trajectory_planner.obs_prediction.gnn_predictin_model.history_frame)

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
        
        his_obs_frames = []
        
        # Loop over steps
        while True:
            obs = np.array(obs)
            dynamic_map.update_map_from_list_obs(obs, env)
            rule_trajectory, rule_action = trajectory_planner.trajectory_update(dynamic_map)
            rule_trajectory = trajectory_planner.trajectory_update_CP(rule_action, rule_trajectory)
            # Control
            control_action =  controller.get_control(dynamic_map,  rule_trajectory.trajectory, rule_trajectory.desired_speed)
            action = [control_action.acc, control_action.steering]
            new_obs, reward, done, collision_signal = env.step(action)   
            
            his_obs_frames.append(obs)
            if len(his_obs_frames) > trajectory_planner.obs_prediction.gnn_predictin_model.history_frame-1:
                results.add_data_for_real_time_metrics(his_obs_frames, trajectory_planner.all_trajectory[int(rule_action - 1)][0], collision_signal)

                his_obs_frames.pop(0)
                
            obs = new_obs
            episode_reward += reward  
            
            # draw debug signal
            for trajectory in trajectory_planner.all_trajectory:
                for i in range(len(trajectory[0].x)):
                    env.debug.draw_point(carla.Location(x=trajectory[0].x[i],y=trajectory[0].y[i],z=env.ego_vehicle.get_location().z+1),
                                         size=0.05, color=carla.Color(r=0,g=0,b=255), life_time=0.2)
            
            for ref_point in env.ref_path.central_path:
                env.debug.draw_point(carla.Location(x=ref_point.position.x,y=ref_point.position.y,z=env.ego_vehicle.get_location().z+1),
                                        size=0.03, color=carla.Color(r=0,g=0,b=0), life_time=0.2)
            

            for predict_trajectory in trajectory_planner.obs_prediction.predict_paths:
                # print("predict_trajectory",predict_trajectory.x)
                for i in range(len(predict_trajectory.x)):
                    env.debug.draw_point(carla.Location(x=predict_trajectory.x[i],y=predict_trajectory.y[i],z=env.ego_vehicle.get_location().z+0.5),
                                         size=0.05, color=carla.Color(r=255,g=0,b=0), life_time=0.2)  
                    
            
            if done:
                his_obs_frames = []
                trajectory_planner.clear_buff(clean_csp=False)
                task_time += 1
                if reward > 0:
                    pass_time += 1
                break
            
    # Calculate Experiment Results
    results.calculate_all_state_visited_time()     

    
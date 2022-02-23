import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(profile='short')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tqdm import tqdm

from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.prediction.coordinates import Coordinates
from Agent.zzz.prediction.gnn_prediction import GNN_Prediction_Model
from Agent.zzz.prediction.KinematicBicycleModel.kinematic_model import \
    KinematicBicycleModel
from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed


class Prediction_Model_Training():
    
    def __init__(self):
        self.data = []
        self.one_trajectory = []
        
        self.ensemble_models = []
        self.ensemble_optimizer = []
        self.train_step = 0


        # Parameters of Prediction Model
        self.heads_num = 1
        self.history_frame = 3
        self.future_frame = 5
        self.obs_scale = 5
        self.action_scale = 5
        self.agent_dimension = 5 # x,y,vx,vy,yaw
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_tensor_type(torch.DoubleTensor)
        
        for i in range(self.heads_num):
            predition_model = GNN_Prediction_Model(self.history_frame * 5, self.future_frame * 2, self.obs_scale).to(self.device)
            predition_model.apply(self.weight_init)
            self.ensemble_models.append(predition_model)
            self.ensemble_optimizer.append(torch.optim.Adam(predition_model.parameters(), lr = 0.0005))
            predition_model.train()
            
        # Vehicle Model
        self.wheelbase = 2.96
        self.max_steer = np.deg2rad(60)
        self.dt = 0.1
        self.c_r = 0.01
        self.c_a = 2.0
        self.kbm = KinematicBicycleModel(self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)

    def add_data(self, obs, done):
        trajectory_length = self.history_frame + self.future_frame
        if not done:
            self.one_trajectory.append(obs)
            if len(self.one_trajectory) > trajectory_length:
                self.data.append(self.one_trajectory[0:trajectory_length])
                self.one_trajectory.pop(0)
        else:
            self.one_trajectory = []
                
    def learn(self, env, load_step, train_steps=100000):
        # Create Agent
        trajectory_planner = JunctionTrajectoryPlanner()
        controller = Controller()
        dynamic_map = DynamicMap()
        target_speed = 30/3.6 

        pass_time = 0
        task_time = 0
        
        # Load_model
        self.load_prediction_model(load_step)
        
        # Collect Data from CARLA and train model
        for episode in tqdm(range(load_step, train_steps + load_step), unit='episodes'):
            
            # Reset environment and get initial state
            obs = env.reset()
            episode_reward = 0
            done = False
            decision_count = 0
            
            # Loop over steps
            while True:
                obs = np.array(obs)
                dynamic_map.update_map_from_list_obs(obs, env)
                rule_trajectory, action = trajectory_planner.trajectory_update(dynamic_map)
                rule_trajectory = trajectory_planner.trajectory_update_CP(action, rule_trajectory)
                # Control
                
                control_action =  controller.get_control(dynamic_map,  rule_trajectory.trajectory, rule_trajectory.desired_speed)
                action = [control_action.acc, control_action.steering]
                new_obs, reward, done, _ = env.step(action)  
                
                self.add_data(new_obs, done)
                self.train_model()
                 
                obs = new_obs
                episode_reward += reward      

                if done:
                    trajectory_planner.clear_buff(clean_csp=False)
                    task_time += 1
                    if reward > 0:
                        pass_time += 1
                    break

    def train_model(self):
        if len(self.data) > 0:
            # take data
            one_trajectory = self.data[0]
            vehicle_num = len(one_trajectory[0])
            
            # input: transfer to ego
            history_obs = one_trajectory[0:self.history_frame]
            # print("one_trajectory",one_trajectory)

            history_data = []
            for j in range(0, vehicle_num):
                vehicle_state = []
                ego_vehicle_coordiate = Coordinates(history_obs[0][0][0],history_obs[0][0][1],history_obs[0][0][4]) # Use the first frame ego vehicle as center
                for obs in history_obs: 
                    x_t, y_t, vx_t, vy_t, yaw_t = ego_vehicle_coordiate.transfer_coordinate(obs[j][0],obs[j][1],
                                                                                obs[j][2],obs[j][3],obs[j][4])
                    scale_state = [x / self.obs_scale for x in [x_t, y_t, vx_t, vy_t, yaw_t]]
                    vehicle_state.extend(scale_state)
                history_data.append(vehicle_state)
            history_data = torch.tensor(history_data).to(self.device).unsqueeze(0)
            # print("history_data",history_data)
            # target: output action
            target_action = self.get_target_action_from_obs(one_trajectory)
            target_action = torch.tensor(target_action).to(self.device).unsqueeze(0)

            for i in range(self.heads_num):
                # compute loss
                predict_action = self.ensemble_models[i](history_data)
                # print("target_action",target_action)
                # print("predict_action",predict_action)
                loss = F.mse_loss(target_action, predict_action)
                print("------------loss",loss)
                
                # train
                self.ensemble_optimizer[i].zero_grad()
                loss.backward()
                self.ensemble_optimizer[i].step()
            
            self.data.pop(0)
            if self.train_step % 10000 == 0:
                self.save_prediction_model(self.train_step)
            self.train_step += 1

        return None
    
    def get_target_action_from_obs(self, one_trajectory):
        action_list = []
        vehicle_num = len(one_trajectory[0])
        action_list = []
        
        for j in range(0, vehicle_num):
            vehicle_action = []
            for i in range(0, self.future_frame):
                x1 = one_trajectory[self.history_frame-1+i][j][0]
                y1 = one_trajectory[self.history_frame-1+i][j][1]
                yaw1 = one_trajectory[self.history_frame-1+i][j][4]
                v1 = math.sqrt(one_trajectory[self.history_frame-1+i][j][2] ** 2 + one_trajectory[self.history_frame-1+i][j][3] ** 2)
                x2 = one_trajectory[self.history_frame+i][j][0]
                y2 = one_trajectory[self.history_frame+i][j][1]
                yaw2 = one_trajectory[self.history_frame+i][j][4]
                v2 = math.sqrt(one_trajectory[self.history_frame+i][j][2] ** 2 + one_trajectory[self.history_frame+i][j][3] ** 2)
                throttle, delta = self.kbm.calculate_a_from_data(x1, y1, yaw1, v1, x2, y2, yaw2, v2)
                
                vehicle_action.append(throttle/self.action_scale)
                vehicle_action.append(delta/self.action_scale)
            action_list.append(vehicle_action)
      
        return action_list
             
    def predict_future_paths(self, obs, done):
        paths = []
        
        
        return paths         
              
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式 
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def save_prediction_model(self, step):
        for i in range(self.heads_num):
            torch.save(
                self.ensemble_models[i].state_dict(),
                'save_model/ensemble_models_%s_%s.pt' % (step, i)
            )
            
    def load_prediction_model(self, load_step):
        try:
            for i in range(self.heads_num):

                self.ensemble_models[i].load_state_dict(
                torch.load('save_model/transition_model_%s_%s.pt' % (load_step, i))
                )
            print("[Prediction_Model] : Load learned model successful, step=",load_step)
        except:
            load_step = 0
            print("[Prediction_Model] : No learned model, Creat new model")
        return load_step
    


if __name__ == '__main__':

    env = CarEnv_02_Intersection_fixed()

    training = Prediction_Model_Training()
    training.learn(env, load_step=0)


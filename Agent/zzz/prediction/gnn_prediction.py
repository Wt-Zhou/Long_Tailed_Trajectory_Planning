import math

import numpy as np
import torch
import torch.nn as nn
from Agent.zzz.prediction.agent_model.KinematicBicycleModel.kinematic_model import (
    KinematicBicycleModel, KinematicBicycleModel_Pytorch)
from Agent.zzz.prediction.predmlp import TrajPredMLP
from Agent.zzz.prediction.selfatten import SelfAttentionLayer


class GNN_Prediction_Model(nn.Module):
    """
    Self_attention GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, out_channels, obs_scale, global_graph_width=128, traj_pred_mlp_width=128):
        super(GNN_Prediction_Model, self).__init__()
        self.polyline_vec_shape = in_channels
        self.self_atten_layer = SelfAttentionLayer(
            self.polyline_vec_shape, global_graph_width)
       
        self.traj_pred_mlp = TrajPredMLP(
            global_graph_width, out_channels, traj_pred_mlp_width)
        
        # Vehicle Model
        self.wheelbase = 2.96
        self.max_steer = np.deg2rad(60)
        self.dt = 0.1
        self.c_r = 0.0
        self.c_a = 0.0
        self.vehicle_model_torch = KinematicBicycleModel_Pytorch(self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)

        self.obs_scale = obs_scale

    def forward(self, obs):
        out = self.self_atten_layer(obs)
        pred_action = self.traj_pred_mlp(out)
        return pred_action
        
    def forward_torch_vehicle_model(self, obs, pred_action):
        pred_state = []
        for i in range(len(pred_action[0])):         
            x = torch.mul(obs[i][0], self.obs_scale)
            y = torch.mul(obs[i][1], self.obs_scale)
            yaw = torch.mul(obs[i][4], self.obs_scale)
            v = torch.tensor(math.sqrt(torch.mul(obs[i][2], self.obs_scale) ** 2 + torch.mul(obs[i][3], self.obs_scale) ** 2))
            x, y, yaw, v, _, _ = self.vehicle_model_torch.kinematic_model(x, y, yaw, v, pred_action[0][i][0], pred_action[0][i][1])
            tensor_list = [torch.div(x, self.obs_scale), torch.div(y, self.obs_scale), torch.div(torch.mul(v, torch.cos(yaw)), self.obs_scale),
                           torch.div(torch.mul(v, torch.sin(yaw)), self.obs_scale), torch.div(yaw, self.obs_scale)]
            next_vehicle_state = torch.stack(tensor_list)
            # print("next_vehicle_state",next_vehicle_state)

            pred_state.append(next_vehicle_state)
            
        print("pred_state",pred_state)
        pred_state = torch.stack(pred_state)
        return pred_state
    
    
class KinematicBicycleModel_Pytorch():

    def __init__(self, wheelbase=1.0, max_steer=0.7, dt=0.1, c_r=0.0, c_a=0.0):
        """
        2D Kinematic Bicycle Model

        At initialisation
        :param wheelbase:       (float) vehicle's wheelbase [m]
        :param max_steer:       (float) vehicle's steering limits [rad]
        :param dt:              (float) discrete time period [s]
        :param c_r:             (float) vehicle's coefficient of resistance 
        :param c_a:             (float) vehicle's aerodynamic coefficient
    
        At every time step  
        :param x:               (float) vehicle's x-coordinate [m]
        :param y:               (float) vehicle's y-coordinate [m]
        :param yaw:             (float) vehicle's heading [rad]
        :param velocity:        (float) vehicle's velocity in the x-axis [m/s]
        :param throttle:        (float) vehicle's accleration [m/s^2]
        :param delta:           (float) vehicle's steering angle [rad]
    
        :return x:              (float) vehicle's x-coordinate [m]
        :return y:              (float) vehicle's y-coordinate [m]
        :return yaw:            (float) vehicle's heading [rad]
        :return velocity:       (float) vehicle's velocity in the x-axis [m/s]
        :return delta:          (float) vehicle's steering angle [rad]
        :return omega:          (float) vehicle's angular velocity [rad/s]
        """

        self.dt = dt
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.c_r = c_r
        self.c_a = c_a
        self.dt_discre = 100

    def kinematic_model(self, x, y, yaw, velocity, throttle, delta):
        # Compute the local velocity in the x-axis
        for i in range(self.dt_discre):
            throttle = torch.mul(throttle, 1) # throttle * 10, steer (0-1)
            delta = torch.mul(delta, 1) # throttle * 10, steer (0-1)
            ca = torch.mul(velocity, self.c_a)
            temp = torch.add(ca, self.c_r)
            f_load = torch.mul(velocity, temp) # 
                                                                                                                
            dv = torch.mul(torch.sub(throttle, f_load), self.dt/self.dt_discre)
            velocity = torch.add(velocity, dv)  

            # Compute the state change rate
            x_dot = torch.mul(velocity, torch.cos(yaw))
            y_dot = torch.mul(velocity, torch.sin(yaw))
            omega = torch.mul(velocity, torch.tan(delta))
            omega = torch.mul(omega, 1/self.wheelbase)
            
            # Compute the final state using the discrete time model
            x = torch.add(x, torch.mul(x_dot, self.dt/self.dt_discre))
            y = torch.add(y, torch.mul(y_dot, self.dt/self.dt_discre))
            yaw = torch.add(yaw, torch.mul(omega, self.dt/self.dt_discre))
            yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

        return x, y, yaw, velocity, delta, omega
    
    def calculate_a_from_data(self, x, y, yaw, velocity, x2, y2, yaw2, velocity2):

        ca = torch.mul(velocity, self.c_a)
        temp = torch.add(ca, self.c_r)
        f_load = torch.mul(velocity, temp)         

        dv = torch.sub(velocity2, velocity)
        dv_dt = torch.div(dv, self.dt)
        throttle = torch.add(dv_dt, f_load)
        
        if velocity == 0:
            delta = torch.zeros_like(x)
        else:
            dyaw = torch.sub(yaw2, yaw)
            delta = torch.div(dyaw, self.dt/self.wheelbase)
            delta = torch.div(delta, velocity)
            delta = torch.atan(delta)
       
        return throttle, delta


        

import torch
import torch.nn as nn
import math
import numpy as np
from Agent.world_model.self_attention.predmlp import TrajPredMLP
from Agent.world_model.self_attention.selfatten import SelfAttentionLayer
from Agent.world_model.agent_model.KinematicBicycleModel.kinematic_model import KinematicBicycleModel, KinematicBicycleModel_Pytorch


class Interaction_Transition_Model(nn.Module):
    """
    Self_attention GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, out_channels, obs_scale, global_graph_width=128, traj_pred_mlp_width=128):
        super(Interaction_Transition_Model, self).__init__()
        self.polyline_vec_shape = in_channels
        self.self_atten_layer = SelfAttentionLayer(
            self.polyline_vec_shape, global_graph_width)

        self.traj_pred_mlp = TrajPredMLP(
            global_graph_width+2, self.polyline_vec_shape, traj_pred_mlp_width)
        
        self.self_atten_layer2 = SelfAttentionLayer(
            self.polyline_vec_shape, global_graph_width)
        
        self.traj_pred_mlp2 = TrajPredMLP(
            global_graph_width+2, out_channels, traj_pred_mlp_width)
        
        # Vehicle Model
        self.wheelbase = 2.96
        self.max_steer = np.deg2rad(60)
        self.dt = 0.1
        self.c_r = 0.0
        self.c_a = 0.0
        self.vehicle_model_torch = KinematicBicycleModel_Pytorch(self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)

        self.obs_scale = obs_scale

    def forward(self, obs, action_torch):
        """
        args: 
            data (Data): [x, y, cluster, edge_index, valid_len]

        """

        out = self.self_atten_layer(obs.unsqueeze(0))
        # action_torch = torch.cat((action_torch, action_torch, action_torch, action_torch), dim=0).unsqueeze(0)
        action_torch = torch.cat((action_torch, action_torch), dim=0).unsqueeze(0)
        # concat out and control_action
        out_with_action = torch.cat((out, action_torch),dim=2)
        pred_action1 = self.traj_pred_mlp(out_with_action)
        
        out = self.self_atten_layer2(pred_action1)
        pred_action = self.traj_pred_mlp2(out_with_action)

        # print("pred_action",pred_action)
        # pred_state = self.forward_torch_vehicle_model(obs, pred_action) # it is quite slow to use that
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

        
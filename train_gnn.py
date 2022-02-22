import gym
import numpy as np
import os
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed
from Agent.zzz.prediction.gnn_prediction import DataStore_Training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = CarEnv_02_Intersection_fixed()

data_store = DataStore_Training()
data_store.collect_carla_data(env)









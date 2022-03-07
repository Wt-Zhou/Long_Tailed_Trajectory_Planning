import math
import os

import numpy as np
import torch
import torch.nn as nn

from Agent.zzz.prediction.gnn_prediction import Prediction_Model_Training

torch.set_printoptions(profile='short')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Test_Scenarios.TestScenario_Town02 import CarEnv_02_Intersection_fixed

if __name__ == '__main__':

    env = CarEnv_02_Intersection_fixed()

    training = Prediction_Model_Training()
    training.learn(env, load_step=50000, train_episode=61)


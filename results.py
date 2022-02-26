import math
import os
import os.path as osp

import numpy as np
import torch
from rtree import index as rindex

from Agent.zzz.prediction.coordinates import Coordinates
from Agent.zzz.prediction.KinematicBicycleModel.kinematic_model import \
    KinematicBicycleModel


class Results():
    def __init__(self, create_new_train_file=True):
        
        
        if create_new_train_file:
            if osp.exists("results/state_index.dat"):
                os.remove("results/state_index.dat")
                os.remove("results/state_index.idx")
            if osp.exists("results/visited_state.txt"):
                os.remove("results/visited_state.txt")

            self.visited_state_counter = 0
            self.visited_state_effiency_d = []
            self.visited_state_effiency_v = []
            self.visited_state_safety = []
            
        else:
            self.visited_state_effiency_d = np.loadtxt("results/effiency_d.txt").tolist()
            self.visited_state_effiency_v = np.loadtxt("results/effiency_v.txt").tolist()
            self.visited_state_safety = np.loadtxt("results/safety.txt").tolist()
            self.visited_state_counter = len(self.visited_state_effiency_d)

        self.visited_state_outfile = open("results/visited_state.txt", "a")
        self.visited_state_format = " ".join(("%f",)*(10))+"\n"

        visited_state_tree_prop = rindex.Property()
        visited_state_tree_prop.dimension = 10 # 4 vehicles
        
        self.all_state_list = []
        self.visited_state_dist = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
        self.visited_state_tree = rindex.Index('results/state_index',properties=visited_state_tree_prop)

    def calculate_visited_times(self, state):
        
        return sum(1 for _ in self.visited_state_tree.intersection(state.tolist()))

    def add_data(self, obs, trajectory, collision):    
        
        obs = np.array(obs).flatten().tolist()
        
        self.all_state_list.append(obs)
        self.visited_state_tree.insert(self.visited_state_counter,
            tuple((obs-self.visited_state_dist).tolist()[0]+(obs+self.visited_state_dist).tolist()[0]))
        self.visited_state_outfile.write(self.visited_state_format % tuple(obs))
        self.visited_state_counter += 1
        
        # safety metrics
        if collision:
            self.visited_state_safety.append(0)
        else:
            self.visited_state_safety.append(1)
            
        # effiency metrics
        trajectory_d = []
        trajectory_v = []
        for i in range(len(trajectory.x)):
            trajectory_d.append(trajectory.d[i])
            trajectory_v.append(trajectory.s_d[i])
            
        self.visited_state_effiency_d.append(np.mean(trajectory_d))
        self.visited_state_effiency_v.append(np.mean(trajectory_v))
        
        return None
    
    def calculate_predition_results(self, dataset, predition_model):
        for one_trajectory in dataset:
            vehicle_num = len(one_trajectory[0])
            
            # get model prediction
            history_obs = one_trajectory[0:history_frame]

            for i, obs in enumerate(history_obs):
                for j in range(len(predition_model)):
                    if i < len(history_obs):
                        predition_model[j].predict_future_paths(obs)
                    else:
                        paths_of_all_models = predition_model[j].predict_future_paths(obs)
            
            for predict_path in paths_of_all_models:
                
                # ade - over a whole trajectory 
                de_list = []
                
                for k in range(len(predict_path.x)):
                    dx = predict_path.x[k] - one_trajectory[k][predict_path.c][0]
                    dy = predict_path.y[k] - one_trajectory[k][predict_path.c][1]
                    de_list.append(math.sqrt(dx**2 + dy**2))
                    
                ade = np.mean(de_list)

                # fde - final point of trajectory 
                dx = predict_path.x[-1] - one_trajectory[-1][predict_path.c][0]
                dy = predict_path.y[-1] - one_trajectory[-1][predict_path.c][1]
                
                fde = math.sqrt(dx**2 + dy**2)
                
                print("ade.fde",ade,fde)
                

    def calculate_all_state_visited_time(self):
        self.mark_list = np.zeros(self.visited_state_counter)
        print("debug result",self.visited_state_counter,len(self.visited_state_effiency_d),len(self.visited_state_effiency_v),len(self.visited_state_safety))
        for i in range(self.visited_state_counter):
            if self.mark_list[i] == 0:
                state = self.all_state_list[i]
                
                visited_times = sum(1 for _ in self.visited_state_tree.intersection(state))
                # mark similar state
                state_effiency_v = 0
                state_effiency_d = 0
                state_safety = 0
                for n in self.visited_state_tree.intersection(state):
                    state_effiency_d += math.fabs(self.visited_state_effiency_d[n])
                    state_effiency_v += self.visited_state_effiency_v[n]
                    state_safety += self.visited_state_safety[n]
                    
                    self.mark_list[n] = 1
                    
                state_effiency_d /= visited_times
                state_effiency_v /= visited_times
                state_safety /= visited_times
                
                print("results",state[0], visited_times, state_effiency_d, state_effiency_v, state_safety)
                     
                # write to txt but not save in list
                # with open(self.log_dir, 'a') as fw:
                #     fw.write(str(state)) 
                #     fw.write(", ")
                #     fw.write(str(visited_times)) 
                #     fw.write(", ")
                #     fw.write(str(state_safety)) 
                #     fw.write(", ")
                #     fw.write(str(state_effiency_d)) 
                #     fw.write(", ")
                #     fw.write(str(state_effiency_v)) 
                #     fw.write("\n")
                #     fw.close()               
                

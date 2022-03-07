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
    def __init__(self, history_frame, create_new_train_file=True):
        
        
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
            self.prediction_ade = []
            self.prediction_fde = []
            
        else:
            self.visited_state_effiency_d = np.loadtxt("results/effiency_d.txt").tolist()
            self.visited_state_effiency_v = np.loadtxt("results/effiency_v.txt").tolist()
            self.visited_state_safety = np.loadtxt("results/safety.txt").tolist()
            self.visited_state_counter = len(self.visited_state_effiency_d)

        self.visited_state_outfile = open("results/visited_state.txt", "a")
        self.visited_state_format = " ".join(("%f",)*(history_frame * 20))+"\n"

        visited_state_tree_prop = rindex.Property()
        visited_state_tree_prop.dimension = history_frame * 20 # 4 vehicles
        
        self.all_state_list = []
        # self.visited_state_dist = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
        self.visited_state_dist = np.full(shape=history_frame * 20,
                                    fill_value=1)
        self.visited_state_tree = rindex.Index('results/state_index',properties=visited_state_tree_prop)

    def calculate_visited_times(self, state):
        
        return sum(1 for _ in self.visited_state_tree.intersection(state.tolist()))

    def add_data_for_real_time_metrics(self, his_obs_frames, trajectory, collision):    
        
        obs = his_obs_frames
        obs = np.array(obs).flatten().tolist()

        self.all_state_list.append(obs)
      
        self.visited_state_tree.insert(self.visited_state_counter,
            tuple((obs-self.visited_state_dist).tolist()+(obs+self.visited_state_dist).tolist()))
        # self.visited_state_tree.insert(self.visited_state_counter,
        #     tuple((obs-self.visited_state_dist).tolist()[0]+(obs+self.visited_state_dist).tolist()[0]))
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
            trajectory_d.append(math.fabs(trajectory.d[i]))
            trajectory_v.append(trajectory.s_d[i])
            
        self.visited_state_effiency_d.append(np.mean(trajectory_d))
        self.visited_state_effiency_v.append(np.mean(trajectory_v))
        
        return None
    
    def add_data_for_dataset_prediciton_metrics(self, his_obs_frames, fde, ade):    
        
        obs = his_obs_frames
        obs = np.array(obs).flatten().tolist()
        
        self.all_state_list.append(obs)
        self.visited_state_tree.insert(self.visited_state_counter,
            tuple((obs-self.visited_state_dist).tolist()+(obs+self.visited_state_dist).tolist()))
        self.visited_state_outfile.write(self.visited_state_format % tuple(obs))
        self.visited_state_counter += 1
        
        self.prediction_ade.append(ade) 
        self.prediction_fde.append(fde) 
        
        return None
    
    def calculate_predition_results(self, dataset, predict_future_paths, history_frame):
        # run over the dataset
        for one_trajectory in dataset:
            vehicle_num = 0
            for i in range(len(one_trajectory[0])):
                if one_trajectory[0][i][0] != -999: # use -999 as signal, very unstable
                    vehicle_num += 1
            # get model prediction
            history_obs = one_trajectory[0:history_frame]
            for i, obs in enumerate(history_obs):
                if i < len(history_obs)-1:
                    predict_future_paths(obs, done=False)
                else:
                    paths_of_all_models = predict_future_paths(obs, done=False)
            
            ade = 0
            fde = 0
            for c in range(1,vehicle_num):
                min_ade_head = 99999
                min_fde_head = 99999
                for predict_path in paths_of_all_models:
                    if predict_path.c == c:
                        # ade - over a whole trajectory 
                        de_list = []
                        for k in range(len(predict_path.x)):
                            dx = predict_path.x[k] - one_trajectory[k+history_frame][predict_path.c][0]
                            dy = predict_path.y[k] - one_trajectory[k+history_frame][predict_path.c][1]
                            de_list.append(math.sqrt(dx**2 + dy**2))
                            # print("dx",dx,predict_path.x[k],one_trajectory[k+history_frame][predict_path.c][0])
                            # print("dy",dy,predict_path.y[k],one_trajectory[k+history_frame][predict_path.c][1])
                        if np.mean(de_list) < min_ade_head:
                            min_ade_head = np.mean(de_list)
                            
                        # print("len predict path",len(predict_path.x))
                        # print("len predict path",len(predict_path.y))
                        # print("len one_trajectory path",len(one_trajectory))
                        # print("min_ade_head",np.mean(de_list))
                        # print("min_ade_head",min_ade_head)
                        # fde - final point of trajectory 
                        d_x = predict_path.x[-1] - one_trajectory[-1][predict_path.c][0]
                        d_y = predict_path.y[-1] - one_trajectory[-1][predict_path.c][1]
                        # print("d_x",d_x,predict_path.x[-1],one_trajectory[-1][predict_path.c][0])
                        # print("d_y",d_y,predict_path.y[-1],one_trajectory[-1][predict_path.c][1])

                        if math.sqrt(d_x**2 + d_y**2) < min_fde_head:
                            min_fde_head = math.sqrt(d_x**2 + d_y**2)
                #         print("min_fde_head",math.sqrt(d_x**2 + d_y**2))
                #         print("min_fde_head",min_fde_head)
                # print("min_ade_head",min_ade_head)
                # print("min_fde_head",min_fde_head)

                ade += min_ade_head # divide by heads num
                fde += min_fde_head
                        
            ade /= vehicle_num
            fde /= vehicle_num
                        
            self.add_data_for_dataset_prediciton_metrics(history_obs, fde, ade)
                        
        # count the results from rtree
        self.mark_list = np.zeros(self.visited_state_counter)
        for i in range(self.visited_state_counter):
            if self.mark_list[i] == 0:
                state = self.all_state_list[i]

                visited_times = sum(1 for _ in self.visited_state_tree.intersection(state))
                # mark similar state
                state_ade = 0
                state_fde = 0
                for n in self.visited_state_tree.intersection(state):
                    state_ade += self.prediction_ade[n]
                    state_fde += self.prediction_fde[n]

                    self.mark_list[n] = 1
    
                state_ade /= visited_times
                state_fde /= visited_times
                
                print("results", visited_times, state_ade, state_fde)
                # write to txt
                with open("prediction_results.txt", 'a') as fw:
                    fw.write(str(state)) 
                    fw.write(", ")
                    fw.write(str(visited_times)) 
                    fw.write(", ")
                    fw.write(str(state_ade)) 
                    fw.write(", ")
                    fw.write(str(state_fde)) 
                    fw.write("\n")
                    fw.close()           
              
    def calculate_all_state_visited_time(self):
        self.mark_list = np.zeros(self.visited_state_counter)
        for i in range(self.visited_state_counter):
            if self.mark_list[i] == 0:
                state = self.all_state_list[i]
                # mark similar state
                state_effiency_v = 0
                state_effiency_d = 0
                state_safety = 0
                visited_times = 0#sum(1 for _ in self.visited_state_tree.intersection(state)) #using sum from rtree would lead to repeat problem

                for n in self.visited_state_tree.intersection(state):
                    if self.mark_list[n] == 0:
                        state_effiency_d += self.visited_state_effiency_d[n]
                        state_effiency_v += self.visited_state_effiency_v[n]
                        state_safety += self.visited_state_safety[n]
                        self.mark_list[n] = 1
                        visited_times += 1
    
                state_effiency_d /= visited_times
                state_effiency_v /= visited_times
                state_safety /= visited_times
                
                print("results", visited_times, state_effiency_d, state_effiency_v, state_safety)
                # print("state", state)
  
                # write to txt
                with open("results.txt", 'a') as fw:
                    fw.write(str(state)) 
                    fw.write(", ")
                    fw.write(str(visited_times)) 
                    fw.write(", ")
                    fw.write(str(state_safety)) 
                    fw.write(", ")
                    fw.write(str(state_effiency_d)) 
                    fw.write(", ")
                    fw.write(str(state_effiency_v)) 
                    fw.write("\n")
                    fw.close()               
                

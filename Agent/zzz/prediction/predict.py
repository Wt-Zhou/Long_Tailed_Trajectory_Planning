import numpy as np
import copy
import math
from Agent.zzz.frenet import Frenet_path
from Agent.zzz.prediction.gnn_prediction import GNNPredictionModel
# from Agent.zzz.prediction.gnn_prediction import GNNPredictionModel

Use_Learned_Model = False

class Prediction():
    def __init__(self, considered_obs_num, maxt, dt, robot_radius, radius_speed_ratio, move_gap):

        self.maxt = maxt
        self.dt = dt
        self.robot_radius = robot_radius
        self.radius_speed_ratio = radius_speed_ratio
        self.move_gap = move_gap
        self.considered_obs_num = considered_obs_num
        
        # Ensemble Predition Models
        history_frame = 3
        future_frame = 5
        self.obs_scale = 1
        self.heads_num = 10
        
        if Use_Learned_Model:
            self.ensemble_models = []
            self.ensemble_optimizer = []

            for i in range(self.heads_num):
                predition_model = GNNPredictionModel(history_frame * 2, future_frame * 2, self.obs_scale).to(self.device)
                predition_model.apply(self.weight_init)
                self.ensemble_models.append(predition_model)
                self.ensemble_optimizer.append(torch.optim.Adam(predition_model.parameters(), lr = 0.005))
                predition_model.train()
  
    def update_prediction(self, dynamic_map):
        self.dynamic_map = dynamic_map
        self.check_radius = self.robot_radius + self.radius_speed_ratio * self.dynamic_map.ego_vehicle.v
        if Use_Learned_Model:
            pass
        else:
            try:
                interested_vehicles = self.found_interested_vehicles(self.considered_obs_num)
                self.predict_paths = self.prediction_obstacle_uniform_speed(interested_vehicles, self.maxt, self.dt)
            except:
                self.predict_paths = []

    def check_collision(self, fp):
        
        if len(self.predict_paths) == 0 or len(fp.t) < 2 :
            return True
            
        # two circles for a vehicle
        fp_front = copy.deepcopy(fp)
        fp_back = copy.deepcopy(fp)
        
        fp_front.x = (np.array(fp.x)+np.cos(np.array(fp.yaw))*self.move_gap).tolist()
        fp_front.y = (np.array(fp.y)+np.sin(np.array(fp.yaw))*self.move_gap).tolist()
        # print("fp_front.x",fp_front.x)
        # print("fp_front.y",fp_front.y)


        fp_back.x = (np.array(fp.x)-np.cos(np.array(fp.yaw))*self.move_gap).tolist()
        fp_back.y = (np.array(fp.y)-np.sin(np.array(fp.yaw))*self.move_gap).tolist()

        for path in self.predict_paths:
            # print("path.x",path.x)
            # print("path.y",path.y)

            len_predict_t = min(len(fp.x)-1, len(path.t)-1)
            predict_step = 2
            start_predict = 2
            for t in range(start_predict, len_predict_t, predict_step):
                d = (path.x[t] - fp_front.x[t])**2 + (path.y[t] - fp_front.y[t])**2
                if d <= self.check_radius**2: 
                    return False
                d = (path.x[t] - fp_back.x[t])**2 + (path.y[t] - fp_back.y[t])**2
                if d <= self.check_radius**2: 
                    return False

        return True

    def found_interested_vehicles(self, interested_vehicles_num=3):

        interested_vehicles = []

        # Get interested vehicles by distance
        distance_tuples = []
        ego_loc = np.array([self.dynamic_map.ego_vehicle.x,self.dynamic_map.ego_vehicle.y])

        for vehicle_idx, vehicle in enumerate(self.dynamic_map.vehicles): 
            vehicle_loc = np.array([vehicle.x, vehicle.y])
            d = np.linalg.norm(vehicle_loc - ego_loc)

            distance_tuples.append((d, vehicle_idx))
            
        sorted_vehicle = sorted(distance_tuples, key=lambda vehicle_dis: vehicle_dis[0])

        for _, vehicle_idx in sorted_vehicle:
            interested_vehicles.append(self.dynamic_map.vehicles[vehicle_idx])
            if len(interested_vehicles) >= interested_vehicles_num:
                break
        return interested_vehicles

    def prediction_obstacle_uniform_speed(self, vehicles, max_prediction_time, delta_t): 
        predict_paths = []
        for vehicle in vehicles:

            predict_path_front = Frenet_path()
            predict_path_back = Frenet_path()
            predict_path_front.t = [t for t in np.arange(0.0, max_prediction_time, delta_t)]
            predict_path_back.t = [t for t in np.arange(0.0, max_prediction_time, delta_t)]
            ax = 0 #one_ob[9]
            ay = 0 #one_ob[10]
            # print("vehicle information",vehicle.x, vehicle.y, vehicle.vx, vehicle.vy, vehicle.yaw)

            vx_predict = vehicle.vx*np.ones(len(predict_path_front.t))
            vy_predict = vehicle.vy*np.ones(len(predict_path_front.t))

            x_predict = vehicle.x + np.arange(len(predict_path_front.t))*delta_t*vx_predict
            y_predict = vehicle.y + np.arange(len(predict_path_front.t))*delta_t*vy_predict
            
            predict_path_front.x = (x_predict + math.cos(vehicle.yaw)*np.ones(len(predict_path_front.t))*self.move_gap).tolist()
            predict_path_front.y = (y_predict + math.sin(vehicle.yaw)*np.ones(len(predict_path_front.t))*self.move_gap).tolist()
            predict_path_back.x = (x_predict - math.cos(vehicle.yaw)*np.ones(len(predict_path_back.t))*self.move_gap).tolist()
            predict_path_back.y = (y_predict - math.sin(vehicle.yaw)*np.ones(len(predict_path_back.t))*self.move_gap).tolist()
        
            predict_paths.append(predict_path_front)
            predict_paths.append(predict_path_back)

        return predict_paths

    def prediction_obstacle_gnn(self, vehicles, max_prediction_time, delta_t): 
        predict_paths = []
        for vehicle in vehicles:

            predict_path_front = Frenet_path()
            predict_path_back = Frenet_path()
            predict_path_front.t = [t for t in np.arange(0.0, max_prediction_time, delta_t)]
            predict_path_back.t = [t for t in np.arange(0.0, max_prediction_time, delta_t)]
            ax = 0 #one_ob[9]
            ay = 0 #one_ob[10]
            # print("vehicle information",vehicle.x, vehicle.y, vehicle.vx, vehicle.vy, vehicle.yaw)

            vx_predict = vehicle.vx*np.ones(len(predict_path_front.t))
            vy_predict = vehicle.vy*np.ones(len(predict_path_front.t))

            x_predict = vehicle.x + np.arange(len(predict_path_front.t))*delta_t*vx_predict
            y_predict = vehicle.y + np.arange(len(predict_path_front.t))*delta_t*vy_predict
            
            predict_path_front.x = (x_predict + math.cos(vehicle.yaw)*np.ones(len(predict_path_front.t))*self.move_gap).tolist()
            predict_path_front.y = (y_predict + math.sin(vehicle.yaw)*np.ones(len(predict_path_front.t))*self.move_gap).tolist()
            predict_path_back.x = (x_predict - math.cos(vehicle.yaw)*np.ones(len(predict_path_back.t))*self.move_gap).tolist()
            predict_path_back.y = (y_predict - math.sin(vehicle.yaw)*np.ones(len(predict_path_back.t))*self.move_gap).tolist()
        
            predict_paths.append(predict_path_front)
            predict_paths.append(predict_path_back)

        return predict_paths

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
    
    
    
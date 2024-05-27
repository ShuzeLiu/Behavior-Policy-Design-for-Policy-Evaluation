import numpy as np
import collections
import copy
import wandb
import pickle
import os

from agents.BaseAgent import BaseAgent
from utilities import *


class OPEAgent(BaseAgent):
    agent_name = "OPEAgent"
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        
        self.model_w_q = OPENet(config)
        self.model_w_q_target = OPENet(config)
        self.model_w_q_target.load_state_dict(self.model_w_q.state_dict())
        self.model_w_r = OPENet(config)
        self.model_w_u = OPENet(config)
        self.model_w_u_target = OPENet(config)
        self.model_w_u_target.load_state_dict(self.model_w_u.state_dict())
        
        # self.

        self.loss_fn_w_q  = torch.nn.MSELoss()
        self.loss_fn_w_r  = torch.nn.MSELoss()
        self.loss_fn_w_u  = torch.nn.MSELoss()

        self.optimizer_w_q = torch.optim.Adam(self.model_w_q.parameters(), lr=config.learning_rate_w_q)
        self.optimizer_w_r = torch.optim.Adam(self.model_w_r.parameters(), lr=config.learning_rate_w_r)
        self.optimizer_w_u = torch.optim.Adam(self.model_w_u.parameters(), lr=config.learning_rate_w_u)
        
        self.offline_data = self.config.offline_data
        self.data_w_q = None
        self.data_w_r = None
        self.data_w_u = None


        self.batch_size = config.batch_size
        self.num_epoch = config.num_epoch
        self.train_percent = config.training_percent
        
        self.learn_from_offline_data()
        

    def data_format(self):
        # s,a,r,s',mask  => s,a,r,s',a',mask
        self.offline_data_new = []
        for i in range(len(self.offline_data)):
            np_state = self.offline_data[i][0]
            np_action = self.offline_data[i][1][0]
            int_action_index = self.offline_data[i][1][1]
            float_reward = self.offline_data[i][2]
            # np_next_state = self.offline_data[i][3]
            float_done = self.offline_data[i][4]

            # np_next_action, int_next_action_index = self.config.target_policy.get_action_and_index(np_next_state)
            np_state_action = np.hstack((np_state, np_action[1:]))
            # np_next_state_action = np.hstack((np_next_state, np_next_action[1:]))

            # print(np_state)
            # print(np_action)
            # print(np_state_action)
            # print(int_action_index)
            # print(float_reward)
            # print(np_next_state)
            # print(float_done)

            # print("-----------------------------------------------------")
            self.offline_data_new_point = []
            self.offline_data_new_point.append(tensor(np_state_action, np.float32, self.config.device))
            self.offline_data_new_point.append(tensor(int_action_index, np.int64, self.config.device))
            self.offline_data_new_point.append(tensor(float_reward, np.float32, self.config.device))
            # self.offline_data_new_point.append(tensor(np_next_state_action, np.float32, self.config.device))
            # self.offline_data_new_point.append(tensor(int_next_action_index, np.int64, self.config.device))
            self.offline_data_new_point.append(1)
            self.offline_data_new_point.append(2)
            self.offline_data_new_point.append(tensor(1 - float_done, np.float32, self.config.device))
            self.offline_data_new.append(self.offline_data_new_point)

            # self.offline_data[i][0] = tensor(np_state_action, np.float32, self.config.device)
            # self.offline_data[i][1] = tensor(int_action_index, np.int64, self.config.device)
            # self.offline_data[i][2] = tensor(float_reward, np.float32, self.config.device)
            # self.offline_data[i][3] = tensor(np_next_state_action, np.float32, self.config.device)
            # self.offline_data[i][4] = tensor(int_next_action_index, np.int64, self.config.device)
            # self.offline_data[i].append(tensor(1 - float_done, np.float32, self.config.device))  #mask
            
        # self.offline_data = self.offline_data_new

    def data_prepare_w_q(self):
        self.data_w_q = self.offline_data_new
    
    def data_resample_w_q(self):
        for i in range(len(self.offline_data)):
            # np_state = self.offline_data[i][0]
            # np_action = self.offline_data[i][1][0]
            # int_action_index = self.offline_data[i][1][1]
            # float_reward = self.offline_data[i][2]
            np_next_state = self.offline_data[i][3]
            # float_done = self.offline_data[i][4]
            if self.config.time_feature:
                np_next_action, int_next_action_index = self.config.target_policy.get_action_and_index(np_next_state[1:])
            else:
                np_next_action, int_next_action_index = self.config.target_policy.get_action_and_index(np_next_state)
            # np_state_action = np.hstack((np_state, np_action[1:]))
            np_next_state_action = np.hstack((np_next_state, np_next_action[1:]))

            # print(np_state)
            # print(np_action)
            # print(np_state_action)
            # print(int_action_index)
            # print(float_reward)
            # print(np_next_state)
            # print(float_done)
            self.data_w_q[i][3] = tensor(np_next_state_action, np.float32, self.config.device)
            self.data_w_q[i][4] = tensor(int_next_action_index, np.int64, self.config.device)
        # print("----------------------data_prepare_w_q begin-------------------------------")
        # print(len(self.data_w_q))
        # for row in self.data_w_q:
        #     print(row)
        # print("----------------------data_prepare_w_q end-------------------------------")
            # self.offline_data_new_point = []
            # self.offline_data_new_point.append(tensor(np_state_action, np.float32, self.config.device))
            # self.offline_data_new_point.append(tensor(int_action_index, np.int64, self.config.device))
            # self.offline_data_new_point.append(tensor(float_reward, np.float32, self.config.device))
            # self.offline_data_new_point.append(tensor(np_next_state_action, np.float32, self.config.device))
            # self.offline_data_new_point.append(tensor(int_next_action_index, np.int64, self.config.device))
            # self.offline_data_new_point.append(tensor(1 - float_done, np.float32, self.config.device))
            # self.offline_data_new.append(self.offline_data_new_point)
        # print(self.data_w_q[0])
        # print(self.data_w_q[0][0].get_device())
        # self.data_w_q.to(self.device)

    def learn_w_q(self):

        update_count = 0
        for epoch in range(1, self.config.num_epoch_w_q + 1):
            self.data_resample_w_q()
            data_set_w_q = OPEDataset(self.data_w_q)
            train_num = int(len(data_set_w_q)*(self.train_percent))
            test_num =  len(data_set_w_q) -  train_num
            train_set_w_q, test_set_w_q = torch.utils.data.random_split(data_set_w_q, [train_num, test_num] )
            train_generator_w_q = torch.utils.data.DataLoader(train_set_w_q, batch_size=self.batch_size, shuffle = True)
            test_generator_w_q = torch.utils.data.DataLoader(test_set_w_q, batch_size=self.batch_size)

            for local_batch in train_generator_w_q:
                update_count += 1
                # print("-------------------")
                # print(local_batch)
                # print("-------------------")
                self.model_w_q.eval()
                state = local_batch[0]
                # state = self.config.env.state_list_to_phi_list_device(state, self.config.device)
                action = local_batch[1]
                reward = local_batch[2]
                next_state = local_batch[3]
                # next_state = self.config.env.state_list_to_phi_list_device(next_state, self.config.device)
                next_action = local_batch[4]
                mask = local_batch[5]
                with torch.no_grad():
                    next_y = self.model_w_q_target(next_state)
                    next_y = next_y.gather(1, next_action.view(-1, 1))
                    next_y = next_y.view(-1) * mask
                    y = next_y + reward

                self.model_w_q.train()
                pred = self.model_w_q(state).gather(1, action.view(-1, 1))
                pred = pred.view(-1)
                loss = self.loss_fn_w_q(pred, y)
                self.optimizer_w_q.zero_grad()
                loss.backward()
                self.optimizer_w_q.step()
                
                if update_count % self.config.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(self.model_w_q_target.parameters(), self.model_w_q.parameters()):
                        target_network_param.data.copy_(
                            q_network_param.data
                            # args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )

            if epoch % max((self.config.num_epoch // 20), 1) == 0:
                train_loss = TD_loss(train_generator_w_q, self.model_w_q, self.loss_fn_w_q, self.config)
                test_loss = TD_loss(test_generator_w_q, self.model_w_q, self.loss_fn_w_q, self.config)
                target_train_loss = TD_loss(train_generator_w_q, self.model_w_q_target, self.loss_fn_w_q, self.config)
                target_test_loss = TD_loss(test_generator_w_q, self.model_w_q_target, self.loss_fn_w_q, self.config)

                if self.config.wandb_switch:
                    wandb.log({"w_q_test_TD_error": test_loss, "w_q_train_TD_error": train_loss, "step_w_q": epoch,\
                               "w_q_target_test_TD_error": target_test_loss, "w_q_target_train_TD_error": target_train_loss})    


    def data_prepare_w_r(self):
        self.data_w_r = []
        for i in range(len(self.offline_data_new)):
            data = [self.offline_data_new[i][0], self.offline_data_new[i][1], self.offline_data_new[i][2]] 
            self.data_w_r.append(data)



    def learn_w_r(self):
        data_set_w_r = OPEDataset(self.data_w_r)
        train_num = int(len(data_set_w_r)*(self.train_percent))
        train_set_w_r, test_set_w_r = torch.utils.data.random_split(data_set_w_r, [train_num, len(data_set_w_r) -  train_num] )
        training_generator_w_r = torch.utils.data.DataLoader(train_set_w_r, batch_size=self.batch_size, shuffle = True)
        test_generator_w_r = torch.utils.data.DataLoader(test_set_w_r, batch_size=self.batch_size)
        
        for epoch in range(1, self.config.num_epoch_w_r + 1):
            training_loss = 0
            for local_batch in training_generator_w_r:
                state = local_batch[0]
                # state = self.config.env.state_list_to_phi_list_device(state, self.config.device)
                action = local_batch[1]
                v = local_batch[2]

                self.model_w_r.train()
                pred = self.model_w_r(state)
                pred = pred.gather(1, action.view(-1, 1)).view(-1)
                loss = self.loss_fn_w_r(pred, v)
                self.optimizer_w_r.zero_grad()
                loss.backward()
                self.optimizer_w_r.step()
                training_loss += loss.item()
            

            if epoch % max((self.config.num_epoch // 20), 1) == 0:
                train_loss = supervised_loss(training_generator_w_r, self.model_w_r, self.loss_fn_w_r, self.config)
                test_loss = supervised_loss(test_generator_w_r, self.model_w_r, self.loss_fn_w_r, self.config)
                if self.config.wandb_switch:
                    wandb.log({"w_r_test_TD_error": test_loss, "w_r_train_TD_error": train_loss, "step_w_r": epoch})
            
    



    def data_prepare_w_u(self):
        self.data_w_u = []
        self.model_w_q.eval()
        self.model_w_r.eval()

        for i in range(len(self.offline_data_new)):
            with torch.no_grad():
                state = self.offline_data_new[i][0]
                # state = self.config.env.state_to_phi_device(state, self.config.device)
                action = self.offline_data_new[i][1]

                r_value = self.model_w_r(state)[action]
                q_value = self.model_w_q(state)[action]
                r_hat = 2 * r_value * q_value - torch.square(r_value)
                # print("r_value:", r_value)
                # print("q_value:", q_value)
                # print("r_hat:", r_hat)
                r_hat = tensor(r_hat, np.float32, self.config.device)
                self.offline_data_new[i][2] = r_hat
            # data = [self.offline_data_new[i][0], self.offline_data_new[i][1], r_hat, self.offline_data_new[i][3], self.offline_data_new[i][4], self.offline_data_new[i][5]] 
            # self.data_w_u.append(data)
        self.data_w_u = self.offline_data_new


    def data_resample_w_u(self):
        for i in range(len(self.offline_data)):
            # np_state = self.offline_data[i][0]
            # np_action = self.offline_data[i][1][0]
            # int_action_index = self.offline_data[i][1][1]
            # float_reward = self.offline_data[i][2]
            np_next_state = self.offline_data[i][3]
            # float_done = self.offline_data[i][4]
            if self.config.time_feature:
                np_next_action, int_next_action_index = self.config.target_policy.get_action_and_index(np_next_state[1:])
            else:
                np_next_action, int_next_action_index = self.config.target_policy.get_action_and_index(np_next_state)
            # np_state_action = np.hstack((np_state, np_action[1:]))
            np_next_state_action = np.hstack((np_next_state, np_next_action[1:]))

            # print(np_state)
            # print(np_action)
            # print(np_state_action)
            # print(int_action_index)
            # print(float_reward)
            # print(np_next_state)
            # print(float_done)
            self.data_w_u[i][3] = tensor(np_next_state_action, np.float32, self.config.device)
            self.data_w_u[i][4] = tensor(int_next_action_index, np.int64, self.config.device)
        # print("----------------------data_prepare_w_u begin-------------------------------")
        # print(len(self.data_w_u))
        # for row in self.data_w_u:
        #     print(row)
        # print("----------------------data_prepare_w_u end-------------------------------")
            # self.offline_data_new_point = []
            # self.offline_data_new_point.append(tensor(np_state_action, np.float32, self.config.device))
            # self.offline_data_new_point.append(tensor(int_action_index, np.int64, self.config.device))
            # self.offline_data_new_point.append(tensor(float_reward, np.float32, self.config.device))
            # self.offline_data_new_point.append(tensor(np_next_state_action, np.float32, self.config.device))
            # self.offline_data_new_point.append(tensor(int_next_action_index, np.int64, self.config.device))
            # self.offline_data_new_point.append(tensor(1 - float_done, np.float32, self.config.device))
            # self.offline_data_new.append(self.offline_data_new_point)
        # print(self.data_w_q[0])
        # print(self.data_w_q[0][0].get_device())
        # self.data_w_q.to(self.device)
            
    def learn_w_u(self):

        
        update_count = 0
        for epoch in range(1, self.config.num_epoch_w_u + 1):
            self.data_resample_w_u()
            data_set_w_u = OPEDataset(self.data_w_u)
            train_num = int(len(data_set_w_u)*(self.train_percent))
            test_num =  len(data_set_w_u) -  train_num
            train_set_w_u, test_set_w_u = torch.utils.data.random_split(data_set_w_u, [train_num, test_num] )
            train_generator_w_u = torch.utils.data.DataLoader(train_set_w_u, batch_size=self.batch_size, shuffle = True)
            test_generator_w_u = torch.utils.data.DataLoader(test_set_w_u, batch_size=self.batch_size)
            for local_batch in train_generator_w_u:
                update_count += 1
                self.model_w_u.eval()
                state = local_batch[0]
                # state = self.config.env.state_list_to_phi_list_device(state, self.config.device)
                action = local_batch[1]
                reward = local_batch[2]
                next_state = local_batch[3]
                # next_state = self.config.env.state_list_to_phi_list_device(next_state, self.config.device)
                next_action = local_batch[4]
                mask = local_batch[5]
                with torch.no_grad():
                    next_y = self.model_w_u_target(next_state)                    
                    next_y = next_y.gather(1, next_action.view(-1, 1))
                    next_y = next_y.view(-1) * mask
                    y = next_y + reward

                self.model_w_u.train()
                pred = self.model_w_u(state).gather(1, action.view(-1, 1))
                pred = pred.view(-1)
                loss = self.loss_fn_w_u(pred, y)
                self.optimizer_w_u.zero_grad()
                loss.backward()
                self.optimizer_w_u.step()

                if update_count % self.config.target_network_frequency == 0:
                    for target_network_param, u_network_param in zip(self.model_w_u_target.parameters(), self.model_w_u.parameters()):
                        target_network_param.data.copy_(
                            u_network_param.data
                            # args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )

            if epoch % max((self.config.num_epoch // 20), 1) == 0:
                train_loss = TD_loss(train_generator_w_u, self.model_w_u, self.loss_fn_w_u, self.config)
                test_loss = TD_loss(test_generator_w_u, self.model_w_u, self.loss_fn_w_u, self.config)
                target_train_loss = TD_loss(train_generator_w_u, self.model_w_u_target, self.loss_fn_w_u, self.config)
                target_test_loss = TD_loss(test_generator_w_u, self.model_w_u_target, self.loss_fn_w_u, self.config)

                if self.config.wandb_switch:
                    wandb.log({"w_u_test_TD_error": test_loss, "w_u_train_TD_error": train_loss, "step_w_u": epoch,\
                               "w_u_target_test_TD_error": target_test_loss, "w_u_target_train_TD_error": target_train_loss})
                                    
            # wandb.log({"test_TD_error_w_u": test_loss, "train_TD_error_w_u": train_loss})
    def save_w_u(self):
        if self.config.wandb_sweep_id != "test":
            name = self.config.tag
            p = f"data_mujoco/03_trained_OPE_model_100_repeat_model/{self.config.env_id}/"
            #check if p exists, otherwise create p
            if not os.path.exists(p):
                os.makedirs(p)
            torch.save(self.model_w_u, p + name)
    



    def run_all_episode(self):
        return

    

#------------------- learn r q


    def learn_from_offline_data(self):
        print("data_format")
        self.data_format()
        print("data_prepare_w_q")
        self.data_prepare_w_q()
        print("learn_model_w_q")
        self.learn_w_q()
        print("data_prepare_w_r")
        self.data_prepare_w_r()
        print("learn_model_w_r")
        self.learn_w_r()
        print("data_prepare_w_u")
        self.data_prepare_w_u()
        print("learn_model_w_u")
        self.learn_w_u()
        print("save_model_w_u")
        self.save_w_u()
        print("done")
    



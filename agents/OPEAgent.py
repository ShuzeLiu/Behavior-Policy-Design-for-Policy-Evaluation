import numpy as np
import collections
import copy
import wandb
import pickle

from agents.BaseAgent import BaseAgent
from utilities import *


class OPEAgent(BaseAgent):
    agent_name = "OPEAgent"
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        
        self.model_w_q = OPENet(config)
        self.model_w_1 = OPENet(config)
        self.model_w_2 = OPENet(config)
        self.model_w_r = OPENet(config)
        self.model_w_u = OPENet(config)

        self.loss_fn_w_q  = torch.nn.MSELoss()
        self.loss_fn_w_1  = torch.nn.MSELoss()
        self.loss_fn_w_2  = torch.nn.MSELoss()
        self.loss_fn_w_r  = torch.nn.MSELoss()
        self.loss_fn_w_u  = torch.nn.MSELoss()

        self.optimizer_w_q = torch.optim.Adam(self.model_w_q.parameters(), lr=config.learning_rate_w_q)
        self.optimizer_w_1 = torch.optim.Adam(self.model_w_1.parameters(), lr=config.learning_rate_w_1)
        self.optimizer_w_2 = torch.optim.Adam(self.model_w_2.parameters(), lr=config.learning_rate_w_2)
        self.optimizer_w_r = torch.optim.Adam(self.model_w_r.parameters(), lr=config.learning_rate_w_r)
        self.optimizer_w_u = torch.optim.Adam(self.model_w_u.parameters(), lr=config.learning_rate_w_u)

        # self.optimizer_w_q = torch.optim.SGD(self.model_w_q.parameters(), lr=config.learning_rate_w_q)
        # self.optimizer_w_1 = torch.optim.SGD(self.model_w_1.parameters(), lr=config.learning_rate_w_1)
        # self.optimizer_w_2 = torch.optim.SGD(self.model_w_2.parameters(), lr=config.learning_rate_w_2)
        # self.optimizer_w_r = torch.optim.SGD(self.model_w_r.parameters(), lr=config.learning_rate_w_r)
        # self.optimizer_w_u = torch.optim.SGD(self.model_w_u.parameters(), lr=config.learning_rate_w_u)

        
        self.offline_data = self.config.env.offline_data
        self.data_w_q = None
        self.data_w_1 = None
        self.data_w_2 = None
        self.data_w_r = None
        self.data_w_u = None


        self.batch_size = config.batch_size
        self.num_epoch = config.num_epoch
        self.train_percent = config.training_percent
        
        self.learn_from_offline_data2()
        self.ope_policy = OPEPolicy(self.config.env, self.config.target_policy, self.model_w_u, self.config.device)

    def data_format(self):
        # s,a,r,s',mask  => s,a,r,s',a',mask
        for i in range(len(self.offline_data)):
            self.offline_data[i][0] = tensor(self.offline_data[i][0], np.int64, self.config.device)
            self.offline_data[i][1] = tensor(self.offline_data[i][1], np.int64, self.config.device)
            self.offline_data[i][2] = tensor(self.offline_data[i][2], np.float32, self.config.device)
            self.offline_data[i][3] = tensor(self.offline_data[i][3], np.int64, self.config.device)
            self.offline_data[i].append(tensor(1-self.offline_data[i][4], np.float32, self.config.device))  #mask
            self.offline_data[i][4] = tensor(self.config.target_policy.get_action(self.offline_data[i][3]), np.int64, self.config.device)


    def data_prepare_w_q(self):
        self.data_w_q = self.offline_data
        # print(self.data_w_q[0])
        # print(self.data_w_q[0][0].get_device())
        # self.data_w_q.to(self.device)
        

    def learn_w_q(self):
        data_set_w_q = OPEDataset(self.data_w_q)
        train_num = int(len(data_set_w_q)*(self.train_percent))
        test_num =  len(data_set_w_q) -  train_num
        train_set_w_q, test_set_w_q = torch.utils.data.random_split(data_set_w_q, [train_num, test_num] )
        train_generator_w_q = torch.utils.data.DataLoader(train_set_w_q, batch_size=self.batch_size)
        test_generator_w_q = torch.utils.data.DataLoader(test_set_w_q, batch_size=self.batch_size)
        
        for epoch in range(self.config.num_epoch_w_q):
            for local_batch in train_generator_w_q:
                self.model_w_q.eval()
                state = local_batch[0]
                state = self.config.env.state_list_to_phi_list_device(state, self.config.device)
                action = local_batch[1]
                reward = local_batch[2]
                next_state = local_batch[3]
                next_state = self.config.env.state_list_to_phi_list_device(next_state, self.config.device)
                next_action = local_batch[4]
                mask = local_batch[5]
                with torch.no_grad():
                    next_y = self.model_w_q(next_state)
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
                
            if epoch % (self.config.num_epoch // 10) == 0:
                train_loss = TD_loss(train_generator_w_q, self.model_w_q, self.loss_fn_w_q, self.config)
                test_loss = TD_loss(test_generator_w_q, self.model_w_q, self.loss_fn_w_q, self.config)
                if self.config.wandb_switch:
                    wandb.log({"w_q_test_TD_error": test_loss, "w_q_train_TD_error": train_loss, "step_w_q": epoch})
    
    def data_prepare_w_1_2(self):
        self.data_w_2 = []
        self.model_w_q.eval()
        for i in range(len(self.offline_data)):
            with torch.no_grad():
                next_state = self.offline_data[i][3]
                next_state = self.config.env.state_to_phi_device(next_state, self.config.device)
                pred_q = self.model_w_q(next_state)
                v = torch.dot( tensor(self.config.target_policy.get_prob_dis(next_state) , np.float32, self.config.device), pred_q )
            data = [self.offline_data[i][0], self.offline_data[i][1], v * self.offline_data[i][5]  ]
            self.data_w_2.append(data)


        self.data_w_1 = []
        for i in range(len(self.offline_data)):
            data = [self.offline_data[i][0], self.offline_data[i][1], torch.square(self.data_w_2[i][2])] 
            self.data_w_1.append(data)



    def learn_w_1(self):
        data_set_w_1 = OPEDataset(self.data_w_1)
        train_num = int(len(data_set_w_1)*(self.train_percent))
        train_set_w_1, test_set_w_1 = torch.utils.data.random_split(data_set_w_1, [train_num, len(data_set_w_1) -  train_num] )
        training_generator_w_1 = torch.utils.data.DataLoader(train_set_w_1, batch_size=self.batch_size)
        test_generator_w_1 = torch.utils.data.DataLoader(test_set_w_1, batch_size=self.batch_size)
        


        for epoch in range(self.config.num_epoch_w_1):
            training_loss = 0
            for local_batch in training_generator_w_1:
                state = local_batch[0]
                state = self.config.env.state_list_to_phi_list_device(state, self.config.device)
                action = local_batch[1]
                v = local_batch[2]

                self.model_w_1.train()
                pred = self.model_w_1(state)
                pred = pred.gather(1, action.view(-1, 1)).view(-1)
                loss = self.loss_fn_w_1(pred, v)
                self.optimizer_w_1.zero_grad()
                loss.backward()
                self.optimizer_w_1.step()
                training_loss += loss.item()

            if epoch % (self.config.num_epoch // 10) == 0:
                train_loss = supervised_loss(training_generator_w_1, self.model_w_1, self.loss_fn_w_1, self.config)
                test_loss = supervised_loss(test_generator_w_1, self.model_w_1, self.loss_fn_w_1, self.config)
                if self.config.wandb_switch:
                    wandb.log({"w_1_test_TD_error": test_loss, "w_1_train_TD_error": train_loss, "step_w_1": epoch})
    
               
            
            
            # print(f"training_loss: {training_loss/len(training_generator_w_1)}",  f"test_loss: {test_loss/len(test_generator_w_print1) }")
                


    def learn_w_2(self):
        data_set_w_2 = OPEDataset(self.data_w_2)
        train_num = int(len(data_set_w_2)*(self.train_percent))
        train_set_w_2, test_set_w_2 = torch.utils.data.random_split(data_set_w_2, [train_num, len(data_set_w_2) -  train_num] )
        training_generator_w_2 = torch.utils.data.DataLoader(train_set_w_2, batch_size=self.batch_size)
        test_generator_w_2 = torch.utils.data.DataLoader(test_set_w_2, batch_size=self.batch_size)
        
        for epoch in range(self.config.num_epoch_w_2):
            training_loss = 0
            for local_batch in training_generator_w_2:
                state = local_batch[0]
                state = self.config.env.state_list_to_phi_list_device(state, self.config.device)
                action = local_batch[1]
                v = local_batch[2]

                self.model_w_2.train()
                pred = self.model_w_2(state)
                pred = pred.gather(1, action.view(-1, 1)).view(-1)
                loss = self.loss_fn_w_2(pred, v)
                self.optimizer_w_2.zero_grad()
                loss.backward()
                self.optimizer_w_2.step()
                training_loss += loss.item()
            

            if epoch % (self.config.num_epoch // 10) == 0:
                train_loss = supervised_loss(training_generator_w_2, self.model_w_2, self.loss_fn_w_2, self.config)
                test_loss = supervised_loss(test_generator_w_2, self.model_w_2, self.loss_fn_w_2, self.config)
                if self.config.wandb_switch:
                    wandb.log({"w_2_test_TD_error": test_loss, "w_2_train_TD_error": train_loss, "step_w_2": epoch})
            
            
            # print(f"training_loss: {training_loss/len(training_generator_w_2)}",  f"test_loss: {test_loss/len(test_generator_w_2) }")
            # wandb.log({"training_loss": training_loss/len(training_generator_w_2), "test_loss": test_loss/len(test_generator_w_2)})
               



    def data_prepare_w_u(self):
        self.data_w_u = []

        self.model_w_q.eval()
        self.model_w_1.eval()
        self.model_w_2.eval()

        for i in range(len(self.offline_data)):
            with torch.no_grad():
                state = self.offline_data[i][0]
                state = self.config.env.state_to_phi_device(state, self.config.device)
                action = self.offline_data[i][1]
                next_state = self.offline_data[i][3]
                next_state = self.config.env.state_to_phi_device(next_state, self.config.device)
                next_action = self.offline_data[i][4]

                nu = self.model_w_1(state)[action] - torch.square(self.model_w_2(state)[action])
                pred_q = self.model_w_q(next_state)
                v_next_state = torch.dot( tensor(self.config.target_policy.get_prob_dis(next_state), np.float32, self.config.device), pred_q)
                r = nu + torch.square(self.model_w_q(state)[action]) - torch.square(v_next_state)
                r = tensor(r, np.float32, self.config.device)

            data = [self.offline_data[i][0], action, r, self.offline_data[i][3], next_action, self.offline_data[i][5]] 
            self.data_w_u.append(data)


    def learn_w_u(self):
        data_set_w_u = OPEDataset(self.data_w_u)
        train_num = int(len(data_set_w_u)*(self.train_percent))
        test_num =  len(data_set_w_u) -  train_num
        train_set_w_u, test_set_w_u = torch.utils.data.random_split(data_set_w_u, [train_num, test_num] )
        train_generator_w_u = torch.utils.data.DataLoader(train_set_w_u, batch_size=self.batch_size)
        test_generator_w_u = torch.utils.data.DataLoader(test_set_w_u, batch_size=self.batch_size)
        
        for epoch in range(self.config.num_epoch_w_u):
            for local_batch in train_generator_w_u:
                self.model_w_u.eval()
                state = local_batch[0]
                state = self.config.env.state_list_to_phi_list_device(state, self.config.device)
                action = local_batch[1]
                reward = local_batch[2]
                next_state = local_batch[3]
                next_state = self.config.env.state_list_to_phi_list_device(next_state, self.config.device)
                next_action = local_batch[4]
                mask = local_batch[5]
                with torch.no_grad():
                    next_y = self.model_w_u(next_state)                    
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


            if epoch % (self.config.num_epoch // 10) == 0:
                train_loss = TD_loss(train_generator_w_u, self.model_w_u, self.loss_fn_w_u, self.config)
                test_loss = TD_loss(test_generator_w_u, self.model_w_u, self.loss_fn_w_u, self.config)
                if self.config.wandb_switch:
                    wandb.log({"w_u_test_TD_error": test_loss, "w_u_train_TD_error": train_loss, "step_w_u": epoch})
    
                                    
            # wandb.log({"test_TD_error_w_u": test_loss, "train_TD_error_w_u": train_loss})
    def save_w_u(self):
        name = "e_"+ str(self.config.env.width) + "_re_" + str(self.config.repeat)\
                     + "_pid_" + str(self.config.policy_id) + "_" + str(self.config.feature_type)
        torch.save(self.model_w_u, "./data/large_linear/03_trained_model_linear/"+name)


    def learn_from_offline_data(self):
        print("data_format")
        self.data_format()
        print("data_prepare_w_q")
        self.data_prepare_w_q()
        print("learn_model_w_q")
        self.learn_w_q()
        print("data_prepare_w_1_2")
        self.data_prepare_w_1_2()
        print("learn_model_w_1")
        self.learn_w_1()
        print("learn_model_w_2")
        self.learn_w_2()
        print("data_prepare_w_u")
        self.data_prepare_w_u()
        print("learn_model_w_u")
        self.learn_w_u()
        # print("save_model_w_u")
        # self.save_w_u()
        print("done")
    



    def run_all_episode(self):
        return

    

#------------------- learn r q


    def learn_from_offline_data2(self):
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
        self.data_prepare_w_u_2()
        print("learn_model_w_u")
        self.learn_w_u()
        print("save_model_w_u")
        self.save_w_u()
        print("done")
    

    def data_prepare_w_r(self):
        self.data_w_r = []
        for i in range(len(self.offline_data)):
            data = [self.offline_data[i][0], self.offline_data[i][1], self.offline_data[i][2]] 
            self.data_w_r.append(data)



    def learn_w_r(self):
        data_set_w_r = OPEDataset(self.data_w_r)
        train_num = int(len(data_set_w_r)*(self.train_percent))
        train_set_w_r, test_set_w_r = torch.utils.data.random_split(data_set_w_r, [train_num, len(data_set_w_r) -  train_num] )
        training_generator_w_r = torch.utils.data.DataLoader(train_set_w_r, batch_size=self.batch_size)
        test_generator_w_r = torch.utils.data.DataLoader(test_set_w_r, batch_size=self.batch_size)
        
        for epoch in range(self.config.num_epoch_w_r):
            training_loss = 0
            for local_batch in training_generator_w_r:
                state = local_batch[0]
                state = self.config.env.state_list_to_phi_list_device(state, self.config.device)
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
            

            if epoch % (self.config.num_epoch // 10) == 0:
                train_loss = supervised_loss(training_generator_w_r, self.model_w_r, self.loss_fn_w_r, self.config)
                test_loss = supervised_loss(test_generator_w_r, self.model_w_r, self.loss_fn_w_r, self.config)
                if self.config.wandb_switch:
                    wandb.log({"w_r_test_TD_error": test_loss, "w_r_train_TD_error": train_loss, "step_w_r": epoch})
            
    



    def data_prepare_w_u_2(self):
        self.data_w_u = []
        self.model_w_q.eval()
        self.model_w_r.eval()

        for i in range(len(self.offline_data)):
            with torch.no_grad():
                state = self.offline_data[i][0]
                state = self.config.env.state_to_phi_device(state, self.config.device)
                action = self.offline_data[i][1]

                r_value = self.model_w_r(state)[action]
                q_value = self.model_w_q(state)[action]
                r_hat = 2 * r_value * q_value - torch.square(r_value)
                # print("r_value:", r_value)
                # print("q_value:", q_value)
                # print("r_hat:", r_hat)
                r_hat = tensor(r_hat, np.float32, self.config.device)
                
            data = [self.offline_data[i][0], self.offline_data[i][1], r_hat, self.offline_data[i][3], self.offline_data[i][4], self.offline_data[i][5]] 
            self.data_w_u.append(data)


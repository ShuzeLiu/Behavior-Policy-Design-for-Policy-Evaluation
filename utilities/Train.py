import torch


#s,a,r,s,a,mask
def TD_loss(data_generator, model, loss_fn, config):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for local_batch in data_generator:
            state = local_batch[0]
            state = config.env.state_list_to_phi_list_device(state, config.device)
            action = local_batch[1]
            reward = local_batch[2]
            next_state = local_batch[3]
            next_state = config.env.state_list_to_phi_list_device(next_state, config.device)
            next_action = local_batch[4]
            mask = local_batch[5]
            next_y = model(next_state)                    
            next_y = next_y.gather(1, next_action.view(-1, 1))
            next_y = next_y.view(-1) * mask
            y = next_y + reward
            
            pred = model(state).gather(1, action.view(-1, 1))
            pred = pred.view(-1)
        
            loss = loss_fn(pred, y)
            total_loss += loss.item()
        total_loss /= len(data_generator)
    return loss

#Input: s,a  Output: y 
def supervised_loss(data_generator, model, loss_fn, config):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for local_batch in data_generator:
            state = local_batch[0]
            state = config.env.state_list_to_phi_list_device(state, config.device)
            action = local_batch[1]
            v = local_batch[2]
            pred = model(state)
            pred = pred.gather(1, action.view(-1, 1)).view(-1)
            loss = loss_fn(pred, v)
            total_loss += loss.item()
        total_loss /= len(data_generator)
    return loss
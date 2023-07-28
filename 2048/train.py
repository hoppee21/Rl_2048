from game import Game
from actor import Actor
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


ACTION_NAMES = ["left", "up", "right", "down"]


def test():
    game = Game()
    game.print_state()
    print(game.available_actions())

    for action in game.available_actions():
        print(ACTION_NAMES[action])
        print(get_possible_reward(game,action))

def get_possible_reward(game, action):
    temp =game.copy()
    reward =temp.do_action(action)
    return reward

def play_game(test_game):
    while True:
        print(test_game.available_actions())
        test_game.print_state()
        try:
            x = int(input("action: "))
        except:
            continue
        if x == -1: break
        reward = test_game.do_action(x)
        print(reward)



def random_play():

    test_game = Game()
    actor = Actor().to('cuda:0')
    weights_path = 'policy_network_weights.pth'
    actor.load_state_dict(torch.load(weights_path))
    while not test_game.game_over():
        test_game.print_state()
        input_tensor = test_game.ND2tensor().to('cuda:0')
        output_tensor = actor(input_tensor)
        action = torch.argmax(output_tensor).item()
        print(ACTION_NAMES[action])
        time.sleep(1)
        test_game.do_action(action)
    return test_game.score()

def actor_play(game, actor):
    # game.print_state()
    input_tensor = game.ND2tensor().to('cuda:0')
    output_tensor = actor(input_tensor)
    # print(output_tensor)
    action = torch.argmax(output_tensor).item()
    # print(action)
    reward = game.do_action(action)
    # print(ACTION_NAMES[action])
    # print(reward)

    return action ,reward,  output_tensor

def random_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def pre_train():
    num_episodes = 200  # Replace with the desired number of training episodes
    actor = Actor().to('cuda:0')
    actor.train()
    optimizer = optim.Adam(actor.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    for episode in range(num_episodes):
        print(f"episode {episode+1}\n-------------------------------")
        game = Game()

        while not game.game_over():


            input_tensor = game.ND2tensor().to('cuda:0')
            output_tensor = actor(input_tensor)
            label_output = [0,0,0,0]

            for action in game.available_actions(): 
                reward = get_possible_reward(game ,action)
                label_output[action] = reward   
            softmax_np = np.exp(label_output)/np.sum(np.exp(label_output))
            label_tensor = torch.tensor(softmax_np).to('cuda:0')

            loss = loss_fn(output_tensor, label_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            next_action = label_output.index(max(label_output))
            game.do_action(next_action)
    
    weights_path = 'policy_network_weights.pth'
    torch.save(actor.state_dict(), weights_path)
    print("pr-Training completed!")


def compute_returns(rewards, gamma=0.6):
    R = 0
    returns = []
    for r in reversed(rewards):
        if r != 0:
            R = r + gamma * R
            returns.insert(0, R)
        else:
            returns.insert(0, 0)
    return returns

def train_policy(optimizer, returns, log_probs):
    returns = torch.tensor(returns).to('cuda:0')
    log_probs = torch.stack(log_probs).to('cuda:0')
    loss = -torch.mean(log_probs * returns)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train():
    num_episodes = 100000  # Replace with the desired number of training episodes
    actor = Actor().to('cuda:0')
    weights_path = 'policy_network_weights.pth'
    actor.load_state_dict(torch.load(weights_path))
    optimizer = optim.Adam(actor.parameters(), lr=0.001)
    
    for episode in range(num_episodes):

        print(f"episode {episode+1}\n-------------------------------")
        game = Game()
        rewards = []
        log_probs = []
        num_round = 0

        while not game.game_over() and num_round <= 100:
            action ,  reward, action_probs= actor_play(game, actor)
            # next_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            # Compute log probabilities for the selected action
            log_prob = torch.log(action_probs[action])
            log_probs.append(log_prob)
            num_round+=1

        returns = compute_returns(rewards)
        # print(returns)
        train_policy(optimizer, returns, log_probs)

    weights_path = 'policy_network_weights.pth'
    torch.save(actor.state_dict(), weights_path)
    print("Training completed!")

if __name__ == '__main__':
    pre_train()
    # random_play()
    # train()
    # test()
    


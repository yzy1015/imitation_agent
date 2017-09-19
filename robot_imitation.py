from expert1 import *
import numpy as np
import roboschool
import gym 
import tensorflow as tf
from pathlib import Path

class learning_agent(object):
    def __init__(self, env):
        self.env = env
        # agent memory
        self.trajectory = [] # seq of state, union of state
        self.temporary_traj = [] # store new comming unlabel seq
        self.expert_action_history = [] # seq of expert action, union of action
        self.save_freq = 0
        # define network structure (state input and action output)
        self.state, self.action =  self.build_net(self.env)
        # trajectory and regression 
        self.expert_action = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])
        self.loss = tf.losses.mean_squared_error(self.expert_action, self.action)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        # start session
        self.sess = tf.Session()
        self.saver=tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        # check memory existance
        my_file = Path("tmp/model.ckpt.index")
        if my_file.is_file():
            self.saver.restore(self.sess,"tmp/model.ckpt")
            print("model restore")
        else:
            print("start new file")
    
    def build_net(self, env):
        act_shape = env.action_space.shape[0]
        obs_shape = env.observation_space.shape[0]
        state = tf.placeholder(tf.float32, [None, obs_shape])
        # network structure
        l1 = tf.layers.dense(inputs = state, units = 128, activation = tf.nn.relu, 
                             kernel_initializer = tf.random_normal_initializer(0.,.1), 
                             bias_initializer = tf.constant_initializer(0.1), name = "policy_l1")
        
        l2 = tf.layers.dense(inputs = l1, units = 64, activation = tf.nn.relu, 
                             kernel_initializer = tf.random_normal_initializer(0.,.1), 
                             bias_initializer = tf.constant_initializer(0.1), name = "policy_l2")
        
        action = tf.layers.dense(inputs = l1, units = act_shape, activation = tf.nn.tanh, 
                             kernel_initializer = tf.random_normal_initializer(0.,.1), 
                             bias_initializer = tf.constant_initializer(0.1), name = "action")

        scale_action = tf.multiply(action, np.array([env.action_space.high.tolist()]), name = "scale_action")
        return state, scale_action
    
    def imitation_learn(self):
        feed_dict = {self.state: self.trajectory, self.expert_action: self.expert_action_history}
        # optimize loss
        for i in range(300):
            self.sess.run(self.optimizer, feed_dict)
            if i % 299 == 0:
                print(self.sess.run(self.loss, feed_dict))
        
        self.save_freq = self.save_freq + 1 
        if self.save_freq % 50 == 1:
            self.saver.save(self.sess, "tmp/model.ckpt")

    def expert_label(self, expert):
        for i in range(len(self.temporary_traj)):
            if len(self.expert_action_history) == 0:
                self.expert_action_history = np.array([expert.act(self.temporary_traj[i,:], self.env).tolist()])
            else:
                self.expert_action_history = np.append(self.expert_action_history,
                                                       np.array([expert.act(self.temporary_traj[i,:], self.env).tolist()]),0)
        # data aggregate
        if len(self.trajectory) == 0:
            self.trajectory = self.temporary_traj
        else:
            self.trajectory = np.append(self.trajectory, self.temporary_traj,0)
        
        self.temporary_traj = [] # temporary memeory clear to zero
    
    def pick_action(self, obs):
        # store temporal obs
        if len(self.temporary_traj) == 0:
            self.temporary_traj = np.array([obs.tolist()])
        else:
            self.temporary_traj = np.append(self.temporary_traj, np.array([obs.tolist()]), 0)
            
        return self.sess.run(self.action, feed_dict={self.state : np.array([obs.tolist()])})[0]       




def imitation():
    # environment and expert set up
    config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        device_count = { "GPU": 0 } )
    sess = tf.InteractiveSession(config=config)
    env = gym.make("RoboschoolAnt-v1")
    pi = ZooPolicyTensorflow("mymodel1", env.observation_space, env.action_space)
    agent = learning_agent(env) 
    for _ in range(80):
        sc = 0
        observation = env.reset()
        done = False
        frame = 0
        restart_delay = 0
        count_epi = 0 

        while done == False:
            action = agent.pick_action(observation) # collect traj and agent current policy
            observation_, reward, done, info = env.step(action)
            sc = sc + reward
            observation = observation_

            count_epi = count_epi + 1
            if count_epi > 200:
                done = True

            # printing and rendering set up
            frame += 1
            still_open = env.render("human")
            if still_open==False:
                return
            if not done: continue              
            if restart_delay==0:
                #print("score=%0.2f in %i frames" % (score, frame))
                if still_open!=True:      # not True in multiplayer or non-Roboschool environment
                    break
                restart_delay = 60*2  # 2 sec at 60 fps
            restart_delay -= 1
            if restart_delay==0: break
        
        print("sum socre: ", sc)
    
        agent.expert_label(pi) # ask expert to label data
        agent.imitation_learn() # imitation learning


if __name__=="__main__":
    imitation()

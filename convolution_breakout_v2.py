# Deep Q network

import gym
import numpy as np
import tensorflow as tf
import math
import random

# HYPERPARMETERS
H = 100
H2 = 100
batch_number = 5
batch_size = 60
gamma = 0.99
explore = 1
num_of_episodes_between_q_copies = 50
learning_rate=1e-3

#These are from the TensorFlow tutorial to initalize weights and variable with a slight
# positive bias to prevent dead neurons
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)    
 
# Convolutional and Pooling functions also from TensorFlow Tutorials 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')
  
    
if __name__ == '__main__':

    env = gym.make('Breakout-v0')
    print "Gym input is ", env.action_space
    print "Gym observation is ", env.observation_space
    #rint(env.observation_space.high)
    #print(env.observation_space.low)
    #print(env.action_space.high)
    #print(env.action_space.low)
    
 
    env.monitor.start('training_dir', force=True)
    #Setup tensorflow
    shape1 = env.observation_space.shape[0]
    shape2 = env.observation_space.shape[1]
    shape3 = env.observation_space.shape[2]
    
    output_shape = 6
    print output_shape

    tf.reset_default_graph()

    #First Q Network
    images = tf.placeholder(tf.float32, [None, shape1, shape2, shape3], name="images") 
    
    W_conv1 = weight_variable([5, 5, 3, 10])
    b_conv1 = bias_variable([10])
    
    #x_image = tf.reshape(images, [-1,shape1,shape2,shape3]) # X Dim, Y Dim, #of Color Channels
    
    h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 10, 10])
    b_conv2 = bias_variable([10])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([300, 150])
    b_fc1 = bias_variable([150])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 300])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32, name="keep")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([150, output_shape])
    b_fc2 = bias_variable([output_shape])

    Q = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    #w1 = tf.Variable(tf.random_uniform([reshape,H], -.10, .10))
    #bias1 = tf.Variable(tf.random_uniform([H], -.10, .10))
    
    #w2 = tf.Variable(tf.random_uniform([H,H2], -.10, .10))
    #bias2 = tf.Variable(tf.random_uniform([H2], -.10, .10))
    
    #w3 = tf.Variable(tf.random_uniform([H2,output_shape], -.10, .10))
    #bias3 = tf.Variable(tf.random_uniform([output_shape], -.10, .10))
    
    #replaced by images
    #states = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], name="states")  # This is the list of matrixes that hold all observations
    #actions = tf.placeholder(tf.float32, [None, env.action_space.n], name="actions")
    
    #hidden_1 = tf.nn.relu(tf.matmul(reshape, w1) + bias1)
    #hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + bias2)
    #action_values = tf.matmul(hidden_2, w3) + bias3
    
    
    #actions = tf.placeholder(tf.int32, [None], name="training_mask")
    #one_hot_actions = tf.one_hot(actions, 3)
    #Q = tf.reduce_sum(tf.mul(action_values, one_hot_actions), reduction_indices=1) 
    
    #previous_action_masks = tf.placeholder(tf.float32, [None, env.action_space.n], name="p_a_m") # This holds all actions taken 
    #previous_values = tf.reduce_sum(tf.mul(action_values, previous_action_masks), reduction_indices=1) #Combination of action taken and resulting q
    
    #Is there a better way to do this?
    
    #images_ = tf.placeholder(tf.float32, [shape1, shape2, shape3]) 
    
    #Second Q Network
    images_ = tf.placeholder(tf.float32, [None, shape1, shape2, shape3], name="images_") 
    
    W_conv1_ = weight_variable([5, 5, 3, 10])
    b_conv1_ = bias_variable([10])
    
    #x_image_ = tf.reshape(images_, [-1,shape1,shape2,shape3]) # X Dim, Y Dim, #of Color Channels
    
    h_conv1_ = tf.nn.relu(conv2d(images_, W_conv1_) + b_conv1_)
    h_pool1_ = max_pool_2x2(h_conv1_)
    
    W_conv2_ = weight_variable([5, 5, 10, 10])
    b_conv2_ = bias_variable([10])

    h_conv2_ = tf.nn.relu(conv2d(h_pool1_, W_conv2_) + b_conv2_)
    h_pool2_ = max_pool_2x2(h_conv2_)
    
    W_fc1_ = weight_variable([300, 150])
    b_fc1_ = bias_variable([150])

    h_pool2_flat_ = tf.reshape(h_pool2_, [-1, 300])
    h_fc1_ = tf.nn.relu(tf.matmul(h_pool2_flat_, W_fc1_) + b_fc1_)
    
    keep_prob_ = tf.placeholder(tf.float32, name="keep_")
    h_fc1_drop_ = tf.nn.dropout(h_fc1_, keep_prob_)
    
    W_fc2_ = weight_variable([150, output_shape])
    b_fc2_ = bias_variable([output_shape])

    Q_ = tf.matmul(h_fc1_drop_, W_fc2_) + b_fc2_
    
    #w1_prime = tf.Variable(tf.random_uniform([convert_conv_to_nn_,H], -1.0, 1.0))
    #bias1_prime = tf.Variable(tf.random_uniform([H], -1.0, 1.0))
    
    #w2_prime = tf.Variable(tf.random_uniform([H,H2], -1.0, 1.0))
    #bias2_prime = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))

    #w3_prime = tf.Variable(tf.random_uniform([H2,output_shape], -1.0, 1.0))
    #bias3_prime = tf.Variable(tf.random_uniform([output_shape], -1.0, 1.0))
    
    #Second Q network
    
    #next_states = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], name="n_s") # This is the list of matrixes that hold all observations
    #hidden_1_prime = tf.nn.relu(tf.matmul(images_, w1_prime) + bias1_prime)
    #hidden_2_prime = tf.nn.relu(tf.matmul(hidden_1_prime, w2_prime) + bias2_prime)
    #next_action_values =  tf.matmul(hidden_2_prime, w3_prime) + bias3_prime
    #next_values = tf.reduce_max(next_action_values, reduction_indices=1)   
    
    #need to run these to assign weights from Q to Q_prime
    
    
    W_conv1_update = W_conv1_.assign(W_conv1)
    b_conv1_update = b_conv1_.assign(b_conv1)
    
    W_conv2_update = W_conv2_.assign(W_conv2)
    b_conv2_update = b_conv2_.assign(b_conv2)
    
    W_fc1_update =  W_fc1_.assign(W_fc1)
    b_fc1_update =  b_fc1_.assign(b_fc1)

    W_fc2_update = W_fc2_.assign(W_fc2)
    b_fc2_update = b_fc2_.assign(b_fc2)
  
    assign_all = [ 
        W_conv1_update,
        b_conv1_update,
        W_conv2_update,
        b_conv2_update,
        W_fc1_update,
        b_fc1_update,
        W_fc2_update,
        b_fc2_update]
        
    #kernel_update = kernal_.assign(kernal)
    #w1_prime_update= w1_prime.assign(w1)
    #bias1_prime_update= bias1_prime.assign(bias1)
    #w2_prime_update= w2_prime.assign(w2)
    #bias2_prime_update= bias2_prime.assign(bias2)
    #w3_prime_update= w3_prime.assign(w3)
    #bias3_prime_update= bias3_prime.assign(bias3)
    
    #Q_prime = rewards + gamma * tf.reduce_max(next_action_values, reduction_indices=1)
    
    #we need to train Q

    rewards = tf.placeholder(tf.float32, [None, ], name="rewards") # This holds all the rewards that are real/enhanced with Qprime
    #loss = (tf.reduce_mean(rewards - tf.reduce_mean(action_values, reduction_indices=1))) * one_hot
    actions = tf.placeholder(tf.int32, [None], name="training_mask")
    one_hot_actions = tf.one_hot(actions, 6)
    Q_filtered = tf.reduce_sum(tf.mul(Q, one_hot_actions), reduction_indices=1)
    loss = tf.reduce_sum(tf.square(rewards - Q_filtered)) #* one_hot  
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
    
    #Setting up the enviroment
    
    max_episodes = 20000
    max_steps = 10000

    D = []
    explore = 1.0
    
    rewardList = []
    past_actions = []
    
    episode_number = 0
    episode_reward = 0
    reward_sum = 0
    
    init = tf.initialize_all_variables()
   
    with tf.Session() as sess:
        sess.run(init)
        #Copy Q over to Q_prime
        sess.run(assign_all)
    
        for episode in xrange(max_episodes):
            print 'Reward for episode %f is %f. Explore is %f' %(episode,reward_sum, explore)
            reward_sum = 0
            new_state = env.reset()
            
            for step in xrange(max_steps):
                #if(step == (max_steps-1)):
                #    print 'Made 199 steps!'
                
                if episode % batch_number == 0:
                    env.render()
                
                state = list(new_state);
                
                if explore > random.random():
                    action = env.action_space.sample()
                    #action = np.argmax(action_sample)
                    #print action
                    #if(action == 1.0):
                    #    curr_action = [0.0,1.0] 
                    #else:
                    #    curr_action = [1.0,0.0]
                else:
                
                    #get action from policy
                    results = sess.run(Q, feed_dict={images: np.array([new_state]), keep_prob : 1.0})
                    #print "result", results
                    action = (np.argmax(results[0]))
                    #print action
                    #if(action == 1.0):
                    #    curr_action = [0.0,1.0]
                    #else:
                    #    cur_action = [1.0,0.0]
                    
                curr_action = action;
                
                #action_temp = [-1.0,1.0,0.0]
                #action_temp[action] = 1.0
                new_state, reward, done, _ = env.step(action)
                reward_sum += reward
                
                
                #print "D Before", D
                D.append([state, curr_action, reward, new_state, done])
                #print "D After", D
                
                #if done:
                #   # step through and increment until done
                #    help_factor = -10
                #    for x in reversed(xrange(len(D))):
                #        if x != (len(D)-1):
                #            if step == max_steps-1:
                #                help_factor = 20
                #                #print "made 200 steps @ reward"
                #            #print x
                #            (D[x])[2] += help_factor 
                #            help_factor += 1;
                #            #print x
                #            #print (D[x])[2]
                #            if (D[x])[4]:
                #                #print x
                #                break
                #    #print D
                
                
                if len(D) > 10000:
                    D.pop(0)
                #Training a Batch
                #samples = D.sample(50)
                sample_size = len(D)
                if sample_size > 20:
                    sample_size = 20
                else:
                    sample_size = sample_size
                #print "Sample Size", sample_size
                if sample_size > 0:
                    samples = [ D[i] for i in random.sample(xrange(len(D)), sample_size) ]
                    #print "Samples", len(samples)
                    new_states_for_q = [ x[3] for x in samples]
                    #print "new_states_for_q", new_states_for_q
                    all_q_prime = sess.run(Q_, feed_dict={images_: new_states_for_q, keep_prob_ : .5})
                    #print "All The Q Primes:", all_q_prime.shape[0], all_q_prime.shape[1]
                    y_ = []
                    states_samples = []
                    next_states_samples = []
                    actions_samples = []
                    for ind, i_sample in enumerate(samples):
                        #print i_sample
                        if i_sample[4] == True:
                            #print i_sample[2]
                            y_.append(reward)
                            #print y_
                        else:
                            this_q_prime = all_q_prime[ind]
                            #print this_q_prime
                            maxq = max(this_q_prime)
                            #print maxq
                            y_.append(reward + (gamma * maxq))
                            #print y_
                        #y_.append(i_sample[2])
                        states_samples.append(i_sample[0])
                        next_states_samples.append(i_sample[3])
                        actions_samples.append(i_sample[1])
                    #print actions_samples
                    #print sess.run(loss, feed_dict={states: states_samples, next_states: next_states_samples, rewards: y_, actions: actions_samples, one_hot: actions_samples})
                    #print sample_size
                    sess.run(train, feed_dict={images: states_samples, rewards: y_, keep_prob : .7, actions: actions_samples})
                        #y_ = reward + gamma * sess.run(next_action_values, feed_dict={next_states: np.array([i_sample[3]])})
                    #y_ = curr_action * np.vstack([y_])
                    #print y_
                    #y_ = y_
                    #print y_
                    #sess.run(train, feed_dict={states: np.array([i_sample[0]]), next_states: np.array([i_sample[3]]), rewards: y_, actions: np.array([i_sample[1]]), one_hot: np.array([curr_action])})
                    
                if done:
                    break
                        
            if episode % num_of_episodes_between_q_copies == 0:
                sess.run(assign_all)
            
            explore = explore * .99
            #if explore < .1:
            #    explore = .1

    env.monitor.close()
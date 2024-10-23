# Import Required Libraries
import numpy as np
import tensorflow as tf
import gym

# Define the DQN Model
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(24, activation='relu')
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# CartPole has 2 possible actions: push left or push right
num_actions = 2
dqn_agent = DQN(num_actions)

# Define the DQN Algorithm Parameters
learning_rate = 0.001
discount_factor = 0.99
# Initial exploration probability
exploration_prob = 1.0
# Decay rate of exploration probability
exploration_decay = 0.995
# Minimum exploration probability
min_exploration_prob = 0.1

# Initialize the CartPole Environment
env = gym.make('CartPole-v1')
state_shape = env.observation_space.shape[0]

# Define the Loss Function and Optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Training the DQN
num_episodes = 1000
max_steps_per_episode = 500

for episode in range(num_episodes):
    state = env.reset()
    state = np.array(state).reshape(1, -1)  # Reshape state to match the network's input shape
    episode_reward = 0

    for step in range(max_steps_per_episode):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()  # Explore randomly
        else:
            q_values = dqn_agent(state)
            action = np.argmax(q_values)

        next_state, reward, done, _ = env.step(action)
        next_state = np.array(next_state).reshape(1, -1)  # Reshape next_state as well

        # Update the Q-values using Bellman equation
        with tf.GradientTape() as tape:
            current_q_values = dqn_agent(state)
            next_q_values = dqn_agent(next_state)
            max_next_q = tf.reduce_max(next_q_values, axis=-1)
            
            # Update the action's Q-value using the Bellman equation
            target_q_values = tf.identity(current_q_values)
            target_q_values = tf.tensor_scatter_nd_update(
                target_q_values,
                indices=[[0, action]],  # Update the action's Q-value
                updates=[reward + discount_factor * max_next_q * (1 - done)]
            )
            loss = loss_fn(current_q_values, target_q_values)

        # Apply the gradients
        gradients = tape.gradient(loss, dqn_agent.trainable_variables)
        optimizer.apply_gradients(zip(gradients, dqn_agent.trainable_variables))

        state = next_state
        episode_reward += reward

        if done:
            break

    # Decay exploration probability
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)
    
    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

# Evaluating the Trained DQN
num_eval_episodes = 10
eval_rewards = []

for _ in range(num_eval_episodes):
    state = env.reset()
    state = np.array(state).reshape(1, -1)  # Reshape state for evaluation
    eval_reward = 0

    for _ in range(max_steps_per_episode):
        q_values = dqn_agent(state)
        action = np.argmax(q_values)
        next_state, reward, done, _ = env.step(action)
        next_state = np.array(next_state).reshape(1, -1)
        eval_reward += reward
        state = next_state

        if done:
            break

    eval_rewards.append(eval_reward)

average_eval_reward = np.mean(eval_rewards)
print(f"Average Evaluation Reward: {average_eval_reward}")

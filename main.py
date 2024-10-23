# Import Required Libraries
import numpy as np
import tensorflow as tf
import gym
import random

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

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Update Target Network
def update_target_network(main_network, target_network):
    target_network.set_weights(main_network.get_weights())

# DQN Training Function
def train_dqn(env, num_episodes=500, max_steps_per_episode=100, gamma=0.99, 
              learning_rate=0.001, batch_size=32, epsilon_decay=0.995, min_epsilon=0.01):
    
    # Initialize main and target networks
    num_actions = env.action_space.n
    main_dqn = DQN(num_actions)
    target_dqn = DQN(num_actions)
    update_target_network(main_dqn, target_dqn)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    replay_buffer = ReplayBuffer()
    
    epsilon = 1.0  # Initial exploration probability
    rewards_per_episode = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state).reshape(1, -1)
        episode_reward = 0

        for step in range(max_steps_per_episode):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                q_values = main_dqn(state)
                action = np.argmax(q_values)

            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state).reshape(1, -1)
            episode_reward += reward

            replay_buffer.add((state, action, reward, next_state, done))

            state = next_state

            if done:
                break

            # Sample random minibatch from replay buffer
            if len(replay_buffer.buffer) >= batch_size:
                minibatch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                states = np.concatenate(states)
                next_states = np.concatenate(next_states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                dones = np.array(dones)

                # Predict Q-values for next states
                next_q_values = target_dqn(next_states)
                max_next_q_values = np.max(next_q_values, axis=1)

                # Target Q-values using the Bellman equation
                target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

                with tf.GradientTape() as tape:
                    current_q_values = main_dqn(states)
                    indices = np.array([[i, actions[i]] for i in range(batch_size)])
                    chosen_q_values = tf.gather_nd(current_q_values, indices)

                    loss = loss_fn(target_q_values, chosen_q_values)

                gradients = tape.gradient(loss, main_dqn.trainable_variables)
                optimizer.apply_gradients(zip(gradients, main_dqn.trainable_variables))

        # Update exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Update target network periodically
        if episode % 10 == 0:
            update_target_network(main_dqn, target_dqn)

        rewards_per_episode.append(episode_reward)

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_per_episode[-50:])
            print(f"Episode {episode + 1}: Avg Reward (last 50) = {avg_reward}")

    return rewards_per_episode

# Main function to run the demo
if __name__ == "__main__":
    # Initialize environment
    env = gym.make("CartPole-v1")

    # Train the DQN
    rewards = train_dqn(env)

    # Close environment
    env.close()

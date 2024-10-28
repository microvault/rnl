import csv


class ScaleDataset:
    def __init__(self):
        # Initialize lists to store episode data
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

    def store_step(self, state, action, reward, next_state):
        """
        Store data for a single timestep.

        Parameters:
        - state: The current state.
        - action: The action taken.
        - reward: The reward received.
        - next_state: The next state after the action.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def save_dataset(self):
        """
        Save the episode data to a CSV file.
        Resets the episode data after saving or discarding.
        """
        with open("dataset.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header if file is empty
            if csvfile.tell() == 0:
                writer.writerow(["State", "Action", "Reward", "Next State"])
            for state, action, reward, next_state in zip(
                self.states, self.actions, self.rewards, self.next_states
            ):
                writer.writerow([state, action, reward, next_state])
        # Reset episode data
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.cumulative_reward = 0

import numpy as np
import itertools

class DirichletInfiniteAgent:
    def __init__(self, num_agents, S, A, T, p, trans_p, reward):
        """
        S: state space
        A: action space
        s_1: starting state
        T: time horizon
        p: parameter between (0, 1]
        """
        self.num_agents = num_agents
        self.S = S
        self.A = A
        self.T = T
        self.phi = int(np.floor(S * np.log(S * A / p)))
        self.w = np.log(T / p)
        self.k = np.log(T / p)
        # self.M = np.full([S, A, S, num_agents], self.w)
        self.M = np.zeros([S, A, S, num_agents])
        for i in range(num_agents):
            for s, a in itertools.product(range(S), range(A)):
                self.M[s, a, :, i] = np.random.dirichlet(np.ones((S)))

        self.eta = np.sqrt(T * S / A) + 12 * self.w * S ** 4
        self.trans_p = trans_p
        self.reward = reward

    def posterior_sample(self, transition_prob, M, S, A):
        dirichlet_trans_p = np.zeros(transition_prob.shape)
        for s, a in itertools.product(range(S), range(A)):
            dirichlet_trans_p[s, a] = np.random.dirichlet(M[s, a, :])
        return dirichlet_trans_p

    def compute_policy(self, trans_prob, S, A, phi, reward):
        # performs undiscounted value iteration to output an optimal policy
        value_func = np.zeros(S)
        policy = np.zeros(S)
        iter = 0
        gamma = 0.99
        while True:
            tolerance = 0.01
            diff = np.zeros(S)
            for s in range(S):
                value = value_func[s]
                action_returns = []
                for a in range(A):
                    action_return = np.sum(
                        [trans_prob[s, a, s_next] * (reward[s, a, s_next] + gamma * value_func[s_next]) for s_next in range(S)])  # computes the undiscounted returns
                    action_returns.append(action_return)
                value_func[s] = np.max(action_returns)
                policy[s] = np.argmax(action_returns)
                diff[s] = max(diff[s], np.abs(value - value_func[s]))
            iter += 1
            if iter % 10000 == 0:
                print("diff: ", diff)
            if np.max(diff) <= tolerance:
                # print(value_func)
                break

        # after value iteration output a deterministic policy
        # policy = np.zeros(S)
        # for s in range(S):
        #     action_returns = []
        #     for a in range(A):
        #         action_return = np.sum([trans_prob[s, a, s_next] * (reward[s, a, s_next] + value_func[s_next]) for s_next in range(S)])  # computes the undiscounted returns
        #         action_returns.append(action_return)
        #     policy[s] = np.argmax(action_returns) ##doubled actions
        return policy

    def train(self, epochs, s_t):
        phi = self.phi
        w = self.w
        k = self.k
        M = self.M
        T = self.T

        num_visits = np.zeros((self.S, self.A, self.S, self.num_agents))
        curr_states = np.zeros((self.num_agents), dtype=np.int)
        for i in range(len(curr_states)):
            curr_states[i] = int(s_t)
        t = 0

        cumulative_reward = np.zeros(self.num_agents)
        max_reward = np.zeros(self.num_agents)
        for i in range(epochs):
            num_visits = np.sum(num_visits[:, :, :, :], axis=-1)  # TODO: check if sum is correct
            num_visits = np.expand_dims(num_visits, axis=-1).repeat(repeats=self.num_agents, axis=-1)

            policies = []
            for agent in range(self.num_agents):
                trans_prob = self.posterior_sample(self.trans_p, M[:, :, :, agent], self.S, self.A,)
                policy = self.compute_policy(trans_prob, self.S, self.A, phi, self.reward)  # computes the max gain policy
                policies += [policy]

            num_visits_next = np.copy(num_visits)
            while True:
                end_epoch = False
                for agent in range(self.num_agents):
                    s_t = curr_states[agent]
                    a_t = int(policies[agent][s_t])
                    s_next = np.random.choice(range(0,self.S), size=1, p=self.trans_p[s_t, a_t, :])
                    # s_next = np.argmax(self.trans_p[s_t, a_t, :]) #should be np.choice
                    cumulative_reward[agent] += reward[s_t, a_t, s_next]
                    max_reward[agent] += np.amax(reward[s_t, :, :])  # should do expectation of max reward over transition probability?
                    num_visits_next[s_t, a_t, s_next, agent] += 1

                    if np.sum(num_visits_next[s_t, a_t, :, agent]) >= 2 * np.sum(num_visits[s_t, a_t, :, agent]):
                        end_epoch = True
                    curr_states[agent] = s_next

                t += 1
                # if t % 500 == 0:
                #     regret = np.mean(max_reward - cumulative_reward) / t
                #     print("process at timestep", str(t) + ": " + str(regret))
                if t == T:
                    regret = np.mean(max_reward - cumulative_reward) / t
                    print("regret at time step", str(t) + ": " + str(regret))
                    return regret
                    break

                if end_epoch:
                    M = np.maximum(np.ones(num_visits_next.shape), num_visits_next)
                    num_visits = num_visits_next
                    break
            if t == T:
                break


if __name__ == "__main__":
    #Define MDP
    state = 5
    action = 5
    seeds = [105, 106]
    for seed in seeds:
        np.random.seed(seed)
        reward = np.random.normal(0.0, 1.0, size=(state, action, state))
        trans_p = np.zeros([state, action, state])
        for i in range(state):
            for j in range(action):
                sample = np.random.gamma(1, 1, state)
                trans_p[i, j, :] = sample / np.sum(sample)
    #end Define MDP

        total_regret = []

        #TODO: add agents to the list
        num_agents = [1, 2]
        for i in num_agents:
            print("agents: ", i)
            psrl = DirichletInfiniteAgent(i, state, action, 20000, 0.75, trans_p, reward)
            regret = psrl.train(1000, int(np.random.randint(0, state, 1)))
            total_regret += [regret]

        np.savetxt("result" + str(seed) + ".csv", np.column_stack((num_agents, total_regret)), delimiter=",")
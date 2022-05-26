import numpy as np
import random
import itertools
import torch

_device_ddtype_tensor_map = {
    'cpu': {
        torch.float32: torch.FloatTensor,
        torch.float64: torch.DoubleTensor,
        torch.float16: torch.HalfTensor,
        torch.uint8: torch.ByteTensor,
        torch.int8: torch.CharTensor,
        torch.int16: torch.ShortTensor,
        torch.int32: torch.IntTensor,
        torch.int64: torch.LongTensor,
        torch.bool: torch.BoolTensor,
    },
    'cuda': {
        torch.float32: torch.cuda.FloatTensor,
        torch.float64: torch.cuda.DoubleTensor,
        torch.float16: torch.cuda.HalfTensor,
        torch.uint8: torch.cuda.ByteTensor,
        torch.int8: torch.cuda.CharTensor,
        torch.int16: torch.cuda.ShortTensor,
        torch.int32: torch.cuda.IntTensor,
        torch.int64: torch.cuda.LongTensor,
        torch.bool: torch.cuda.BoolTensor,
    }
}


def policy_from_mdp(trans_prob, reward, S, A, tol=0.01):
    # performs undiscounted value iteration to output an optimal policy
    value_func = torch.zeros(S)
    policy = torch.zeros(S)
    iter = 0
    gamma = 0.99
    while True:
        diff = torch.zeros(S)
        value = value_func
        action_return = reward + gamma * torch.einsum('ijk,k->ij', trans_prob, value_func)  # [S, A]
        value_func, policy = torch.max(action_return, dim=-1)  # [S]
        diff = torch.maximum(diff, torch.abs(value - value_func))  # [S]

        if torch.any(diff.max() <= tol) or iter >= 10000:
            break

    return policy  # [S]


class DirichletFiniteAgent:
    def __init__(self, num_agents, num_envs, S, A, trans_p, rewards, optimal_policy):
        """
        num_agents: number of agents
        num_envs: number of envs
        S: size of the state space
        A: size of the action space
        trans_p: transitions probabilities, size [n_env, n_agents, n_S, n_A, n_S]
        rewards: rewards, size [n_envs, n_agents, n_S, n_A]
        optimal_policy: optimal policies for all envs, size [n_envs, n_S]
        """
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.S = S
        self.A = A
        self.trans_p = trans_p  # [n_envs, n_agents, n_S, n_A, n_S]
        self.optimal_trans_p = trans_p[:, 0, ...]  # [n_envs, n_S, n_A, n_S]
        self.rewards = rewards  # [n_envs, n_agents, n_S, n_A]
        self.optimal_rewards = rewards[:, 0, ...]  # [n_envs, n_S, n_A]
        self.optimal_policy = optimal_policy  # [n_envs, n_S]
        # concentration parameters of Dirichlet posterior of transition_p
        self.alpha = torch.ones(num_envs, S, A, S)  
        self.reward_mean = torch.zeros(num_envs, S, A) 
        self.reward_scale = torch.ones(num_envs, S, A)

    def posterior_sample(self, alpha, mu, scale, n_sample):
        dist_trans_p = torch.distributions.dirichlet.Dirichlet(alpha)
        dist_reward = torch.distributions.normal.Normal(mu, scale)
        trans_p = dist_trans_p.sample([n_sample])
        rewards = dist_reward.sample([n_sample])
        return torch.transpose(trans_p, 0, 1), torch.transpose(rewards, 0, 1)

    def train(self, episodes, horizon):
        cum_regret = torch.zeros(self.num_envs, self.num_agents)
        num_visits = torch.zeros((self.num_envs, self.S, self.A, self.S))
        model_reward = torch.zeros((self.num_envs, self.S, self.A))

        for i in range(episodes):
            # initialize state tracking for each agent in each env
            curr_states = torch.randint(0, self.S, size=[self.num_envs, self.num_agents])
            # initialize state tracking for the optimal agent in each env
            curr_optimal_state = torch.randint(0, self.S, size=[self.num_envs])

            # (Alg) sample MDP's from the posterior 
            # [n_envs, n_agents, n_S, n_A, n_S], [n_envs, n_agents, n_S, n_A]
            sampled_trans_p, sampled_reawrds = self.posterior_sample(
                self.alpha, self.reward_mean, self.reward_scale, self.num_agents)  

            # extract optimal policies from sampled MDP's: [n_envs, n_agents, n_S]
            policy = torch.zeros((self.num_envs, self.num_agents, self.S), dtype=torch.int64)
            for env in range(self.num_envs):
                for agent in range(self.num_agents):
                    policy[env, agent, :] = policy_from_mdp(
                        sampled_trans_p[env, agent], sampled_reawrds[env, agent], self.S, self.A)

            for _ in range(horizon):
                # agents rollout
                s_t = curr_states  # [n_envs, n_agents]
                a_t = policy[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1),
                    torch.arange(self.num_agents).unsqueeze(0).unsqueeze(-1),
                    s_t.unsqueeze(-1)].squeeze()  # [n_envs, n_agents]
                trans_p = self.trans_p[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 
                    torch.arange(self.num_agents).unsqueeze(0).unsqueeze(-1).unsqueeze(-1), 
                    s_t.unsqueeze(-1).unsqueeze(-1), 
                    a_t.unsqueeze(-1).unsqueeze(-1)].squeeze()  # [n_envs, n_agents, n_S]
                # [n_envs, n_agents]
                s_next = torch.distributions.categorical.Categorical(trans_p).sample()
                curr_states = s_next

                # optimal agent rollout
                optimal_s_t = curr_optimal_state
                optimal_a_t = self.optimal_policy[
                    torch.arange(self.num_envs).unsqueeze(-1),
                    optimal_s_t.unsqueeze(-1)].squeeze()  # [n_envs]
                optimal_trans_p = self.optimal_trans_p[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1), 
                    optimal_s_t.unsqueeze(-1).unsqueeze(-1), 
                    optimal_a_t.unsqueeze(-1).unsqueeze(-1)].squeeze()  # [n_envs, n_S]
                optimal_s_next = torch.distributions.categorical.Categorical(
                    optimal_trans_p).sample()  # [n_envs]
                curr_optimal_state = optimal_s_next

                # collect rewards and update cum_regret
                reward = self.rewards[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 
                    torch.arange(self.num_agents).unsqueeze(0).unsqueeze(-1).unsqueeze(-1), 
                    s_t.unsqueeze(-1).unsqueeze(-1), 
                    a_t.unsqueeze(-1).unsqueeze(-1)].squeeze()  # [n_envs, n_agents]
                optimal_reward = self.optimal_rewards[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1), 
                    optimal_s_t.unsqueeze(-1).unsqueeze(-1), 
                    optimal_a_t.unsqueeze(-1).unsqueeze(-1)].squeeze()  # [n_envs]

                cum_regret += optimal_reward.unsqueeze(-1) - reward

                # record observed transitions and rewards
                for agent in range(self.num_agents):
                    num_visits[
                        torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                        s_t[:, agent].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                        a_t[:, agent].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                        s_next[:, agent].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)] += 1
                    model_reward[
                        torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1),
                        s_t[:, agent].unsqueeze(-1).unsqueeze(-1),
                        a_t[:, agent].unsqueeze(-1).unsqueeze(-1)] \
                            += reward[:, agent].unsqueeze(-1).unsqueeze(-1)

            # update the posterior of the Dirichlet alpha of transitions
            self.alpha = torch.ones(self.alpha.shape) + num_visits
            # update the posterior of the Gaussian of rewards, [n_envs, n_S, n_A]
            count = torch.ones(self.num_envs, self.S, self.A) + torch.sum(num_visits, dim=-1)
            self.reward_mean = model_reward / count
            self.reward_scale = 1 / torch.sqrt(count)

        # evaluate episodic regret
        per_step_regret = cum_regret / (episodes * horizon)  # [n_envs, n_agents]
        per_step_per_agent_regret = per_step_regret.mean(dim=-1)  # [n_envs]
        per_step_per_agent_Bayesian_regret = per_step_per_agent_regret.mean()
        print("bayesian regret: ", per_step_per_agent_Bayesian_regret)

        return per_step_per_agent_Bayesian_regret



if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type(
            _device_ddtype_tensor_map['cuda'][torch.get_default_dtype()])
        torch.cuda.set_device(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True

    num_states = 20
    num_actions = 10
    num_envs = 10
    num_episodes = 30
    horizon = 75
    # seeds = range(100, 101)
    seeds = (100,)
    for seed in seeds:
        # deterministic settings for current seed
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print("seed: ", seed)

        # define MDP
        all_env_rewards = torch.abs(torch.randn(num_envs, num_states, num_actions))
        dist_trans_p = torch.distributions.dirichlet.Dirichlet(torch.ones(num_states))
        all_env_trans_p = dist_trans_p.sample([num_envs, num_states, num_actions])

        # compute optimal policy for each env
        all_env_optimal_policy = torch.zeros((num_envs, num_states), dtype=torch.int64)
        for env in range(num_envs):
            all_env_optimal_policy[env] = policy_from_mdp(
                all_env_trans_p[env], all_env_rewards[env], num_states, num_actions, tol=0.001)

        regrets = []

        # TODO: there is a bug for num_agents = 1 
        list_num_agents = [2, 4, 10, 20, 30, 40, 50, 60, 70] #, 80, 90, 100]
        # list_num_agents = [80, 90, 100]
        for num_agents in list_num_agents:
            print("num of agents: ", num_agents)
            # [n_envs, n_agents, n_S, n_A, n_S]
            all_env_agent_rewards = all_env_rewards.unsqueeze(1).repeat(1, num_agents, 1, 1)
            all_env_agent_trans_p = all_env_trans_p.unsqueeze(1).repeat(1, num_agents, 1, 1, 1)
            psrl = DirichletFiniteAgent(
                num_agents, num_envs, num_states, num_actions, 
                all_env_agent_trans_p, all_env_agent_rewards, all_env_optimal_policy)
            regret = psrl.train(num_episodes, horizon)
            regrets.append(regret)

        total_regret = torch.stack(regrets)
        total_regret_np = total_regret.cpu().detach().numpy()

        np.savetxt("results/finite_" + str(seed) + "_agents_" + str(list_num_agents[-1]) + ".csv", np.column_stack((list_num_agents, total_regret_np)), delimiter=",")


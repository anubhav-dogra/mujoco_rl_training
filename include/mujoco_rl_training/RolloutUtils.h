#pragma once

namespace mujoco_rl_training {

template <typename Env, typename Policy>
double evaluate_episode_return(Env& env, const Policy& policy) {
    auto observation = env.reset();
    double total_reward = 0.0;

    while (true) {
        const auto action = policy.action_from_obs(observation);
        const auto result = env.step(action);
        observation = result.observation;
        total_reward += result.reward;

        if (result.truncated || result.terminated) {
            break;
        }
    }

    return total_reward;
}

template <typename Env, typename Policy>
double evaluate_average_return(Env& env, const Policy& policy, int num_episodes) {
    double total_return = 0.0;
    for (int episode = 0; episode < num_episodes; ++episode) {
        total_return += evaluate_episode_return(env, policy);
    }
    return total_return / static_cast<double>(num_episodes);
}

}  // namespace mujoco_rl_training

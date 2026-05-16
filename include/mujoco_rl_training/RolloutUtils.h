#pragma once

namespace mujoco_rl_training {

template <typename Env, typename Policy, typename ActionAdaptor>
double evaluate_episode_return(Env& env, const Policy& policy, ActionAdaptor action_adaptor) {
    auto observation = env.reset();
    double total_reward = 0.0;

    while (true) {
        const auto action = action_adaptor(policy, observation, env);
        const auto result = env.step(action);
        observation = result.observation;
        total_reward += result.reward;

        if (result.truncated || result.terminated) {
            break;
        }
    }

    return total_reward;
}

template <typename Env, typename Policy, typename ActionAdaptor>
double evaluate_average_return(Env& env, const Policy& policy, int num_episodes, ActionAdaptor action_adaptor) {
    double total_return = 0.0;
    for (int episode = 0; episode < num_episodes; ++episode) {
        total_return += evaluate_episode_return(env, policy, action_adaptor);
    }
    return total_return / static_cast<double>(num_episodes);
}

}  // namespace mujoco_rl_training

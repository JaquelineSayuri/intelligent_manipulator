import numpy as np
from sac_tf2 import Agent
import reach_env
from graphs import get_graphs, save_results, open_file
from random import choice
from math import degrees

def HER(encountered_goals):
    transition_indexes = range(len(agent.memory.episode_transitions))
    goal_index, new_goal = sample_goal(transition_indexes)
    try:
        for n in transition_indexes:
            transition_index, sampled_transition = sample_transition(transition_indexes, goal_index)
            state, action, reward, next_state, done = sampled_transition
            state = np.concatenate([state[:3], new_goal])
            next_state = np.concatenate([next_state[:3], new_goal])
            reward = env.reward(state[:3], new_goal)
            done = env.is_done(next_state[:3], transition_index, new_goal)
            if done and transition_index != env.max_steps:
                encountered_goals += 1
            agent.remember(state, action, reward, next_state, done)
    except:
        pass
        
    return encountered_goals

def sample_goal(transition_indexes):
    sampled_transition_index, sampled_transition = sample_transition(transition_indexes, len(transition_indexes))
    new_goal = sampled_transition[0][0:3]
    return sampled_transition_index, new_goal

def sample_transition(transition_indexes, max_index):
    sampled_transition_index = choice(transition_indexes[:max_index])
    sampled_transition = agent.memory.episode_transitions[sampled_transition_index]
    return sampled_transition_index, agent.memory.episode_transitions[sampled_transition_index]

def Print(step, observation, action, reward, observation_, done):
    print(f'Step: {step:<3d} ', end = '')
    print('  State: ', end = '')
    for o in observation:
        print(f'{o:<6.2f} ', end = '')
    print('  Action: ', end = '')
    for a in action:
        print(f'{a:<8.4f} ', end = '')
    print(f'  Reward: {reward:<7.2f} ', end = '')
    print('  State_: ', end = '')
    for o in observation_:
        print(f'{o:<6.2f} ', end = '')
    print(f'  Done: {done}')


if __name__ == '__main__':
    n_games = 12000
    T = 30
    seeds = 2
    initial_seed = 1
    filename = 'open_manipulator_x.png'
    figure_file = 'plots/' + filename
    load_checkpoint = False
    saving_frequency = 50

    for s in range(initial_seed,seeds):
        env = reach_env.env()
        agent = Agent(input_dims=[env.num_states],
                        env=env,
                        n_actions=env.num_actions)
        score_history = []

        if load_checkpoint:
            agent.load_models()
            score_history = open_file(f'results/total_score_per_episode_{s}.txt')
            score_history = score_history[-T:]
            agent.memory.load_buffer(s)

        for i in range(n_games):
            observation = env.reset()
            done = False
            score = 0
            step = 0
            path = [list(env.target_position),
                    list(env.manipulator.position)]
            encountered_goals = 0
            while not done:
                step += 1
                action = agent.choose_action(observation)
                observation_, reward, done = env.step(action)
                position = env.manipulator.position
                path.append(list(position))
                score += reward
                agent.memory.episode_transitions.append([observation, action, reward, observation_, done])
                agent.remember(observation, action, reward, observation_, done)
                Print(step, observation, action, reward, observation_, done)
                observation = observation_
                agent.learn()

            if step == env.max_steps:
                encountered_goals = HER(encountered_goals)
            agent.memory.episode_transitions = []

            score_history.append(score)
            if len(score_history) > T:
                score_history.pop(0)
            avg_score = np.mean(score_history)
            save_results(f'results/steps_per_episode_{s}.txt', step)
            save_results(f'results/total_score_per_episode_{s}.txt', score)
            save_results(f'results/paths_{s}.txt', path)
            save_results(f'results/avg_scores_{s}.txt', avg_score)

            if not i%saving_frequency:
                agent.save_models()
                get_graphs(env.max_steps, s)
                agent.memory.save_buffer(s)
            
            print('episode ', i, 'steps ', step, 'score %.0f' % score, 'encountered goals', encountered_goals, 'seed', s)
            print()

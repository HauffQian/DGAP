import copy
import glob
import os, sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import re
import pdb
import pickle
import json
import random
from copy import deepcopy

from utils_bc import utils_interactive_eval
from utils_bc.utils_graph import filter_redundant_nodes
from envs.utils.check_logical import check_env_bug
from gpt_policy import GPTPolicy, split_goal
from sim_compute import Similarity
from memory_graph import MemoryGraph
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM , AutoModelForSequenceClassification



def load_model():
    roberta_model = AutoModelForSequenceClassification.from_pretrained("/home/qhf/Virualhome/roberta/checkpoint-6100")
    roberta_tokenizer  = AutoTokenizer.from_pretrained("/home/qhf/Virualhome/roberta/checkpoint-6100")
    return roberta_model, roberta_tokenizer


def calsco(roberta_model, roberta_tokenizer, task_goal , recent_action, agent_action):
    des = task_goal + ". "
    for action in recent_action:
        action = re.sub(r'\(\d+\)', '(1)', action)
        des += action
        des += ", "
    des = des[:-2] + ". Action: " + agent_action
    inputs = roberta_tokenizer([des, ], return_tensors='pt')
    score = roberta_model(**inputs).logits[0][0].item()
    score = round(score)
    return score


def score_search(roberta_model, roberta_tokenizer, lid_goals, recent_action, curr_goal, env_graph):
    

    #---------------No Need to concern the goal condition!!!!!--------------
    # for k, v in lid_goals.items():
    #     if v > 0:
    #     # 获取相关obj
    #         name = str(k.split('_')[-2])
    #         for node in env_graph['nodes']:
    #             if node['class_name'] == name:
    #                 goal_objs.append((node['id'], name))
            
    #         obj_id = int(k.split('_')[-1])
    #         obj_name = [node['class_name'] for node in env_graph['nodes'] if node['id'] == obj_id][0]
    #         # 判断当前obj是否在goal中已存在避免重复存储
    #         have_exist_in_goal = False
    #         for id, name in goal_objs:
    #             if id == obj_id:
    #                 have_exist_in_goal = True
    #         if not have_exist_in_goal:
    #             goal_objs.append((obj_id, obj_name))
    #         # 判断当前goal_obj在task_obj中是否已存储
    test_objs = [{"name": node['class_name'], "id": node['id']} for node in env_graph['nodes']]
    actions = ['walk', 'find', 'open', 'grab', 'close', 'switchon']
    exec_action_lists = []
    for obj in test_objs:
        for action in actions:
            if action == 'find':  # 将 'find' 动作转换为 'walk'
                action = 'walk'
            action_script = "[{}] <{}> ({})".format(action, obj['name'], obj['id'])
            exec_action_lists.append(action_script)
    actions = ['putback', 'putin']
    for i in range(len(test_objs)):
        for j in range(len(test_objs)):
            if i != j:  # 确保不是同一个物体
                for action in actions:
                    item1 = test_objs[i]
                    item2 = test_objs[j]
                    action_script = "[{}] <{}> ({}) <{}> ({})".format(
                        action, item1['name'], item1['id'], item2['name'], item2['id'])
                    exec_action_lists.append(action_script)
    mem_graph = MemoryGraph(None)
    mem_graph.set_graph(env_graph)
    
    
                    
    des = curr_goal + ". "
    for action in recent_action:
        action = re.sub(r'\(\d+\)', '(1)', action)
        des += action
        des += ", "
    des = des[:-2] + ". Action: " 
    action_scores = []


    for i in range(len(exec_action_lists)):

        inputs = roberta_tokenizer([des + exec_action_lists[i]], return_tensors='pt')
        outputs = roberta_model(**inputs)
        score = outputs.logits[0][0].item()  
        

        action_scores.append((score, exec_action_lists[i]))

    top_20_actions = sorted(action_scores, key=lambda x: x[0], reverse=True)[:20]


    for score, action in top_20_actions:
        if mem_graph.simulate_action(action) is True:
            break
    return round(score), action

    
    
    
    
    
    
    
    


def sample_model_action(args, action_logits, object_logits, resampling, obs, agent_id, type='multinomial'):
    if type == 'argmax':
        agent_action = int(action_logits.argmax())
        agent_obj = int(object_logits.argmax())
    elif type == 'multinomial':
        action_dist = torch.distributions.Multinomial(logits=action_logits, total_count=1)
        obj_dist = torch.distributions.Multinomial(logits=object_logits, total_count=1)
        agent_action = int(torch.argmax(action_dist.sample(), dim=-1))
        agent_obj = int(torch.argmax(obj_dist.sample(), dim=-1))
    elif type == 'multinomial_random':
        p = random.uniform(0, 1)
        if p < args.model_exploration_p:

            count = 0
            while 1:

                if resampling == -1 and count == 0:
                    agent_action = int(torch.argmax(action_logits))
                else:
                    agent_action = int(torch.multinomial(action_logits, 1))

                ## randomly select an action if stuck at a single action
                if count > 50 or resampling > 50:
                    agent_action = random.choice(list(args.vocabulary_action_name_word_index_dict.values()))

                object_logits_tem = deepcopy(object_logits)

                if agent_action == args.vocabulary_action_name_word_index_dict['none']:
                    agent_obj = None
                else:
                    agent_obj_space, agent_obj = utils_interactive_eval.get_valid_action_space(args, agent_action, obs,
                                                                                               agent_id)

                    if agent_obj_space is not None:
                        not_agent_obj_space = [idx for idx in list(range(object_logits_tem.shape[1])) if
                                               idx not in agent_obj_space]
                        object_logits_tem[0][torch.tensor(not_agent_obj_space)] = -99999
                        object_logits_tem = F.softmax(object_logits_tem, -1)

                        if resampling == -1 and count == 0:
                            agent_obj = int(torch.argmax(object_logits_tem))
                        else:
                            agent_obj = int(torch.multinomial(object_logits_tem, 1))

                        assert agent_obj in agent_obj_space
                        break

                count += 1
        else:
            count = 0
            while 1:
                action_logits_uniform = torch.ones_like(action_logits) / action_logits.shape[1]
                agent_action = int(torch.multinomial(action_logits_uniform, 1))
                count += 1

                if agent_action == args.vocabulary_action_name_word_index_dict['none']:
                    agent_obj = None
                else:
                    agent_obj_space, agent_obj = utils_interactive_eval.get_valid_action_space(args, agent_action, obs,
                                                                                               agent_id)

                if agent_obj is not None:
                    break

    agent_action = args.vocabulary_action_name_index_word_dict[agent_action]
    resampling += 1
    return agent_action, agent_obj, resampling


def sample_action(args, obs, agent_id, action_logits, object_logits, all_actions, all_cur_observation, logging):
    graph_nodes = obs[agent_id]['nodes']
    agent_action = None
    agent_obj = None
    valid_action = False
    resampling = -1
    sample_model_action_type = 'multinomial_random'

    while 1:
        if agent_action == None or agent_obj == None or agent_obj >= len(graph_nodes):
            agent_action, agent_obj, resampling = sample_model_action(args, action_logits, object_logits, resampling,
                                                                      obs, agent_id, type=sample_model_action_type)
        else:
            selected_node = graph_nodes[agent_obj]

            print(agent_action, selected_node['class_name'])
            action_obj_str, bad_action_flag = utils_interactive_eval.can_perform_action(agent_action,
                                                                                        o1=selected_node['class_name'],
                                                                                        o1_id=selected_node['id'],
                                                                                        agent_id=agent_id + 1,
                                                                                        graph=obs[agent_id],
                                                                                        teleport=True)

            bad_action_flag_v2, ignore_walk = utils_interactive_eval.check_logical_before_unity(agent_id,
                                                                                                cur_action=action_obj_str,
                                                                                                actions_sofar=all_actions,
                                                                                                observations_sofar=all_cur_observation,
                                                                                                logging=logging,
                                                                                                verbose=False)

            if bad_action_flag or bad_action_flag_v2 or ignore_walk:
                agent_action, agent_obj, resampling = sample_model_action(args, action_logits, object_logits,
                                                                          resampling, obs, agent_id,
                                                                          type=sample_model_action_type)
            else:
                valid_action = True
                break

    if not valid_action:
        ignore_walk = False
        action_obj_str = None

    return action_obj_str, ignore_walk, resampling


def compute_task_complexity(task_goal, graph):
    min_steps = 0
    for goal in task_goal:
        goal_num = task_goal[goal]
        # print(goal, goal_num)
        if 'close' in goal:
            min_steps += 1
        elif 'turn' in goal:
            min_steps += 1
        elif 'inside' in goal:
            obj_name = goal.split('_')[1]
            obj_num = goal_num
            inside_num = 0
            out_num = 0
            # pan duan obj wei zhi
            for node in graph['nodes']:
                if node['class_name'] == obj_name:
                    obj_id = node['id']
                    from_obj_edges = [edge for edge in graph['edges'] if edge['from_id'] == obj_id]
                    for edge in reversed(from_obj_edges):
                        if edge['relation_type'] == 'INSIDE':
                            inside_num += 1
                            break
                        elif edge['relation_type'] == 'ON':
                            out_num += 1
                            break
            if obj_num <= out_num:
                min_steps += 4 * goal_num
            else:
                min_steps += 4 * out_num + 5 * (obj_num - out_num)
            min_steps = min_steps + 1 + obj_num
        elif 'on' in goal:
            obj_name = goal.split('_')[1]
            obj_num = goal_num
            inside_num = 0
            out_num = 0
            # pan duan obj wei zhi
            for node in graph['nodes']:
                if node['class_name'] == obj_name:
                    obj_id = node['id']
                    from_obj_edges = [edge for edge in graph['edges'] if edge['from_id'] == obj_id]
                    for edge in reversed(from_obj_edges):
                        print(edge)
                        if edge['relation_type'] == 'INSIDE':
                            inside_num += 1
                            break
                        elif edge['relation_type'] == 'ON':
                            out_num += 1
                            break
            if obj_num <= out_num:
                min_steps += 4 * obj_num
            else:
                min_steps += 4 * out_num + 5 * (obj_num - out_num)
            min_steps = min_steps + obj_num
    return min_steps


def interactive_interface_fn(args, vh_envs, iteri, agent_model, data_info, logging, tokenizer):
    # control flags
    if_gpt = False
    if_exe_all_action = True
    verbose = True
    valid_run = 0
    success_count = 0
    save_output = []
    camera_num = vh_envs.comm.camera_count()[1]
    save_data_all = []
    if_script = False
    roberta_model, roberta_tokenizer = load_model()
    i = 0
    while 1:
        i += 1
        if i > 0:
            if_gpt = False
            if_script = True
        print('valid_run/current_run', valid_run, i)
        # if valid_run>=args.test_examples:
        #     break
        if i > args.test_examples:
            break
        all_cur_observation = []
        all_actions = []
        all_rewards = []
        all_frames = []

        if True:
            obs, env_graph = vh_envs.reset(task_id=i)
            obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])
            all_cur_observation.append(deepcopy(obs))

            B = 1
            steps = 0

            valid_run_tem = False
            success_run_tem = False

            # -----------compute task complexity-------------------------
            '''
            task_goal = vh_envs.task_goal[0]
            graph = env_graph
            complexity = compute_task_complexity(task_goal, graph)
            print(complexity, task_goal)
            file_path = 'NovelTasks_complexity.txt'
            with open(file_path, 'a') as file:
                file.write('task_id: %s  min_steps: %s\n' % (i, complexity))
            '''
            if if_script:
                with open('script.txt', 'r') as file:
                    script = file.readlines()
                    exe_index = 0
            # --------------
            # set gpt policy
            # --------------
            if if_gpt:
                gpt_policy = GPTPolicy(logging)
                gpt_policy.set_graph(env_graph)  # 设置gpt任务环境
                gpt_policy.set_goal(vh_envs.task_goal[0])  # 设置gpt任务目标
                if if_exe_all_action:
                    pass
                    gpt_policy.generate_recurrent_plan()  # 生成gpt任务规划
                    # gpt_policy.generate_prog_plan(gpt_policy.task_goal)
                    # gpt_policy.generate_plan_old()
                    # gpt_policy.generate_PR_plan(gpt_policy.task_goal)

                else:
                    gpt_policy.split_task_goal, gpt_policy.split_task_goal_num = split_goal(logging,
                                                                                            gpt_policy.task_goal)
            while (1):
                if verbose:
                    logging.info(
                        '----------------------------------------------------------------------------------------------------')
                recent_action = []
                recent_dis = []
                agent_id = 0
                agent_actions = {}
                agent_rewards = {}
                agent_ignore_walk = {}
                ignore_walk = None

                ## ----------------------------------------------------------------------------------------------------
                ## convert data format 
                ## ----------------------------------------------------------------------------------------------------
                # data, bad_observation_flag = utils_interactive_eval.get_interactive_input(args, agent_id, data_info, vh_envs, all_cur_observation, all_actions, tokenizer)

                # if bad_observation_flag:
                #     logging.info('----------------------------------------------------------------------------------')
                #     logging.info('interactive eval: convert data format fail!')
                #     logging.info('----------------------------------------------------------------------------------')
                #     valid_run_tem = False
                #     break

                ## ----------------------------------------------------------------------------------------------------
                ## get action from model and check action
                ## ----------------------------------------------------------------------------------------------------
                # action, obj = agent_model.get_action(data=data)

                # action_logits = F.softmax(action[agent_id], dim=-1)
                # object_logits = F.softmax(obj[agent_id], dim=-1)
                # action_obj_str, ignore_walk, resampling = sample_action(args, obs, agent_id, action_logits, object_logits, all_actions, all_cur_observation, logging)
                # print('[INFO] LID predict:', action_obj_str)

                ## ----------------------------------------------------------------------------------------------------
                ## get action from chatgpt
                ## ----------------------------------------------------------------------------------------------------
                action_obj_str = ''
                if if_gpt:
                    if if_exe_all_action:
                        gpt_action_obj_str = gpt_policy.get_action_from_chatgpt()
                        if gpt_action_obj_str != '':
                            logging.info('[INFO] GPT predict:' + gpt_action_obj_str)
                            # print('[INFO] GPT predict:', gpt_action_obj_str)
                    else:
                        gpt_action_obj_str = gpt_policy.get_action_from_chatgpt()
                        if gpt_action_obj_str == '':
                            if gpt_policy.goal_exe_index < gpt_policy.split_task_goal_num:
                                current_task = gpt_policy.split_task_goal[gpt_policy.goal_exe_index]
                                gpt_policy.goal_exe_index += 1
                                gpt_policy.generate_plan(current_task,roberta_model, roberta_tokenizer)
                            gpt_action_obj_str = gpt_policy.get_action_from_chatgpt()
                    action_obj_str = gpt_action_obj_str
                if if_script:
                    action_obj_str = script[exe_index]
                    exe_index += 1
                agent_actions[agent_id] = action_obj_str
                agent_ignore_walk[agent_id] = ignore_walk

                ## ----------------------------------------------------------------------------------------------------
                ## send action to the environment
                ## ----------------------------------------------------------------------------------------------------
                t_score = calsco(roberta_model, roberta_tokenizer, gpt_policy.task_goal.split("(id:")[0], recent_action, agent_actions[0])
                if t_score < 4:
                    t_score, agent_actions[0] = score_search(roberta_model, roberta_tokenizer, vh_envs.task_goal[0], recent_action, gpt_policy.task_goal.split("(id:")[0], env_graph)
                recent_dis.append(t_score)
                recent_action.append(agent_actions[0])
                obs, rewards, dones, infos, success = vh_envs.step(agent_actions, ignore_walk=agent_ignore_walk,
                                                                   logging=logging)  # next_obs

                if rewards == dones == infos == success == None:
                    logging.info('----------------------------------------------------------------------------------')
                    logging.info('interactive eval: unity action fail!')
                    logging.info('[INFO] failed reason: ' + json.dumps(obs))
                    logging.info('----------------------------------------------------------------------------------')
                    valid_run_tem = False
                    break

                ## ---------------------------------------------------------------------------------------------------------
                ## check action after send to Unity
                ## ---------------------------------------------------------------------------------------------------------
                obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])
                env_bug_count_a0 = not check_env_bug(agent_actions[0], obs[0], agent_i=0, logging=logging)

                if env_bug_count_a0:
                    logging.info('----------------------------------------------------------------------------------')
                    logging.info('interactive eval: check_env_bug outside unity fail!')
                    logging.info('----------------------------------------------------------------------------------')
                    valid_run_tem = False
                    break

                ## ----------------------------------------------------------------------------------------------------
                ## reward
                ## ----------------------------------------------------------------------------------------------------
                reward = torch.tensor(rewards)
                if reward[0] is not None:
                    agent_rewards[0] = reward[0]

                ## ----------------------------------------------------------------------------------------------------
                ## done, bad end
                ## ----------------------------------------------------------------------------------------------------
                all_cur_observation.append(deepcopy(obs))
                all_actions.append(deepcopy(agent_actions))
                all_rewards.append(deepcopy(agent_rewards))

                ## ---------------------------------------------------------------------------------------------------------
                ## log
                ## ---------------------------------------------------------------------------------------------------------
                if verbose:
                    env_task_goal_write = ['%s_%d' % (k, v) for k, v in vh_envs.task_goal[0].items() if v > 0]

                    logging.info('example %d, step %d, goal %s' % (i, steps, str(env_task_goal_write)))
                    logging.info(('A0: Act: %s' % str(agent_actions[0])))
                    logging.info(('A0: Rew: %s' % str(agent_rewards[0])))
                    if agent_actions[0] is not None:
                        logging.info(('ignore_walk: %s' % str(agent_ignore_walk[0])))

                ## ---------------------------------------------------------------------------------------------------------
                ## break if done
                ## ---------------------------------------------------------------------------------------------------------
                steps += 1
                if np.any(dones):
                    valid_run_tem = True

                    if infos[0]['is_success']:
                        success_run_tem = True
                    break

            if valid_run_tem:

                valid_run += 1

                for tem in all_actions: logging.info(tem)

                if success_run_tem:
                    success_count += 1
                    print('-------------------------------------------------------------------')
                    print('success example')
                    print('-------------------------------------------------------------------')

        if args.interactive_eval:
            success_rate = 100. * success_count / valid_run if valid_run != 0 else 0

            if args.eval:
                logging.info(" {} / {} \n \
                            Total / Current_run / Valid / Success: {} / {} / {} / {} \n \
                            Success Rate: {:.3f}"
                             .format(args.pretrained_model_dir, args.subset,
                                     args.test_examples, i, valid_run, success_count,
                                     success_rate))
            else:
                logging.info(" {} / {} \n \
                            Total / Current_run / Valid / Success: {} / {} / {} / {} \n \
                            Success Rate: {:.3f}"
                             .format(args.save_dir, args.subset,
                                     args.test_examples, i, valid_run, success_count,
                                     success_rate))

    return success_rate

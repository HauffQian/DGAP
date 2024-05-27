import re


class MemoryGraph:
    def __init__(self, graph):
        self.graph = graph
        self.action_list = []
        self.id_state_dict = {}

    def set_graph(self, graph):
        self.graph = graph
        for node in graph['nodes']:
            self.id_state_dict[node['id']] = node['states']

    def simulate_action(self, action_script):
        self.action_list.append(action_script)
        action, objects, ids = extract_actions(action_script)
        if action[0] == 'open':
            self.OpenExecute(int(ids[0]))
        elif action[0] == 'close':
            self.CloseExecute(int(ids[0]))
        elif action[0] == 'grab':
            self.GrabExecute(int(ids[0]))
        elif action[0] == 'switchon':
            self.SwitchonExecute(int(ids[0]))
        elif action[0] == 'putin':
            self.PutinExecute(int(ids[0]), int(ids[1]))
        elif action[0] == 'putback':
            self.PutbackExecute(int(ids[0]), int(ids[1]))
        else:
            pass

    def GrabExecute(self, id_):
        for edge in self.graph['edges']:
            if edge['from_id'] == id_ and edge['relation_type'] == 'INSIDE':
                self.graph['edges'].remove(edge)
                break
        new_edge = {'from_id': 0, 'to_id': id_, 'relation_type': 'HOLDS_RH'}
        self.graph['edges'].append(new_edge)

    def CloseExecute(self, id_):
        if id_ in self.id_state_dict:
            state = self.id_state_dict[id_]
            if 'CLOSED' in state:
                state.remove('CLOSED')
                state.append('OPEN')
            elif 'OPENED' in state:
                state.remove('OPENED')
                state.append('CLOSED')
            self.id_state_dict[id_] = state
            self.change_graph(id_, state)
        else:
            return False

    def OpenExecute(self, id_):
        if id_ in self.id_state_dict:
            state = self.id_state_dict[id_]
            if 'CLOSED' in state:
                state.remove('CLOSED')
                state.append('OPENED')
            elif 'OPENED' in state:
                state.remove('OPENED')
                state.append('CLOSED')
            self.id_state_dict[id_] = state
            self.change_graph(id_, state)
        else:
            return False

    def SwitchonExecute(self, id_):
        if id_ in self.id_state_dict:
            state = self.id_state_dict[id_]
            if 'OFF' in state:
                state.remove('OFF')
                state.append('ON')
            self.id_state_dict[id_] = state
            self.change_graph(id_, state)
        else:
            return False

    def change_graph(self, id_, state):
        i = 0
        for node in self.graph['nodes']:
            if node['id'] == id_:
                self.graph['nodes'][i]['states'] = state
            i += 1

    def PutinExecute(self, put_obj_id, in_obj_id):
        pass

    def PutbackExecute(self, put_obj_id, in_obj_id):
        pass


def extract_actions(str):
    # 提取方括号中的内容
    actions = re.findall('\[(.*?)\]', str)
    # 提取尖括号中的内容
    objects = re.findall('<(.*?)>', str)
    # 提取圆括号中的内容
    values = re.findall('\((.*?)\)', str)
    return actions, objects, values

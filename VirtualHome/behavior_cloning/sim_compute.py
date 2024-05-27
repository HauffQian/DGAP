from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class Similarity:
   
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def __init__(self):
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        return
    
    def sim_compute(self, text1, text2):

        # Sentences we want sentence embeddings for
        sentences = [text1, text2]

     

        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        cos = torch.nn.CosineSimilarity(dim = 0)
        # print(sentence_embeddings.shape)
        return cos(sentence_embeddings[0], sentence_embeddings[1]).item()


if __name__ == '__main__':
    sim = Similarity()
    print(sim.sim_compute("I like to eat apples", "I like to eat apples"))


    # exampleTask = ''
    # with open('./demo_set_inDis.txt', 'r') as f:
    #     demo_set_temp = f.read()
    # f.close()

    # # 提取每个task的字符串信息
    # demo_set = demo_set_temp.split('task_id')[1:]
    # demo_list = []
    # for demo in demo_set:
    #     # # 提取task_id
    #     task_id = demo.split('\n')[0]

    #     demo = demo.replace('-', '').replace(task_id, '')

    #     # 提取task_goal
    #     task_goal = demo.split('#The goal means the task is')[1].split('\n')[0]
    #     print(task_goal)

    #     demo_list.append(demo)
        
    #     print('[id]',task_id)
    #     print('[goal]',task_goal)
    #     print(demo)

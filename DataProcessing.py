import pandas as pd
import numpy as np
import random

class DataClean():
    
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_csv(self):
        df = pd.read_csv(self.data_path)
        return df
    
    def post_embedding_clean(self):
        df = self.load_csv()
        
        post_embedding_list = []
        for post_embedding in df['post_embedding']:
            if post_embedding[1] == ' ':
                post_embedding = post_embedding.replace('\n', ' ').split()[1:]
                post_embedding = [float(post_embedding[i]) for i in range(len(post_embedding)-1)] + [float(post_embedding[-1].split(']')[0])]
                post_embedding_list.append(post_embedding)
        
            else:
                post_embedding = post_embedding.replace('\n', ' ').split()
                post_embedding = [float(post_embedding[0].split('[')[1])] + [float(post_embedding[i]) for i in range(1, len(post_embedding)-1)] +\
                                [float(post_embedding[-1].split(']')[0])]
                post_embedding_list.append(post_embedding)
        
        return np.array(post_embedding_list)
    
    def video_embedding_clean(self):
        df = self.load_csv()
        
        video_embedding_list = []
        for video_embedding in df['video_embedding']:
            if video_embedding[1] == ' ':
                video_embedding = video_embedding.replace('\n', ' ').split()[1:]
                video_embedding = [float(video_embedding[i]) for i in range(len(video_embedding)-1)] + [float(video_embedding[-1].split(']')[0])]
                video_embedding_list.append(video_embedding)
        
            else:
                video_embedding = video_embedding.replace('\n', ' ').split()
                video_embedding = [float(video_embedding[0].split('[')[1])] + [float(video_embedding[i]) for i in range(1, len(video_embedding)-1)] +\
                                [float(video_embedding[-1].split(']')[0])]
                video_embedding_list.append(video_embedding)
                
        return np.array(video_embedding_list)
    
    def label_processing(self):
        df = self.load_csv()
        
        def category(score):
            if score == 'bad':
                return 0
            elif score == 'good':
                return 1
            else:
                return 2
        
        df['label'] = df['score'].apply(lambda x: category(x))
        return np.array(df['label']).reshape(-1, 1)
    
    def get_data(self):
        post_embeddings = self.post_embedding_clean()
        video_embeddings = self.video_embedding_clean()
        labels = self.label_processing()
    
        return post_embeddings, video_embeddings, labels
    

class DataSeparation():
    def __init__(self, post_embeddings, video_embeddings, labels, train_ratio, validation_ratio, random_seed):
        """
        post_embeddings: numpy array 
        video_embeddings: numpy array
        labels: numpy array
        """
        
        self.post_embeddings = post_embeddings
        self.video_embeddings = video_embeddings
        self.labels = labels
        self.length = post_embeddings.shape[0]
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.random_seed = random_seed
    
    def shuffle(self):
        random.seed(self.random_seed)
        indices = [i for i in range(self.length)]
        random.shuffle(indices)
        train_indices = indices[:int(self.train_ratio * self.length)]
        validation_indices = indices[int(self.train_ratio * self.length): int((self.train_ratio + self.validation_ratio) * self.length)]
        test_indices = indices[int((self.train_ratio + self.validation_ratio) * self.length):]   
        
        return train_indices, validation_indices, test_indices
    
    def get_data(self):
        train_indices, validation_indices, test_indices = self.shuffle()
        
        train_post = self.post_embeddings[train_indices]
        validation_post = self.post_embeddings[validation_indices]
        test_post = self.post_embeddings[test_indices]
        
        train_video = self.video_embeddings[train_indices]
        validation_video = self.video_embeddings[validation_indices]
        test_video = self.video_embeddings[test_indices]
        
        train_labels = self.labels[train_indices]
        validation_labels = self.labels[validation_indices]
        test_labels = self.labels[test_indices]
        
        posts = {'train': train_post, 'validation': validation_post, 'test': test_post}
        videos = {'train': train_video, 'validation': validation_video, 'test': test_video}
        labels = {'train': train_labels, 'validation': validation_labels, 'test': test_labels}
        indices = {'train': train_indices, 'validation': validation_indices, 'test': test_indices}
        
        return posts, videos, labels, indices
        
    
        
        
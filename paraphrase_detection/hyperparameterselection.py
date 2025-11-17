from sentence_transformers import SentenceTransformer


class BOSearchTrain():
    def __init__(self,
                 hyperparams : dict,
                 sbert_model : SentenceTransformer,
                 fc_layer_sizes : list,

                 ):
        self.hyperparams = hyperparams
    
    def train_eval_set()
        
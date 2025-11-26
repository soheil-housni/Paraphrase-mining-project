from dataclasses import dataclass

class HP():
    def __init__(self):
        pass
    
    @dataclass
    class CrossEntropyHP:
        lr: float
        weight_decay: float
        dropout: float
        batch_size: int
        n_freeze : int
        use_n_layers: int
        fc1: int
        fc2: int
        fc3: int

    @dataclass
    class CrossAttentionHP:
        lr: float
        weight_decay: float
        dropout: float
        batch_size: int
        n_freeze : int
        use_n_layers: int
        fc1: int
        fc2: int
        fc3: int
        use_n_layers_cross_att: int
        fc1_cross_att: int
        fc2_cross_att: int

    @dataclass
    class CosineSimilarityHP:
        lr: float
        weight_decay: float
        dropout: float
        batch_size: int
        n_freeze : int
        use_n_layers: int
        fc1: int
        fc2: int
        fc3: int
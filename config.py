
DATA_DIR = 'data/example_2'
GRAPH_PATH = DATA_DIR + '/edges.tsv'
TRAINING_PATH = DATA_DIR + '/training.tsv'
TEST_PATH = DATA_DIR + '/test.tsv'
MODEL_DIR = 'model_2'

def get_torchbiggraph_config():

    config = dict(
        # I/O data
        entity_path=DATA_DIR,
        edge_paths=[
            DATA_DIR + '/edge_path'
        ],
        checkpoint_path=MODEL_DIR,
        # Graph structure
        entities={"all": {"num_partitions": 1}},
        relations=[
           {
                "name": "all_edges",
                "lhs": "all",
                "rhs": "all",
                "operator": "translation",
            }
        ],

        dynamic_relations=True,
        dimension=200, #400 
        global_emb=False,
        comparator="dot",
        num_epochs=200, #200 
        num_uniform_negs=1000,
        loss_fn="softmax",
        lr=0.1,
        regularization_coef=1e-3,
        eval_fraction=0.,
    )

    return config



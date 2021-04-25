import os
import shutil
import json
import random 
from pathlib import Path
# import h5py
# import torch
# from torchbiggraph.config import parse_config
# from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
# from torchbiggraph.train import train
# from torchbiggraph.util import SubprocessInitializer, setup_logging
# from torchbiggraph.model import DotComparator

DATA_DIR = 'data/example_2'
GRAPH_PATH = DATA_DIR + '/edges.tsv'
TRAINING_PATH = DATA_DIR + '/training.tsv'
TEST_PATH = DATA_DIR + '/test.tsv'
MODEL_DIR = 'model_2'

if __name__ == '__main__':
    try:
        shutil.rmtree('data')
    except:
        pass
    try:
        shutil.rmtree('model_2')
    except:
        pass

    print('inside  MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM')
    
    # ==================================================================
    # 0. PREPARE THE GRAPH
    # the result of this step is a single file 'data/example_2/graph.tsv'
    # ==================================================================

    # This the graph we will be embedding.
    # It has 10 types of nodes, or entities, and 9 types of edges, or relationships. 
    test_edges = []
    train_data = []
    test_data = []

    check_entities = []

    count=0
    count2=0
    with open('kb.txt', 'r') as f: 
        for line in f: 
           line=line.rstrip().split("|")
           line[0] = line[0].split(" ")
           line[0] = "_".join(line[0])
        #    line[2] = line[2].split(" ")
        #    line[2] = "_".join(line[2])
           test_edges.append(line)
           count+=1
           if count == 134741:
               break

    with open('kb.txt', 'r' ) as f:
        for line in f:
           line=line.rstrip().split("|")
           line[0] = line[0].split(" ")
           line[0] = "_".join(line[0])
           count2 +=1 
           if count2 <= 80000:
               train_data.append(line)
               check_entities.append(line[0])
               check_entities.append(line[1])
               check_entities.append(line[2])
           elif count2 > 80000:
               print('a')
               if line[0] in check_entities and line[1] in check_entities and line[2] in check_entities:
                    print('b')
                    test_data.append(line) 

    print(len(train_data))
    print(len(test_data))

    # training set and test set  

    # split_index = len(test_edges) // 1.25  
    # training_edges = test_edges[:split_index]
    # split_edges = test_edges[split_index:]


    os.makedirs(DATA_DIR, exist_ok=True)
    with open(GRAPH_PATH, 'w') as f:
        for edge in test_edges:
            f.write('\t'.join(edge) + '\n')

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TRAINING_PATH, 'w') as f:
        for edge in train_data:
            f.write('\t'.join(edge) + '\n')    
            
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TEST_PATH, 'w') as f:
        for edge in test_data:
            f.write('\t'.join(edge) + '\n')    
    
    # # # 1. DEFINE CONFIG
    # # # this dictionary will be used in steps 2. and 3.
    # # # ==================================================

#     raw_config = dict(
#         # I/O data
#         entity_path=DATA_DIR,
#         edge_paths=[
#             DATA_DIR + '/edges_partitioned',
#         ],
#         checkpoint_path=MODEL_DIR,
#         # Graph structure
#         entities={
#             "movie": {"num_partitions": 1},
#             "director": {"num_partitions": 1},
#             "writer": {"num_partitions": 1},
#             "starred_actor": {"num_partitions": 1},
#             "year": {"num_partitions": 1},
#             "language": {"num_partitions": 1},
#             "tags": {"num_partitions": 1},
#             "genre": {"num_partitions": 1}, 
#             "votes": {"num_partitions":1}, 
#             "rating": {"num_partitions":1}
#         },
#         relations=[
#             {
#                 "name": "directed_by",
#                 "lhs": "movie",
#                 "rhs": "director",
#                 "operator": "complex_diagonal",
#             },
#             {
#                 "name": "written_by",
#                 "lhs": "movie",
#                 "rhs": "writer",
#                 "operator": "complex_diagonal",
#             },
#             {
#                 "name": "starred_actors",
#                 "lhs": "movie",
#                 "rhs": "starred_actor",
#                 "operator": "complex_diagonal",
#             },
#             {
#                 "name": "release_year",
#                 "lhs": "movie",
#                 "rhs": "year",
#                 "operator": "complex_diagonal",
#             },
#             {
#                 "name": "in_language",
#                 "lhs": "movie",
#                 "rhs": "language",
#                 "operator": "complex_diagonal",
#             },
#             {
#                 "name": "has_tags",
#                 "lhs": "movie",
#                 "rhs": "tags",
#                 "operator": "complex_diagonal",
#             },            
#             {
#                 "name": "has_genre",
#                 "lhs": "movie",
#                 "rhs": "genre",
#                 "operator": "complex_diagonal",
#             },
#             {
#                 "name": "has_imdb_votes",
#                 "lhs": "movie",
#                 "rhs": "votes",
#                 "operator": "complex_diagonal",
#             },
#             {
#                 "name": "has_imdb_rating",
#                 "lhs": "movie",
#                 "rhs": "rating",
#                 "operator": "complex_diagonal",
#             }

#         ],

#         dynamic_relations=False,
#         dimension=4,  
#         global_emb=False,
#         comparator="dot",
#         num_epochs=7,
#         num_uniform_negs=1000,
#         loss_fn="softmax",
#         lr=0.1,
#         regularization_coef=1e-3,
#         eval_fraction=0.,
#     )

#     # =================================================
#     # 2. TRANSFORM GRAPH TO A BIGGRAPH-FRIENDLY FORMAT
#     # This step generates the following metadata files:
    
#     # data/example_2/entity_count_director_0.txt
#     # data/example_2/entity_count_genre_0.txt
#     # data/example_2/entity_count_language_0.txt
#     # data/example_2/entity_count_movie_0.txt
#     # data/example_2/entity_count_rating_0.txt
#     # data/example_2/entity_count_starred_actor_0.txt   
#     # data/example_2/entity_count_tags_0.txt
#     # data/example_2/entity_count_votes_0.txt
#     # data/example_2/entity_count_writer_0.txt
#     # data/example_2/entity_count_year_0.txt

#     # data/example_2/entity_count_director_0.json
#     # data/example_2/entity_count_genre_0.json
#     # data/example_2/entity_count_language_0.json
#     # data/example_2/entity_count_movie_0.json
#     # data/example_2/entity_count_rating_0.json
#     # data/example_2/entity_count_starred_actor_0.json   
#     # data/example_2/entity_count_tags_0.json
#     # data/example_2/entity_count_votes_0.json
#     # data/example_2/entity_count_writer_0.json
#     # data/example_2/entity_count_year_0.json
    
#     # and this file with data:
#     # data/example_2/edges_partitioned/edges_0_0.h5
#     # =================================================
#     setup_logging()
#     config = parse_config(raw_config)
#     subprocess_init = SubprocessInitializer()
#     input_edge_paths = [Path(GRAPH_PATH)]

#     convert_input_data(
#         config.entities,
#         config.relations,
#         config.entity_path,
#         config.edge_paths,
#         input_edge_paths,
#         TSVEdgelistReader(lhs_col=0, rel_col=1, rhs_col=2),
#         dynamic_relations=config.dynamic_relations,
#     )

#     # ===============================================
#     # 3. TRAIN THE EMBEDDINGS
#     # files generated in this step:
#     #
#     # checkpoint_version.txt
#     # config.json
#     # embeddings_director_0.v7.h5
#     # embeddings_genre_0.v7.h5
#     # embeddings_language_0.v7.h5
#     # embeddings_movie_0.v7.h5
#     # embeddings_rating_0.v7.h5
#     # embeddings_starred_actor_0.v7.h5
#     # embeddings_tags_0.v7.h5
#     # embeddings_votes_0.v7.h5
#     # embeddings_writer_0.v7.h5
#     # embeddings_year_0.v7.h5

#     # model.v7.h5
#     # training_stats.json
#     # ===============================================

#     train(config, subprocess_init=subprocess_init)

#     # =======================================================================
#     # 4. LOAD THE EMBEDDINGS
#     # The final output of the process consists of a dictionary mapping each entity to its embedding

#     # =======================================================================

#     # entities_path = DATA_DIR + '/entity_names_entities_0.json'

#     # entities_emb_path = MODEL_DIR + "/embeddings_entities.v{NUMBER_OF_EPOCHS}.h5" \
#     #     .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

#     # with open(entities_path, 'r') as f:
#     #     entities = json.load(f)

#     # with h5py.File(entities_emb_path, 'r') as g:
#     #     entity_embeddings = g['embeddings'][:]

#     # entity2embedding = dict(zip(entities, entity_embeddings))
#     # print('entity embeddings')
#     # print(entity2embedding)

#     movies_path = DATA_DIR + '/entity_names_movie_0.json'
#     directors_path = DATA_DIR + '/entity_names_director_0.json'
#     writers_path = DATA_DIR + '/entity_names_writer_0.json'
#     actors_path = DATA_DIR + '/entity_names_starred_actor_0.json'
#     years_path = DATA_DIR + '/entity_names_year_0.json'
#     languages_path = DATA_DIR + '/entity_names_language_0.json'
#     tags_path = DATA_DIR + '/entity_names_tags_0.json'
#     genres_path = DATA_DIR + '/entity_names_genre_0.json'
#     votes_path = DATA_DIR + '/entity_names_votes_0.json'
#     rating_path = DATA_DIR + '/entity_names_rating_0.json'


#     movie_emb_path = MODEL_DIR + "/embeddings_movie_0.v{NUMBER_OF_EPOCHS}.h5" \
#         .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

#     director_emb_path = MODEL_DIR + "/embeddings_director_0.v{NUMBER_OF_EPOCHS}.h5" \
#         .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

#     writer_emb_path = MODEL_DIR + "/embeddings_writer_0.v{NUMBER_OF_EPOCHS}.h5" \
#         .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])
    
#     actor_emb_path = MODEL_DIR + "/embeddings_starred_actor_0.v{NUMBER_OF_EPOCHS}.h5" \
#         .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

#     year_emb_path = MODEL_DIR + "/embeddings_year_0.v{NUMBER_OF_EPOCHS}.h5" \
#         .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])
    
#     language_emb_path = MODEL_DIR + "/embeddings_language_0.v{NUMBER_OF_EPOCHS}.h5" \
#         .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])
    
#     tags_emb_path = MODEL_DIR + "/embeddings_tags_0.v{NUMBER_OF_EPOCHS}.h5" \
#         .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

#     genre_emb_path = MODEL_DIR + "/embeddings_genre_0.v{NUMBER_OF_EPOCHS}.h5" \
#         .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

#     votes_emb_path = MODEL_DIR + "/embeddings_votes_0.v{NUMBER_OF_EPOCHS}.h5" \
#         .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

#     rating_emb_path = MODEL_DIR + "/embeddings_rating_0.v{NUMBER_OF_EPOCHS}.h5" \
#         .format(NUMBER_OF_EPOCHS=raw_config['num_epochs'])

#     with open(movies_path, 'r') as f:
#         movies = json.load(f)

#     with h5py.File(movie_emb_path, 'r') as g:
#         movie_embeddings = g['embeddings'][:]

#     movie2embedding = dict(zip(movies, movie_embeddings))
#     # print('movie embeddings')
#     # print(movie2embedding)

#     with open(directors_path, 'r') as f:
#         directors = json.load(f)

#     with h5py.File(director_emb_path, 'r') as g:
#         director_embeddings = g['embeddings'][:]

#     director2embedding = dict(zip(directors, director_embeddings))
#     # print('director embeddings')
#     # print(director2embedding)

#     with open(writers_path, 'r') as f:
#         writers = json.load(f)

#     with h5py.File(writer_emb_path, 'r') as g:
#         writer_embeddings = g['embeddings'][:]

#     writer2embedding = dict(zip(writers, writer_embeddings))
#     # print('writer embeddings')
#     # print(writer2embedding)

#     with open(actors_path, 'r') as f:
#         actors = json.load(f)

#     with h5py.File(actor_emb_path, 'r') as g:
#         actor_embeddings = g['embeddings'][:]

#     actor2embedding = dict(zip(actors, actor_embeddings))
#     # print('actor embeddings')
#     # print(actor2embedding)

#     with open(years_path, 'r') as f:
#         years = json.load(f)

#     with h5py.File(year_emb_path, 'r') as g:
#         year_embeddings = g['embeddings'][:]

#     year2embedding = dict(zip(years, year_embeddings))
#     # print('year embeddings')
#     # print(year2embedding)

#     with open(languages_path, 'r') as f:
#         languages = json.load(f)

#     with h5py.File(language_emb_path, 'r') as g:
#         language_embeddings = g['embeddings'][:]

#     language2embedding = dict(zip(languages, language_embeddings))
#     # print('language embeddings')
#     # print(language2embedding)

#     with open(tags_path, 'r') as f:
#         tags = json.load(f)

#     with h5py.File(tags_emb_path, 'r') as g:
#         tags_embeddings = g['embeddings'][:]

#     tag2embedding = dict(zip(tags, tags_embeddings))
#     # print('tag embeddings')
#     # print(tag2embedding)

#     with open(genres_path, 'r') as f:
#         genres = json.load(f)

#     with h5py.File(genre_emb_path, 'r') as g:
#         genre_embeddings = g['embeddings'][:]

#     genre2embedding = dict(zip(genres, genre_embeddings))
#     # print('genre embeddings')
#     # print(genre2embedding)

#     with open(votes_path, 'r') as f:
#         votes = json.load(f)

#     with h5py.File(votes_emb_path, 'r') as g:
#         votes_embeddings = g['embeddings'][:]

#     votes2embedding = dict(zip(votes, votes_embeddings))
#     # print('votes embeddings')
#     # print(votes2embedding)

#     with open(rating_path, 'r') as f:
#         ratings = json.load(f)

#     with h5py.File(rating_emb_path, 'r') as g:
#         rating_embeddings = g['embeddings'][:]

#     rating2embedding = dict(zip(ratings, rating_embeddings))
#     # print('rating embeddings')
#     # print(rating2embedding)

#     entity2embedding = {**movie2embedding, **director2embedding, **writer2embedding, **actor2embedding, **year2embedding, **language2embedding, **tag2embedding, **genre2embedding, **votes2embedding, **rating2embedding}
#     print('entity embeddings')
#     print(entity2embedding)


# #########################################################################

# # # Load count of dynamic relations
# # with open("data/FB15k/dynamic_rel_count.txt", "rt") as tf:
# #     dynamic_rel_count = int(tf.read().strip())

# # # Load the operator's state dict
# # with h5py.File("model/fb15k/model.v50.h5", "r") as hf:
# #     operator_state_dict = {
# #         "real": torch.from_numpy(hf["model/relations/0/operator/rhs/real"][...]),
# #         "imag": torch.from_numpy(hf["model/relations/0/operator/rhs/imag"][...]),
# #     }

# # operator = ComplexDiagonalDynamicOperator(400, dynamic_rel_count)
# # operator.load_state_dict(operator_state_dict)
# # comparator = DotComparator()

# # # Load the names of the entities, ordered by offset.
# # with open("data/FB15k/entity_names_all_0.json", "rt") as tf:
# #     entity_names = json.load(tf)
# # src_entity_offset = entity_names.index("/m/0f8l9c")  # France
# # dest_entity_offset = entity_names.index("/m/05qtj")  # Paris

# # # Load the names of the relation types, ordered by index.
# # with open("data/FB15k/dynamic_rel_names.json", "rt") as tf:
# #     rel_type_names = json.load(tf)
# # rel_type_index = rel_type_names.index("/location/country/capital")

# # # Load the trained embeddings
# # with h5py.File("model/fb15k/embeddings_all_0.v50.h5", "r") as hf:
# #     src_embedding = torch.from_numpy(hf["embeddings"][src_entity_offset, :])
# #     dest_embedding = torch.from_numpy(hf["embeddings"][dest_entity_offset, :])

# # # Calculate the scores
# # scores, _, _ = comparator(
# #     comparator.prepare(src_embedding.view(1, 1, 400)),
# #     comparator.prepare(
# #         operator(
# #             dest_embedding.view(1, 400),
# #             torch.tensor([rel_type_index]),
# #         ).view(1, 1, 400),
# #     ),
# #     torch.empty(1, 0, 400),  # Left-hand side negatives, not needed
# #     torch.empty(1, 0, 400),  # Right-hand side negatives, not needed
# # )

# # print(scores)

# #############################################################################

# # print("Now let's do some simple things within torch:")

# # src_entity_offset = dictionary["entities"]["user_id"].index("0")  # France
# # dest_1_entity_offset = dictionary["entities"]["user_id"].index("7")  # Paris
# # dest_2_entity_offset = dictionary["entities"]["user_id"].index("1")  # Paris
# # rel_type_index = dictionary["relations"].index("follow") # note we only have one...

# # with h5py.File("model/example_2/embeddings_user_id_0.v10.h5", "r") as hf:
# #     src_embedding = hf["embeddings"][src_entity_offset, :]
# #     dest_1_embedding = hf["embeddings"][dest_1_entity_offset, :]
# #     dest_2_embedding = hf["embeddings"][dest_2_entity_offset, :]
# #     dest_embeddings = hf["embeddings"][...]


# comparator = DotComparator()

# scores_1, _, _ = comparator(
#     comparator.prepare(torch.tensor(movie_embeddings.reshape([1,1,10]))),
#     comparator.prepare(torch.tensor(director_embeddings.reshape([1,1,10]))),
#     torch.empty(1, 0, 10),  # Left-hand side negatives, not needed
#     torch.empty(1, 0, 10),  # Right-hand side negatives, not needed
# )

# scores_2, _, _ = comparator(
#     comparator.prepare(torch.tensor(movie_embeddings.reshape([1,1,10]))),
#     comparator.prepare(torch.tensor(actor_embeddings.reshape([1,1,10]))),
#     torch.empty(1, 0, 10),  # Left-hand side negatives, not needed
#     torch.empty(1, 0, 10),  # Right-hand side negatives, not needed
# )

# print(scores_1)
# print(scores_2)
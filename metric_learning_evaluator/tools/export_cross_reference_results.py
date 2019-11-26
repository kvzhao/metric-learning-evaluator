"""Cross Reference Results
    Function: Use `query` searches on `anchor`.
"""
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from metric_learning_evaluator.index.agent import IndexAgent
from metric_learning_evaluator.data_tools.result_container import ResultContainer
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer


class Fields:
    query_instance_id = 'query_instance_id'
    retrieved_instance_id = 'retrieved_instance_id'
    retrieved_distance = 'retrieved_distance'


def main(args):

    data_dir = args.data_dir
    out_dir = args.out_dir
    query_command = args.query_command
    anchor_command = args.anchor_command
    # TODO: sanity check

    container = EmbeddingContainer()
    result = ResultContainer()
    container.load(data_dir)

    command = '{}->{}'.format(query_command, anchor_command)
    query_ids, anchor_ids = container.get_instance_id_by_cross_reference_command(command)
    query_embeddings = container.get_embedding_by_instance_ids(query_ids)
    anchor_embeddings = container.get_embedding_by_instance_ids(anchor_ids)

    num_of_anchor = anchor_embeddings.shape[0]
    num_of_query = query_embeddings.shape[0]

    agent = IndexAgent(agent_type='HNSW',
                       instance_ids=anchor_ids,
                       embeddings=anchor_embeddings)

    all_query_ids, all_retrieved_ids, all_retrieved_distances = [], [], []
    with tqdm(total=num_of_query) as pbar:
        for _idx, (query_id, qeury_emb) in enumerate(zip(query_ids, query_embeddings)):
            retrieved_ids, retrieved_distances = agent.search(qeury_emb, top_k=num_of_anchor)
            retrieved_ids = np.squeeze(retrieved_ids)
            retrieved_distances = np.squeeze(retrieved_distances)

            all_query_ids.extend(np.array(query_id).repeat(num_of_anchor))
            all_retrieved_ids.extend(retrieved_ids)
            all_retrieved_distances.extend(retrieved_distances)

            pbar.update()
    print('Indexing finished, {} retrieved events'.format(len(all_retrieved_ids)))
    print('Start exporting results...')
    start_time = time()
    result._event_buffer = pd.DataFrame(
        {
            Fields.query_instance_id: all_query_ids,
            Fields.retrieved_instance_id: all_retrieved_ids,
            Fields.retrieved_distance: all_retrieved_distances,
        }
    )
    result.save(out_dir)
    print('Done, saving results take {} seconds.'.format(time() - start_time))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Execute cross_reference on EmbeddingContainer & save ResultContainer.')

    parser.add_argument('-qc', '--query_command', type=str, default='query')
    parser.add_argument('-cc', '--anchor_command', type=str, default='anchor')
    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='Path to input EmbeddingContainer.')
    parser.add_argument('-od', '--out_dir', type=str, default=None,
                        help='Path to output EmbeddingContainer.')

    args = parser.parse_args()
    main(args)

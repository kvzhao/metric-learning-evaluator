
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

#from metric_learning_evaluator.index.np_agent import NumpyAgent
#from metric_learning_evaluator.index.hnsw_agent import HNSWAgent
from metric_learning_evaluator.core.registered import REGISTERED_INDEX_AGENT
from metric_learning_evaluator.utils.switcher import switch

class IndexAgent:
    # Wrapper of the IndexAgent

    def __init__(self,
                agent_type,
                instance_ids, embeddings,
                distance_measure='l2',
                ef_construction=200,
                num_threads=4,
                M=32):
        
        if not agent_type in REGISTERED_INDEX_AGENT:
            raise ValueError('index agent:{} is not registerred'.format(agent_type))
        else:
            self.agent_type = agent_type

        self.agent_engine = REGISTERED_INDEX_AGENT[self.agent_type](
            instance_ids, embeddings, distance_measure, ef_construction, num_threads, M)
    
    def search(self, query_embeddings, top_k=10):
        return self.agent_engine.search(query_embeddings, top_k)
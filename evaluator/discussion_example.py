"""
    Evaluator Builder

    The builder function organize evaluators according to given config.
"""

EVAL_METRIC_CLASS_DICT = {
    "" : "",
}

def build(eval_list):
    ops = [eval.get_ops() for eval in eval_list]
    return ops


class Builder(object):
    def __init__():
        data_obj = DataObj()
    
    def add_():
        self.data_obj.add_emb

    def get_evaluators(configs):
        return eval_list
    

class DataObj:

    def __init__(self):

        self.logits = None
        self.embedding = None
        self.attribute = None

    def add_single_emb(self):
        pass
    


class EvalCls(object):

    def __init__(self, DataObj):
        self.Data

    def compute_acc(self):
        metrics.cal_prob(DataObj.logits)

    def get_update_Op

class EvalRank:
    def __init__(DataObj)
        self.Data
    def rank():
    def get_update_Op
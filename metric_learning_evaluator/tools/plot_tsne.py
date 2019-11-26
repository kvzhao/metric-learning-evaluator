import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


from metric_learning_evaluator.analysis.tsne import TSNE
from metric_learning_evaluator.data_tools.embedding_container import EmbeddingContainer


def main(args):

    data_dir = args.data_dir
    out_dir = args.output_dir

    container = EmbeddingContainer()
    container.load(data_dir)

    tsne = TSNE(container,
                n_iter=args.iterations,
                n_jobs=args.n_jobs,
                perplexity=args.perplexity)

    tsne.run()
    tsne.save_fig(out_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('.')

    parser.add_argument('-dd', '--data_dir', type=str, default=None,
                        help='Path to exported features.')
    parser.add_argument('-od', '--output_dir', type=str, default=None,
                        help='Path to exported features.')
    parser.add_argument('-i', '--iterations', type=int, default=1000,
                        help='Number of iteration for running t-SNE.')
    parser.add_argument('-p', '--perplexity', type=float, default=30.0)
    parser.add_argument('-j', '--n_jobs', type=int, default=4)

    args = parser.parse_args()
    main(args)

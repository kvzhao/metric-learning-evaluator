"""
  A script for drawing tensorboard exported data.

  Example:
    python metric_learning_evaluator/tools/plot_tb_exported_curves.py
        -f run_triplet-rank_all_top_k_hit_accuracy@k=1 run_triplet-rank_all_top_k_hit_accuracy@k=5
        -l top_1 top_5
        -t Accuracy
        -of triplet_accuracy.png
"""
import json
import argparse
from pathlib import Path
# Need a prettier plotting styles
import matplotlib.pyplot as plt


class CurveReader(object):

    def __init__(self, filenames, legends=None):
        if isinstance(filenames, str):
            self._filenames = [filenames]
        elif isinstance(filenames, list):
            self._filenames = filenames
        if legends is not None:
            self._legends = {}
            if not isinstance(legends, list):
                legends = [legends]
            for _filename, _legend in zip(self._filenames, legends):
                self._legends[_filename] = _legend
        else:
            self._legends = None

        self._curves = {}
        self._load()

    def _load(self):
        for _filename in self._filenames:
            try:
                with open(_filename, 'r') as f:
                    time_series = json.load(f)
                    if self._legends is None:
                        curve_name = Path(_filename).stem
                    else:
                        curve_name = self._legends[_filename]
                    steps, accuracies = [], []
                    for timestep in time_series:
                        _, step, acc = timestep
                        steps.append(step)
                        accuracies.append(acc)
                    self._curves[curve_name] = {
                        'steps': steps,
                        'accuracy': accuracies,
                        }
            except:
                print('{} can not loaded'.format(_filename))

    @property
    def curves(self):
        return self._curves

class ExtendAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        items.extend(values)
        setattr(namespace, self.dest, items)

def main(args):

    json_files = args.json_files
    legends = args.legends

    if legends is not None:
        assert len(legends)==len(json_files), 'Length of legends and curves should be equal'

    if not json_files:
        raise ValueError('empty input')

    curve_reader = CurveReader(json_files, legends)

    for name, content in curve_reader.curves.items():
        plt.plot(content['steps'], content['accuracy'], label=name, linewidth=.8)
        print('Apex accuracy [{}] = {}'.format(name, max(content['accuracy'])))
    plt.title(args.plot_title)
    plt.legend()
    plt.savefig(args.output_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Plotting Training Curves from TensorBoard')
    parser.register('action', 'extend', ExtendAction)
    parser.add_argument('-f', '--json_files', nargs='+', action='extend',
                        help='Path to json files to be plotted.', required=True)
    parser.add_argument('-l', '--legends', nargs='+', action='extend',
                        help='Titles of each curves.', required=False)
    parser.add_argument('-of', '--output_file', type=str,
                        default='exported_curves.png', help='Path to exported folder.')
    parser.add_argument('-t', '--plot_title', type=str,
                        default='Evaluation Curve', help='String of plot title.')
    args = parser.parse_args()
    main(args)
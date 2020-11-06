import argparse

import numpy
import torch

from complex import ComplEx
from dataset import ALL_DATASET_NAMES, Dataset
from evaluation import Evaluator
from model import DIMENSION, INIT_SCALE

parser = argparse.ArgumentParser(
    description="Kelpie"
)

parser.add_argument('--dataset',
                    choices=ALL_DATASET_NAMES,
                    help="Dataset in {}".format(ALL_DATASET_NAMES)
)

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Embedding dimension"
)

parser.add_argument('--init_scale',
                    default=1e-3,
                    type=float,
                    help="Initial scale"
)

parser.add_argument('--learning_rate',
                    default=1e-1,
                    type=float,
                    help="Learning rate"
)

parser.add_argument('--load',
                    help="path to the model to load",
                    required=True)

parser.add_argument('--facts_file',
                    help="path to the file containing the list of facts to predict",
                    required=True)



def read_facts(facts_filepath, separator="\t"):

    facts = []
    with open(facts_filepath, "r") as facts_file:
        lines = facts_file.readlines()
        for line in lines:
            head, relation, tail = line.strip().split(separator)
            facts.append((head, relation, tail))
    return facts

args = parser.parse_args()

model_path = args.load

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

hyperparameters = {DIMENSION: args.dimension, INIT_SCALE: args.init_scale}
print("Initializing model...")
model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True)   # type: ComplEx
model.to('cuda')
model.load_state_dict(torch.load(model_path))
model.eval()

facts_to_test = read_facts(args.facts_file)
samples_to_test = numpy.array([dataset.fact_to_sample(x) for x in facts_to_test])
all_scores = model.all_scores(samples_to_test).detach()

with open("output.txt", "w") as outfile:
    dataset_entity_lines = [str(i)+"\t"+dataset.entity_id_2_name[i]+"\n" for i in range(dataset.num_entities)]
    dataset_relation_lines = [str(i)+"\t"+dataset.relation_id_2_name[i]+"\n" for i in range(dataset.num_direct_relations)]

    scores_lines = []
    for i in range(len(samples_to_test)):
        cur_sample = samples_to_test[i]
        (h_id, r_id, t_id) = cur_sample
        scores_lines.append(";".join([str(h_id), str(r_id), str(t_id)]) + "\t" + ";".join([str(x) for x in all_scores[i].cpu().numpy()]) + "\n")

    outfile.writelines(dataset_entity_lines + ["###\n"] + dataset_relation_lines + ["###\n"] + scores_lines)
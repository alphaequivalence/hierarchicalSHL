# Hierarchical Learning of Dependent Concepts for Human Activity Recognition

Code accompanying the paper

> [Hierarchical Learning of Dependent Concepts for Human Activity Recognition](/doc/296.pdf)

> [Supplementary material (containing the proof)](/doc/296_supplementary.pdf)

In multi-class classification tasks, like human activity recognition, it is often assumed that classes are separable. In real applications, this assumption becomes strong and generates inconsistencies. Besides, the most commonly used approach is to learn classes one-by-one against the others. This computational simplification principle introduces strong inductive biases on the learned theories. In fact, the natural connections among some classes, and not others, deserve to be taken into account. In this paper, we show that the organization of overlapping classes (multiple inheritances) into hierarchies considerably improves classification performances. This is particularly true in the case of activity recognition tasks featured in the Sussex-Huawei Locomotion (SHL) dataset.
After theoretically showing the exponential complexity of possible class hierarchies, we propose an approach based on transfer affinity among the classes to determine an optimal hierarchy for the learning process.
Extensive experiments on the SHL dataset show that our approach improves classification performances while reducing the number of examples needed to learn.



<p align="center">
    <img src="/img/proposed-hierarchical-approach.png" width="80%">
</p>
<p align="center">
Figure:     The framework of the proposed approach involving 3 major steps:
    (1) Concept similarity analysis:
    encoders are trained to output, for each source concept, an appropriate representation which is then fine-tuned in order to account for target concepts.
    Affinity scores are depicted by the arrows between concepts (the thicker the arrow, the higher the affinity score).
    (2) Hierarchy derivation:
    based on the obtained affinity scores, a hierarchy is derived using an agglomerative approach.
    (3) Hierarchy refinement:
    each non-leaf node of the derived hierarchy is assigned with a model that encompasses the most appropriate representation as well as an ERM which is optimized to separate the considered concepts.
    The process is repeated until convergence.
</p>


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Prerequisites
* `numpy`
* `TensorFlow`
* `nni` (https://github.com/microsoft/nni#installation)


If you are using `pip` package manager, you can simply install all requirements via the following command(s):

    python -m virtualenv .env -p python3 [optional]
    source .env/bin/activate [optional]
    pip3 install -r requirements.txt

### Installing
#### Get the dataset
1. You can get the preview of the SHL dataset (`zip` files) from [here](http://www.shl-dataset.org/activity-recognition-challenge-2020/). Make sure to put the downloaded files into `./data/` folder. Alternatively, you can run `scripts/get_data.sh` to download the dataset automatically.
2. Run `./scripts/extract_data.sh` script which will extract the dataset into `./generated/tmp/` folder.

## Running
### on your laptop:

    python cnn_split_channels.py

this will load the data and train the model defined inside `cnn_split_channels.py`. This same way, you can run other models.

### via Microsoft NNI

     nnictl create --config stjohns.yml --foreground

### on the computing platform Magi:

    /opt/slurm-19.05.1-2/bin/sbatch stjohns.slurm

information about the execution of the job can be found in `*.err` and `*.out` files, which output, respectively, messages sent to stderr and stdout streams.

## Results
Our model achieves the following performance:

<p align="center">
    <img src="/img/per-node-accuracy-improvement.png" width="80%">
</p>
<p align="center">
Figure: Per-node performance gains averaged over the entire derived architectures (similar nodes are grouped and their performances are averaged).
The appearance frequency of the nodes is also illustrated.
For each bar, the corresponding concepts can be found in the code repository. For example, the 8th bar corresponds to the concepts 2:{\it walk}-3:{\it run}-4:{\it bike} grouped.
</p>

| Method    | Agreement | Perf. avg.±std.  |
| ----------|-----------| ---------------  |
| Expertise | -         | 72.32±0.17       |
| Random    | 0.32      | 48.17±5.76       |
| Proposed  | 0.77    	| 75.92±1.13       |


<p align="center">
    <img src="/img/per-concept-accuracy-improvement.png" width="60%">
</p>
<p align="center">
Figure: Recognition performances of each individual concept averaged over the entire derived hierarchies.
For reference, the recognition performances of the baseline model are also shown.
</p>


<p align="center">
    <img src="/img/hierarchy-linerarly-separable.png" width="20%">
    <img src="/img/linerarly-separable.png" width="36%">
</p>
<p align="center">
Figure: Decision boundaries generated by the considered ERMs using two different encoders (representations) fine-tuned to account for (top right) a dissimilar concept (exhibiting a low-affinity score); (bottom right) a similar concept (exhibiting a high-affinity score).
</p>



<p align="center">
    <img src="/img/decision-boundaries.png" width="80%">
</p>
<p align="center">
Figure: Decision boundaries obtained by SVM-based classifiers trained on the representations $\mathcal{Z}_t$ as a function of the distance between the concepts (y-axis) and the supervision budget (x-axis).
</p>



<p align="center">
    <img src="/img/hierarchies.png" width="70%">
</p>
<p align="center">
Figure: Hierarchies:
(a) defined by domain expert
(b-c) derived using our approach
(d) randomly sampled.
</p>

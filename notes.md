# `CausalLearn` notes

## ML model strategy
We phrase the problem as a link prediction problem, where we start with a fully
connected graph with the correlation coefficients as edge features. We want to
then successively add causal edges to the graph, to end up with the target
CPDAG.

### Training
1. Start with the fully connected input graph, having both correlation- and
   causal edges.
2. Remove causal edges at random.
3. The model generates node features using a heterogeneous GNN, and from those
   node features it also generates edge features.
4. The model has to output probabilities for each node pair `(src, tgt)`, being
   the probability of there being a causal edge from `src` to `tgt`.

### Inference
1. From a fully connected input graph, having only correlation edges, we pass
   it to the model and get causal edge probabilities for all the edges.
2. We add the causal edge with the highest probability, as long as the
   probability is above a chosen threshold.
3. We pass the new graph, with correlation edges and the new causal edge, to
   the model again.
4. We continue (2)-(3) until all edge probabilities are below a chosen
   threshold.
5. We apply an algorithm that ensures that the output graph is in fact a DAG.

### Notes
- The removal of causal edges from the graph could perhaps just be done with
  some kind of edge dropout.
- We will initially just add 1-vectors as node features.

# One-Class Graph Autoencoder: A New, End-to-End, Low-Dimensional, and Interpretable Approach for Node Classification
- Marcos P. S. Gôlo (University of São Paulo) | marcosgolo@usp.br
- José G. B. de Medeiros Junior (University of São Paulo) | gilberto.barbosa@usp.br
- Diego F. Silva (University of Porto) | diegofsilva@icmc.usp.br
- Ricardo M. Marcacini (University of São Paulo) | ricardo.marcacini@icmc.usp.br

# Citing:
If you use any part of this code in your research, please cite it using the following BibTex entry
```latex
@article{ref:Golo2025,
    title = {One-class graph autoencoder: A new end-to-end, low-dimensional, and interpretable approach for node classification},
    journal = {Information Sciences},
    volume = {708},
    pages = {122060},
    year = {2025},
    issn = {0020-0255},
    doi = {https://doi.org/10.1016/j.ins.2025.122060},
    url = {https://www.sciencedirect.com/science/article/pii/S0020025525001926},
    author={G{\^o}lo, Marcos Paulo Silva and de Medeiros Junior, Jos{\'e} Gilberto Barbosa and Silva, Diego Furtado and Marcacini, Ricardo Marcondes},
```

# Abstract 
One-class learning (OCL) for graph neural networks (GNNs) comprises a set of techniques applied when real-world problems are modeled through graphs and have a single class of interest. These methods may employ a two-step strategy: first representing the graph and then classifying its nodes. End-to-end methods learn the node representations while classifying the nodes in OCL process. We highlight three main gaps in this literature: (i) non-customized representations for OCL; (ii) the lack of constraints on hypersphere learning; and (iii) the lack of interpretability. This paper presents One-cLass Graph Autoencoder (OLGA), a new OCL for GNN approach. OLGA is an end-to-end method that learns low-dimensional representations for nodes while encapsulating interest nodes through a proposed and new hypersphere loss function. Furthermore, OLGA combines this new hypersphere loss with the graph autoencoder reconstruction loss to improve model learning. The reconstruction loss is a constraint to the sole use of the hypersphere loss that can bias the model to encapsulate all nodes. Finally, our low-dimensional representation makes the OLGA interpretable since we can visualize the representation learning at each epoch. OLGA achieved state-of-the-art results and outperformed six other methods with statistical significance while maintaining the learning process interpretability with its low-dimensional representations.

# How to use || Replication of our results
```
python main.py --k [k=1/k=2/k=3] --h1 48 --h2 2 --radius 0.4 --lr 0.0001 --patience 300 --n-epochs 5000 --dataset [TUANDROMD/fakenews/food/musk/pneumonia/relevant_reviews/strawberry/terrorism]
```
 
# requirements
- networkx==2.6
- sklearn
- pandas
- numpy
- torch
- torch-cluster
- torch-geometric
- torch-scatter
- torch-sparse
- torch-spline-conv

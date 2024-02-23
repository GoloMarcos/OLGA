# One-Class Graph Autoencoder: A New, End-to-End, Low-Dimensional, and Interpretable Approach for Node Classification

# Example

python main.py --k [k=1/k=2/k=3] --h1 48 --h2 2 --radius 0.4 --lr 0.0001 --patience 300 --n-epochs 5000 --dataset [TUANDROMD/fakenews/food/musk/pneumonia/relevant_reviews/strawberry/terrorism]
 
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

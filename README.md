# One-Class Graph Autoencoder: A New, End-to-End, Low-Dimensional, and Interpretable Approach for Node Classification

# Example

python main.py --k [k=1/k=2/k=3] --h1 48 --h2 2 --radius 0.4 --lr 0.0001 --patience 300 --n-epochs 5000 --dataset [TUANDROMD/fakenews/food/musk/pneumonia/relevant_reviews/strawberry/terrorism]
 
# requirements
- networkx==2.6
- pandas==1.3.5
- scikit-learn==1.0.2
- torch==1.13.1
- torch-cluster==1.6.0
- torch-geometric==2.2.0
- torch-scatter==2.1.0
- torch-sparse==0.6.16
- torch-spline-conv==1.2.1 

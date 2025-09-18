# CnaHGNN: Co-Embedding of Nodes and Attributes With Hypergraph Neural Networks  

## 📖 Introduction
we propose **CnaHGNN** — a **node–attribute co-embedding framework** built upon hypergraph neural networks. CnaHGNN introduces both **bipartite graph** and **hypergraph structures** to capture cross-neighborhood dependencies without significantly increasing topological complexity. Attribute node embeddings from the bipartite graph are aggregated into node embeddings in the hypergraph, and a hypergraph neural network is used to learn **high-order representations**.  

During training, embeddings are refined through an **unsupervised link prediction task**, combined with a **reconstruction function** and an **adaptive dynamic adjustment strategy**, which improves both embedding **quality** and **robustness**.  

Extensive experiments on multiple real-world datasets demonstrate that CnaHGNN consistently outperforms state-of-the-art methods, highlighting its **effectiveness** and **generalizability**.  

## ⚙️ Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### Node Classification Task
```bash
python main_cf.py --dataset "PubMed"
```
### Link Prediction Task
```bash
python main_lp.py --dataset "PubMed"
```
# GraphSR: A Data Augmentation Algorithm for Imbalanced Node Classification


> **Abstract:** 
 Graph neural networks (GNNs) have achieved great success in node classification tasks. However, existing GNNs naturally bias towards the majority classes with more labelled data and ignore those minority classes with relatively few labelled ones. The traditional techniques often resort over-sampling methods, but they may cause overfitting problem. More recently, some works propose to synthesize additional nodes for minority classes from the labelled nodes, however, there is no any guarantee if those generated nodes really stand for the corresponding minority classes.  In fact, improperly synthesized nodes may result in insufficient generalization of the algorithm.
	To resolve the problem, in this paper we seek to automatically augment the minority classes from the massive unlabelled nodes of the graph. 
	Specifically, we propose \textit{GraphSR}, a novel self-training strategy to augment the minority classes with significant diversity of unlabelled nodes, which is based on a Similarity-based selection module and a Reinforcement Learning(RL) selection module.
	The first module finds a subset of unlabelled nodes which are most similar to those labelled minority nodes, and the second one further determines the representative and reliable nodes from the subset via RL technique. 
	Furthermore, the RL-based module can adaptively determine the sampling scale according to current training data.
	This strategy is general and can be easily combined with different GNNs models.
	Our experiments demonstrate the proposed approach outperforms the state-of-the-art baselines on various class-imbalanced datasets.
## Getting started
### Test

Run the following script to test the trained model:

```sh
python main.py
```

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{Zhou_Gong_2023, 
    title={GraphSR: A Data Augmentation Algorithm for Imbalanced Node Classification}, 
    volume={37}, url={https://ojs.aaai.org/index.php/AAAI/article/view/25622}, 
    DOI={10.1609/aaai.v37i4.25622}, 
    number={4}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Zhou, Mengting and Gong, Zhiguo}, 
    year={2023}, 
    month={Jun.}, 
    pages={4954-4962} 
}
```
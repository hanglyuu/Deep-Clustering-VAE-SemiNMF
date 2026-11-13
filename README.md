# Deep Clustering with VAE and Semi-NMF

Notebooks for my MSc thesis at TU/e: **Deep Clustering via Joint Variational Autoencoder and Non-negative Matrix Factorization with Proximal Gradient Optimization**.

The main idea is to learn the representation and the clustering together. A VAE encodes the data into a latent mean `mu`, and the matrix factorization fits `mu ≈ Y @ C`. `Y` contains the cluster assignments and is kept nonnegative; `C` contains the cluster centers and can have negative values. The binary update and Exclusive Lasso encourage clearer assignments in `Y`.

[Thesis](https://research.tue.nl/p/en/studenttheses/deep-clustering-via-joint-variational-autoencoder-and-non-negativ)

## Start here

Install the libraries in your Python environment:

```bash
pip install torch torchvision numpy scipy scikit-learn pandas matplotlib jupyter
jupyter notebook
```

Python 3.12 was used for the execution checks.

Open [mnist.ipynb](mnist.ipynb) and run the cells from top to bottom. In VS Code, open the project folder and select the Python environment you installed the libraries into.

The experiment settings are ordinary variables near the beginning of each notebook. Start with a small sample and a few epochs, then increase them for a longer run. MNIST and Fashion-MNIST download on the first run. PyTorch uses a GPU when available and otherwise runs on the CPU.

Keep all the notebooks in the same folder. They use `%run ./functions.ipynb` to load the shared definitions, so you do not need to run `functions.ipynb` separately or follow a numbered notebook sequence.

## Notebooks

| Notebook | What is in it |
|---|---|
| [mnist.ipynb](mnist.ipynb) | Main joint VAE–Semi-NMF experiment, fully connected and convolutional models, training curves and checkpoint/resume |
| [functions.ipynb](functions.ipynb) | Data loading, evaluation, AE/VAE models, matrix factorization, binary updates and training loops |
| [baselines.ipynb](baselines.ipynb) | k-means, GMM, NMF, AE/VAE followed by clustering, and MF on fixed embeddings |
| [deep_clustering.ipynb](deep_clustering.ipynb) | DEC/IDEC, DCN, ADMM clustering, VaDE, pseudo-label DeepCluster and joint VAE+k-means |
| [spectralnet.ipynb](spectralnet.ipynb) | SpectralNet with Euclidean and learned Siamese affinities |
| [synthetic.ipynb](synthetic.ipynb) | Circles, moons, blobs, unstructured, anisotropic and varied-variance data; original-space and latent-space comparisons, binary penalty curves |
| [matrix_factorization.ipynb](matrix_factorization.ipynb) | NMF/Semi-NMF optimizers and the earlier small matrix-factorization experiments |
| [analysis.ipynb](analysis.ipynb) | Latent plots, confusion matrices, generated digits, cluster-count experiments and ablations |
| [reuters_fashion.ipynb](reuters_fashion.ipynb) | Fashion-MNIST experiments and Reuters-10K preparation and experiments |

## Main training loop

The VAE is pretrained first. Then each minibatch has two updates:

1. Update `C` and the VAE, with `Y` fixed.
2. Run the VAE again, then update `Y` and the VAE, with `C` fixed.
3. Apply the selected Exclusive Lasso and binary updates to `Y`.

This keeps the two separate Adam optimizers from the original notebooks. Clustering uses `mu`; the decoder uses a sampled latent vector. KL warm-up continues from pretraining; MF warm-up starts when joint training begins. `restart_kl=True` restores the earlier KL restart experiment. Binary and exclusivity strengths can increase during joint training.

The notebooks report ACC, NMI, ARI, weighted F1 and purity. Labels are used for evaluation, not for the training loss. MNIST/Fashion-MNIST can use the combined training and test images for clustering; this is a transductive clustering experiment, not held-out classification.



## A few implementation details

- Standard PRIMP, the adaptive top-two update, and the older accumulated update are separate options. The adaptive update encourages near one-hot assignments; it does not guarantee exactly one active entry.
- The Exclusive Lasso shrinkage printed in the thesis is preserved for comparison with an exact proximal version. Applying Exclusive Lasso and PRIMP one after the other is not generally the exact proximal map of their combined penalty.
- Some old cells had issues with sample indexing, VAE sampling, k-means initialization or cluster-label matching. These are corrected rather than copied unchanged. In particular, unmatched clusters are no longer assigned to the majority true class when computing ACC/F1.

Reuters needs prepared data or the RCV1 preparation cells in `reuters_fashion.ipynb`. The original Reuters sample IDs and vocabulary were not saved, and the thesis does not specify its convolutional text preprocessing. The image CNN follows the image architecture. The previous GitHub square-padding text CNN is retained as an optional reconstructed experiment (`run_padded_text_conv`); its feature layout depends on vocabulary order and does not recover the historical Reuters convolutional result.

These notebooks contain the experiments and analysis needed to rerun the project. Short example runs check that the code works; they are not reproductions of the thesis scores. The thesis remains the record of the historical results.

Small CPU runs were checked for every notebook and all nine deep-clustering methods. The checks included real MNIST/Fashion-MNIST subsets, synthetic data, proximal updates and checkpoint continuation. Reuters preparation was checked with a small input fixture.

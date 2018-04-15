# Word Cluster Embeddings
Code for "Smaller Text Classifiers with Discriminative Cluster Embeddings" (NAACL 2018)

## Citation
If you use our code, please cite:

@inproceedings{chen2018smaller,
  author = {Mingda Chen and Kevin Gimpel},
  booktitle = {North American Association for Computational Linguistics (NAACL)},
  title = {Smaller Text Classifiers with Discriminative Cluster Embeddings},
  year = {2018},
}

## Dependencies

- Python 3.5
- TensorFlow 1.3
- NLTK (for tokenizing IMDB dataset)

## Prepare Data

You can download AG News, DBpedia, Yelp Review Full, Yelp Review Polarity from [here](http://goo.gl/JyCnZq) and IMDB data from [here](http://ai.stanford.edu/~amaas/data/sentiment/). Then run the corresponding data processing code to generate data file.

Note that in this code Yelp Review Full and Yelp Review Polarity were renamed to yelp-1 and yelp-2 respectively. If you want to use other names, please modify the code accordingly.

## Training

Use `run_se_ce_cae.py` for training standard embeddings, cluster embeddings or cluster adjustment embeddings. Use `run_me.py` for training mixture embeddings.
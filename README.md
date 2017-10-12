# Visualizing the Structure of Knowledge

## Motivation:
In any given field, I know so little that I may not even know what there is to know. It would be nice to be able to choose a topic and browse an interactive visualization of all the research being done on and related to that topic such that each research paper was represented by a point in space. Related papers would form clusters, which together would constitute a structure in space visually representing the structure of our knowledge. Reading about a subfield would amount to exploring a cluster. Rather than clicking on a series of links to figure out which related topics are important, I would be able to quickly gauge the popularity and relevance of other topics by the size and proximity of nearby clusters. 

## Question:
### _Can we easily reveal some underlying structure in the relationships between academic paper abstracts using a combination of TruncatedSVD and TSNE?_

## Execution:

### The Dataset

I originally planned to approach this task by interacting with the arXiv API, but I then came across the relatively new website called [_Semantic Scholar_](https://semanticscholar.org), which uses AI to analyze publically avaiable research papers and extract "authors, references, figures, and topics" from them. Since my project requires access to large amounts of data, I downloaded its entire "open research corpus," which contains information on over 7 million research papers. The download is a single massive (18 GB) JSON file, so I began by splitting the file up into more manageable pieces, each containing information on only about 100,000 papers.

### Embedding

I decided to compare paper abstracts by assuming that if the same words appear in multiple abstracts, those abstracts are considered similar. It's not a very sophisticated analysis—since (1) it doesn't take into account the fact that different words can be synomnyms of each other and (2) the context in which words are used can affect their meaning—but it should allow us to extract a shallow level of connection between papers.

To process each abstract, I first passed the text through a [Porter Stemmer](https://pythonprogramming.net/stemming-nltk-tutorial/), which is an algorithm from the late 1970s that prunes words into "stems" such that, for example, all the words in {"traveled", "traveling", "travel", "traveler"} are transformed into the stem "travel". This normalizes text to quench noise caused by differences in tense, plurality, etc. 

With that taken into account, I surveyed 10,000 abstracts drawn at random from one of my 100,000-paper files to tally up the number of appearances of each word stem found. From the resulting list of ~70,000 stems, I made an list of the 10,000 most commonly used stems (with "the" in first place at ~110,000 counts, and a host of seemingly random stems like "religion", "mileston", and "ductal" near last place at just 8 counts). These stems serve as the basis vectors which span the space into which we can embed each abstract. For each of another 5,000 abstracts randomly drawn from the file, I recorded the number of times each common stem appeared, and stored that number in the corresponding vector position, normalizing each paper-abstract vector such that its components summed to 1 (thus, the value of a component represents the fraction of the abstract constituted by the corresponding stem). This process results in a collection of 10,000-dimensional vectors, which, without any further processing, would be impossible to visualize. To do that, we need some way of mapping the high-dimensional data into an analog of physical space. For this project, I attempted to use a combination of Truncated SVD and t-SNE.

### Dimensionality Reduction

Though the [t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) algorithm tends to be very useful for representing high-dimensional data in 2- or 3-dimensional space while preserving local structure, it is quite computationally expensive (it minimizes the [_Kullback-Leibler divergence_](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), or _relative entropy_ between the high- and low-dimensional data distributions, which is nonlinear and pretty complicated). Before applying t-SNE, then, it helps to map to some lower-dimensional representation. For dense data, PCA works better, and for sparse data, Truncated SVD works better (see [here](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)). Since many of the components of each paper-abstract vector are 0, the data is "sparse", and Truncated SVD is the way to go. [Singular-value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular-value_decomposition) is essentially a generalization of eigenvalue decomposition, so Truncated SVD just keeps the most important vector directions and prunes the rest to project the data into a lower-dimensional space.

Thus, mapping from 10,000 to 128 dimensions with Truncated SVD, then from 128 to 2 dimensions with t-SNE, we end up with something that looks like a blob:

![image]

I've colored the points according to publication date (redder is older, bluer is newer)—clearly, t-SNE did not separate the data according to date. Without any other means of labeling, it is quite difficult to tell whether t-SNE has managed to successfully extract any structural information from the dataset. If we inspect the content of the points, however, we find that the small lobe near the top right consists of abstracts largely written in French. So the embedding and dimensionality-reduction did preserve _something_, it's just not clear at the moment the extent to which it did that something. Ideally, I would like to redo this analysis with papers which have their field of study labeled (e.g. "physics", "medicine", "mathematics", etc.), so that I can generate a more meaningful color scheme. Additionally, there are a number of adjustable parameters for each of these algorithms, on top of the fact that they are both stochastic (thus generating different results for each run anyway). t-SNE is also prone to getting stuck in local minima, so the result we obtained here may not be optimal. 

## The Future:

It may be worthwhile to approach this visualization project with a better initial embedding model, since this frequency-based embedding may not be capturing enough of the meaning of each abstract. To address the problem of synonyms and context (see (1) and (2) in _Embedding_ above), we could try training some form of Recurrent Neural Network (RNN) to learn relationships between words, and generalize this so learn relationships between abstracts. Again, the initial high-dimensional embedding would be reduced using t-SNE at some point. 

I wonder, however, whether t_SNE is even capable of capturing the level of structure I desire. The algorithm is sensitive to distance between nearby points, but because of its use of Gaussian kernels, past a certain range, it doesn't care whether a point is "far away" or "really far away". This might limit the model's ability to represent the fact that two very different subjects are in fact very different from each other. On the other hand, maybe only the topology of the cluster network matters, since disparity between subjects may be inferrable from distance _along_ a chain of clusters.

Rather than creating embeddings based on the text in abstracts, it may also be useful to visualize structure using a graph-based approach where edges between nodes are formed by paper citations. Using the Semantic Scholar resources which automatically identify how influential a source is on the work of the paper citing the source, we could assign edge weights based both on number of citations and importance of each citation. The result might look something like [what Chris Olah did with fanfiction.net](http://colah.github.io/posts/2014-07-FFN-Graphs-Vis/). 

[image]: https://raw.githubusercontent.com/chrismbryant/arXiv-structure/master/images/papers-2017-02-21-70-p_2.png 

# anomaly_detection_persistent_homology
Detecting anomalous texts with persistent homology of context vectors computed by individual attention heads in language transformers

## Methods

1. Compute the context vectors of some input text for s pecific `head` in a specific `layer` of a language model.
2. Compute the pairwise Euclidean distances between them.
3. Use the distance matrix to compute persistent homology and a persistence diagram for the text.
4. Do this for multiple baseline texts on the same topic.
5. Compute the Fréchet mean of the baseline texts persistence diagrams.
6. Find outliers (and potentially remove those with large Wasserstein distance from the Fréchet mean). 
7. Compute persistent homology and persistence diagrams for new potentially anomalous texts.
8. Compute the Wasserstein distances between the potentially anomalous texts persistence diagrams and the Fréchet mean of the baseline texts persistence diagrams.
9. Find outlier Wasserstein distances and classify the corresponding texts as anomalous. 

## Some Heuristics and Guiding Principles

We must note that this form of anomalous text detection is not perfect, and as seen in the notebooks, there are false positive and false negatives. There are some things we should keep in mind while using this methods of anomaly detection. The length of the text matters. If one of the texts is significantly larger that the others in terms of token count, then it has a higher chance of being labeled an anomaly, regardless of whether it is on the same topic as the baseline texts. The next thing to keep in mind is that if the baseline texts too loosely clustered around the Fréchet mean, this makes detecting outliers (anomalous text) more difficult. This can happen if the topics mentioned in the baseline texts are only loosely related in content. For example, if we have baseline text talking more extensively about the applications of deep learning to healthcare, with an emphasis on healthcare applications, this will likely be considered an outlier in the initial calculation of the Féchet mean. This also leaves us open to the possibility of *not detecting* a text about healthcare instead of deep learning as anomalous. We must also note that some models perform better at this than others. For example, `xlm-roberta-large` forms better persistent homology features than `xlm-roberta-base` on average. We must also take note of the fact that certain heads may perform better than others for certain topic classes as well. This is an interesting feature of this analysis that is as much about anomaly detection as it is about analyzing the topics modeled by individual heads of the model. 

## Anomaly Detection with Persistent Homology of Hidden States

We also perform the same persistent homology analysis with layer outputs (hidden states) of texts, forming persistent diagrams of several baseline texts, computing their Fréchet mean persistence diagram, and then comparing some new potentially anomalous texts to the Fréchet mean of the baseline texts. This is done in [this notebook](https://github.com/Amelie-Schreiber/anomaly_detection_persistent_homology/blob/main/anomaly_detection_xlm_roberta_large_english_layer_outputs.ipynb) for example. 

## Next Steps

The next obvious thing to do would be to perform this analysis on all attention heads, and then determine what percentage of them classifies a given text as anomalous. This can help us better understand both the information that the individual attention heads are capturing, as well as get a better determination of whether a text is anomalous. Unfortunately, due to computational constraints this will have to wait until later. 

# paraphrase-detection-ml
Duplicate questions are common on large online discussion platforms (e.g., Quora),
fragmenting information and degrading search quality. We study paraphrase iden
tification on the Quora Question Pairs (QQP) benchmark using Sentence-BERT
(SBERT) as an efficient Siamese baseline. We then propose an interaction-aware
variant (CA-SBERT)that inserts a Cross-Attention module before pooling, allowing
token-level alignment across the two questions while preserving a shared-encoder
structure. Due to computational constraints, we train and tune both models on
a 15,000-pair subset and perform Bayesian hyperparameter optimization using
validation F1 as the objective. Our results show that CA-SBERT matches but does
not consistently outperform the SBERT baseline under limited data, suggesting that
the added capacity increases optimization sensitivity and may require larger-scale
training to yield clear gains.

You can find the complete pdf report in the repository

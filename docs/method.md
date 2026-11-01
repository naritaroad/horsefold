# Method

## Question

When several samples come from the same subject, how much can two common mistakes inflate a reported test accuracy?

1. **Unit-level split.** Samples are split at random, so samples from one subject land in both the training and the test set.
2. **Preprocessing before the split.** Feature selection and scaling are fitted on all samples, test samples included.

The case study is Basran et al. (2022), who trained classifiers on 102 μCT radiomic features of 129 proximal sesamoid bones from 50 Thoroughbred racehorses. The paper states a 90/39 bone split and a 10-fold cross-validation, but not whether bones of one horse were kept together, or whether feature selection used only training data. This project does not claim the paper leaked. It shows what each choice would do to the numbers on a synthetic dataset with the same layout, where the truth is known.

## Layout

From the paper, p.3 and Figure 1:

| | Horses | Limbs | Bones |
|---|---|---|---|
| Cases | 30 | 30 (one limb each) | 60 (medial and lateral) |
| Controls | 20 | 35 (15 horses with two limbs, 5 with one) | 69 (one lateral bone image corrupted) |
| Total | 50 | 65 | 129 |

The paper does not say which control horse lost the corrupted bone. The main analysis puts it on a two-limb horse; E4 moves it to a one-limb horse.

## Generative model

For bone *i* in limb *l* of horse *h*, feature *j*:

```
x_ij = a_hj + b_lj + e_ij + d · y_i · [j ≤ k]
a_hj ~ N(0, r·s),  b_lj ~ N(0, r·(1−s)),  e_ij ~ N(0, 1−r)
```

- Total variance is 1. Two bones of the same limb correlate at *r*; two bones of the same horse but different limbs at *r·s*. Main analysis: *r* = 0.7, *s* = 5/7, so the cross-limb correlation is 0.5.
- *d* is the true signal in standard deviations on the first *k* = 5 of *p* = 102 features. *d* = 0 means the features say nothing about the label, so any accuracy above chance is an artefact.
- The values of *r*, *s* and *d* are assumptions; the paper does not report within-horse correlation. E2 and E4 vary them.

## Pipeline

Following the paper (pp.4–6, Table 2):

1. Select the top *N* = 20 features. The paper ranks features by out-of-bag importance of 100 trees; this project uses the absolute Welch *t* statistic for speed.
2. Z-score the selected features.
3. Fit six classifiers configured after Table 2: logistic regression (no penalty), quadratic SVM, KNN with 10 neighbours, bagged trees (30), RUSBoost (30 trees, 21 leaves, learning rate 0.1) and a one-layer neural network (25 units). A 1-nearest-neighbour model is added as an extreme case that only memorises neighbours.

Steps 1 and 2 are fitted either on the training rows only (**clean**) or on all 129 rows (**leaky**).

## Experiments

| ID | What varies | Replicates | Output |
|---|---|---|---|
| E1 | Split (bone / horse) × preprocessing (clean / leaky) × 7 models, at *d* = 0 and *d* = 1.0 | 500 | Main comparison, figure F1 |
| E2 | *r* ∈ {0, 0.3, 0.5, 0.7, 0.9} at *d* = 0; *r* = 0 is the negative control | 300 | Figure F2 |
| E3 | *d* from 0 to 1.5 at *r* = 0.7 | 300 | Figure F3 |
| E4 | Missing bone on a one-limb horse; limb-dominant correlation (*s* = 0.3); medial/lateral offset | 300 | Sensitivity |
| E5 | 10-fold CV by bone (`KFold`) vs by horse (`GroupKFold`), preprocessing inside folds | 200 | Cross-validation counterpart |
| E6 | Horse-grouped 10-fold CV over 6 models × *N* ∈ {3, 5, 10, 20}: mean of 24 configurations vs the best one | 200 | Optimism of picking the best |

Each replicate draws a new feature matrix and a new split. Summaries are the mean accuracy, its Monte Carlo standard error (SD / √replicates), the 2.5th and 97.5th percentiles of single-split accuracy, balanced accuracy and AUC.

`d = 1.0` was chosen so that horse-grouped accuracy lands near 0.70, the range the paper reports; it is a calibration for illustration, not an estimate.

### Leak rate

Under a random 90/39 bone split, a test bone from a horse with *m* bones has no same-horse bone in training only if the other *m* − 1 bones are all in the test set:

```
P(leak) = 1 − C(38, m−1) / C(128, m−1)
```

Averaged over the 129 bones this is 0.826; `horsefold leak-rate` checks it against 20,000 simulated splits.

## Limitations

- Synthetic features. The size of the inflation depends on *r*, which is unknown for the real bones.
- A *t* filter replaces the paper's out-of-bag importance ranking, and scikit-learn replaces MATLAB's Classification Learner; hyperparameters are matched where Table 2 gives them.
- One structure is built in. Other layouts work through `Structure.from_groups`, but the E1–E6 suite is specific to this case.
- Accuracy is reported because the paper reports it. With 60 cases and 69 controls, a classifier that always predicts the majority class scores 0.53 on average, and a no-signal classifier scored by group split can fall below 0.5 because whole horses move between sets and shift the class balance. Balanced accuracy and AUC are in the JSON.

## References

- Ambroise, C., & McLachlan, G. J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS, 99*(10), 6562–6566. https://doi.org/10.1073/pnas.102102699
- Basran, P. S., et al. (2022). Radiomics modeling of catastrophic proximal sesamoid bone fractures in Thoroughbred racehorses using μCT. *Animals, 12*(21), 3033. https://doi.org/10.3390/ani12213033 (free full text: [PMC9658779](https://pmc.ncbi.nlm.nih.gov/articles/PMC9658779/))
- Demircioğlu, A. (2021). Measuring the bias of incorrect application of feature selection when using cross-validation in radiomics. *Insights into Imaging, 12*, 172. https://doi.org/10.1186/s13244-021-01115-1
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), 100804. https://doi.org/10.1016/j.patter.2023.100804
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience, 6*(5), gix019. https://doi.org/10.1093/gigascience/gix019

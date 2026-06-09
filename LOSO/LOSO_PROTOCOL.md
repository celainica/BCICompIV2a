# Protocol Specification: Modern Zero-Target LOSO on BCI Competition IV 2a

## Dataset

We use **BCI Competition IV Dataset 2a**. The dataset contains 9 subjects performing four motor imagery classes: left hand, right hand, both feet, and tongue. Each subject has two sessions recorded on different days. Each session contains 288 trials. The EEG montage contains 22 EEG channels; EOG channels are not used in this protocol. The original sampling rate is 250 Hz.

In MOABB, the two sessions are exposed as:

```text
0train
1test
```

We use these names throughout this document.

## Definition of This LOSO

For each target subject \(s_t\), one fold is constructed:

```text
Target subject:
  0train: not used
  1test : test set

Source subjects:
  0train: train/validation pool
  1test : not used
```

The target subject is never used before final evaluation. This includes supervised training, validation, early stopping, normalization, model selection, and hyperparameter tuning.

With 9 subjects, the protocol produces 9 folds. The reported score is the mean over the 9 held-out target subjects.

## Source Train/Validation Split

For a target subject \(s_t\), the source set contains the other 8 subjects. We split the source subjects, not trials, into training and validation subsets:

```text
7 source subjects -> training set
1 source subject  -> validation set
```

The validation subject is selected deterministically with a fixed seed:

```text
seed = 2026 + target_subject_id
```

This gives approximately:

```text
train: 7 x 288 = 2016 trials
val  : 1 x 288 = 288 trials
test : 1 x 288 = 288 trials
```

The validation set may be used for early stopping and model checkpoint selection only. It must not be merged with the training set after the checkpoint has been selected unless the entire model-selection procedure is repeated without target access and explicitly reported.

## Preprocessing

The default preprocessing for the EEGNet baseline is:

```text
Channels     : 22 EEG channels
Classes      : left_hand, right_hand, feet, tongue
Band-pass    : 4-40 Hz
Time window  : 0.0-4.0 s
Sampling rate: 250 Hz
```

The HA-FuseNet-lite exploratory script currently uses:

```text
Band-pass    : 4-40 Hz
Time window  : 1.0-4.0 s
Sampling rate: 250 Hz
```

Different temporal windows are considered preprocessing ablations. Main-table comparisons should use the same preprocessing unless the window is explicitly listed as an experimental factor.

## Normalization

Normalization is performed independently for each fold. The channel-wise mean and standard deviation are computed from the training set only:

```text
mean = train.mean(axis=(trials, time))
std  = train.std(axis=(trials, time))
```

The same training-set statistics are then applied to training, validation, and test data:

```text
train = (train - mean) / std
val   = (val   - mean) / std
test  = (test  - mean) / std
```

Validation and test samples must not contribute to these statistics. Target-subject statistics are not permitted.

## Disallowed Procedures

The following procedures violate Our LOSO:

- using the target subject's `0train` data for training, validation, early stopping, fine-tuning, or hyperparameter selection;
- using the target subject's `1test` data for anything other than final scoring;
- using unlabeled target-subject data for domain adaptation, batch-normalization adaptation, test-time adaptation, or normalization;
- computing preprocessing statistics on all subjects or all sessions;
- using source subjects' `1test` sessions for training or validation;
- selecting hyperparameters by repeatedly checking target-subject test accuracy;
- reporting the best target fold after multiple target-aware runs.

If any of these procedures are used, the experiment must be reported under a different setting, such as target calibration, unsupervised domain adaptation, test-time adaptation, or transductive evaluation.

## Evaluation Metrics

For each fold, we report:

```text
test accuracy
test balanced accuracy
best validation accuracy
best epoch
```

The main aggregate metrics are:

```text
mean test accuracy over 9 subjects
standard deviation over 9 subjects
mean test balanced accuracy over 9 subjects
standard deviation of balanced accuracy over 9 subjects
```

Because BCI Competition IV 2a is class-balanced in the official sessions, accuracy and balanced accuracy are usually identical or very close. Balanced accuracy is still reported for robustness.

## Hyperparameter Tuning Policy

Hyperparameters may be selected only by source-domain information. Acceptable options include:

1. fixing hyperparameters before running the 9 target folds;
2. selecting hyperparameters using only the source training/validation split inside each fold;
3. selecting hyperparameters on a separate development dataset or a separately declared subset that excludes the final target test folds.

The target subject's `1test` score must be computed once per declared model configuration. If a grid search is performed and target scores are inspected for all candidates, the result is not a valid Our LOSO estimate.

## Comparison With Other Protocols

### Lawhern-Style Cross-Subject Split

Lawhern et al. introduced EEGNet and evaluated cross-subject SMR decoding using a different split. In the Lawhern-style BCI2a setting used earlier in this project, each target subject is evaluated using:

```text
train: 5 source subjects' training sessions
val  : 3 source subjects' training sessions
test : target subject's test session
repeats: 10 per target subject
folds: 90 total
```

This is a valid zero-target protocol, but it is not the same as Our LOSO. It uses fewer source subjects for training and repeats random 5/3 source splits. Results from Lawhern-style evaluation should not be placed in the same table as Our LOSO results unless the protocol column explicitly distinguishes them.

### Strict LOSO Over All Sessions

The previous `strict_loso_all_sessions` setting used:

```text
train/val: all sessions from non-target subjects
test     : all sessions from the target subject
```

This is subject-independent, but it does not respect the official train/evaluation session distinction in BCI2a. It also evaluates on the target subject's `0train` session, which Our LOSO intentionally excludes.

### Target Calibration and Domain Adaptation

Any method that uses target-subject data before final scoring is not zero-target under this protocol. This includes supervised fine-tuning on `0train`, unsupervised adaptation on target data, and test-time normalization using target data. Such experiments may be useful, but they must be reported separately.

## Current Project Scripts

### EEGNet Baseline

```bash
python EEGNet_BCI2A_MODERN_LOSO.py
```

Output directory:

```text
/root/autodl-tmp/EEG/results/eegnet_bci2a_modern_loso/
```

Current result:

```text
mean_test_acc = 0.4900
std_test_acc  = 0.1326
```

### HA-FuseNet-Lite

```bash
python HAFuseNet_BCI2A_MODERN_LOSO.py
```

Output directory:

```text
/root/autodl-tmp/EEG/results/hafusenet_bci2a_modern_loso/
```

This script is an independent lightweight implementation inspired by HA-FuseNet modules. It is not the official HA-FuseNet code.

## Recommended Reporting Template

```text
Dataset: BCI Competition IV Dataset 2a
Protocol: Our LOSO, zero-target cross-subject
Target data used before evaluation: none
Train set: 7 source subjects, 0train only
Validation set: 1 source subject, 0train only
Test set: target subject, 1test only
Source test sessions used: no
Normalization: training-set statistics only, per fold
Metric: mean accuracy over 9 held-out target subjects
```

Recommended result table:

```text
model | window | mean_acc | std_acc | mean_balanced_acc | std_balanced_acc | notes
```

## References

[1] C. Brunner, R. Leeb, G. R. Müller-Putz, A. Schlögl, and G. Pfurtscheller, “BCI Competition 2008 - Graz data set A,” BCI Competition IV Dataset 2a description. https://bbci.de/competition/iv/desc_2a.pdf

[2] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, “EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces,” Journal of Neural Engineering, 2018. https://arxiv.org/abs/1611.08024

[3] Y. Zhu et al., “A study of motor imagery EEG classification based on feature fusion and attentional mechanisms,” Frontiers in Human Neuroscience, 2025. https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2025.1611229/full


# TCAV4OCT
Robust and Interpretable CNNs for Glaucoma Detection from OCT Images

Code and Scripts for robust end-to-end deep learning models and interpretabilty analysis using Testing with Concept Activation Vectors (TCAVs, Been Kim, et al.) and corroboration with expert eye tracking, as described in paper: "Robust and Interpretable Convolutional Neural Networks to Detect Glaucoma in Optical Coherence Tomography Images."

by Kaveri A. Thakoor, Sharath Koorathota, Donald C. Hood, and Paul Sajda

src: 
1. end2endDLModels: contains jupyter notebooks/python code for robust end-to-end deep learning models fine-tuned on OCT data
2. eyeTracking: contains script for generating expert eye fixation heatmaps superimposed on OCT reports and for computing fixation density per Area of Interest, contains modified plotting toolkit (PyGazeAnalyzer)
3. TCAVRandomConcepts10: scripts for TCAV interpretability analysis using 10 random concepts
4. TCAVRandomConcepts160: scripts for TCAV interpretability analysis using 160 random concepts

models:
Info for accessing saved models available on IEEE DataPort

results:
Results file for 160 random concepts TCAV experiment

doc:
Info on 150 ImageNet concepts used for TCAV experiment, additional OCT concepts and hand-selected random concepts listed in scripts under src/TCAVRandomConcepts10


References (and more in paper under review):

1. F. Chollet, Deep Learning with Python. Manning Publications, 2018.
2. B. Kim, M. Wattenberg, J. Gilmer, C. Cai, J. Wexler, F. Viegas, and R. Sayres. "Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV)", International Conference on Machine Learning, 2017.
3. E.S. Dalmaijer, S. Mathôt, and S. Van der Stigchel, "PyGaze: An open-source, cross-platform toolbox for minimal-effort programming of eyetracking experiments." Behavior research methods, 46(4), pp.913-921, 2014.
4. J. Deng, W. Dong, R. Socher, L. Li, K. Li, F. Li. "Imagenet: A large-scale hierarchical image database." IEEE Conference on Computer Vision and Pattern Recognition, pp. 248-255, 2009.


Enhancing Feature Distance with a Siamese Architecture for Video Anomaly Detection


## Authors


Xi Luo , Shifeng Li , Xiaoru Liu , Guolong Li Corresponding author: Shifeng Li limax_2008@outlook.com
## Description
Traditional autoencoders (AEs) in video anomaly detection (VAD) are typically trained on normal data only, limiting their ability to distinguish normal from abnormal samples and potentially reconstructing abnormal data effectively.  To address this challenge, we propose an unsupervised learning framework that integrates Siamese network architectures with pseudo-anomaly generation to explicitly model feature differences in data distributions.
   First, Noise injection and frame skipping are employed to generate anomaly samples, enabling the model to learn robust, discriminative features without relying on labeled anomalies.
   Second, a Siamese network structure is adopted to enforce a latent feature space where normal and abnormal samples are maximally separated. Furthermore, we incorporate the Kullback-Leibler Divergence (KLD) as a regularation term to explicitly keep the  normal and pseudo-anomaly distributions away from each other.	Experimental results on the public datasets demonstrate the effectiveness of the proposed method.
## Training/Inference
Train and evaluate the model in file "T3.py".

```bash
python T3.py
```
## Datasets
To validate and benchmark our method against the state of the art, we conduct experiments on several diverse datasets:[ UCSD Ped2](http://www.svcl.ucsd.edu/projects/anomaly), [ CUHK Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
# ViPMM
### Implementation for "Enhancing Medical Visual Representation Learning with a Knowledge-augmented Multimodal Pre-trained Model"

# Requirements
- Python (>=3.5)
- torch (>=1.1.0)
- [transformers (>=2.3.0)](https://github.com/huggingface/transformers)

# Get Required Data
- [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html)
# Data Preprocessing
```
# For Flickr30K
cd datasets
python split_flickr_data.py


# ViPMM Pretraining
python vipmm_pretraining.py --cfg cfg/pretrain-flickr-resnet.yml
```

# For SNLI
```
python unsupervised_nli.py --cfg cfg/unsupervised/snli.yml
python snli_unsupervised.py --data_folder ViPMM/unsupervised/flickr-resnet/snli
```
# For RTE
```
python unsupervised_nli.py --cfg cfg/unsupervised/rte.yml
python snli_unsupervised.py --data_folder ViPMM/unsupervised/flickr-resnet/rte
```
# For QNLI
```
python unsupervised_nli.py --cfg cfg/unsupervised/qnli.yml
python snli_unsupervised.py --data_folder ViPMM/unsupervised/flickr-resnet/qnli
```
# For MNLI
```
python unsupervised_nli.py --cfg cfg/unsupervised/mnli.yml
python snli_unsupervised.py --data_folder ViPMM/unsupervised/flickr-resnet/mnli
```
# For MNLI-mm
```
python unsupervised_nli.py --cfg cfg/unsupervised/mnli-mm.yml
python snli_unsupervised.py --data_folder ViPMM/unsupervised/flickr-resnet/mnli-mm
```
# For MRPC
```
python unsupervised_nli.py --cfg cfg/unsupervised/mrpc.yml
python snli_unsupervised.py --data_folder ViPMM/unsupervised/flickr-resnet/mrpc
```
# For QQP
```
python unsupervised_nli.py --cfg cfg/unsupervised/qqp.yml
python snli_unsupervised.py --data_folder ViPMM/unsupervised/flickr-resnet/qqp
```
# For QQP
```
python unsupervised_nli.py --cfg cfg/unsupervised/qqp.yml
python snli_unsupervised.py --data_folder ViPMM/unsupervised/flickr-resnet/qqp
```


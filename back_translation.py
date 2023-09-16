import pandas as pd
import torch
# Round-trip translations between English and German:
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
f = open('0.txt', 'r')
f1 = open('1.txt', 'a')
lines = f.readlines()
i = 1
for line in lines:
    paraphrase = de2en.translate(en2de.translate(line))
    print(i)
    i+=1
    f1.write('\n'+paraphrase)

# # Compare the results with English-Russian round-trip translation:
# en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
# ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')
#
# paraphrase = ru2en.translate(en2ru.translate('PyTorch Hub is an awesome interface!'))
# assert paraphrase == 'PyTorch is a great interface!'
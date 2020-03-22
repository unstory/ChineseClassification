# ChineseClassification
summary of chinese classification.

##### 1. TextCNN文本分类
1.1 先下载300维的[知乎词向量](https://github.com/Embedding/Chinese-Word-Vectors)，放在项目根目录，执行utils.py生成需要用到的词向量和存放模型的models目录

1.2 去掉TextCNN.py的main函数里面的train前面的注释

1.3 执行TextCNN.py

```
python TextCNN.py
```
##### note:
大部分代码参考[中文文本分类](https://github.com/649453932/Chinese-Text-Classification-Pytorch),不同的地方在于使用pytorch的dataloader和使用结巴分词，最终的acc为91.95%
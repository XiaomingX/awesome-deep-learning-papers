# 超赞 - 被引用最多的深度学习论文

这是2012到2016年间被引用最多的深度学习论文的精选列表。

我们认为，有一些深度学习论文是*经典之作*，不论它们的应用领域如何，都值得阅读。与其提供大量的论文列表，我们更愿意提供一份经过精心挑选的深度学习论文清单，这些论文被认为是某些研究领域中的*必读书单*。

## 背景

在这个列表之前，已经有一些*深度学习精选清单*，例如[深度视觉（Deep Vision）](https://github.com/kjw0612/awesome-deep-vision)和[深度学习递归神经网络（Awesome Recurrent Neural Networks）](https://github.com/kjw0612/awesome-rnn)。此外，在这个清单发布后，另一个针对深度学习初学者的优秀清单——[深度学习论文阅读路线图（Deep Learning Papers Reading Roadmap）](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap)也问世，并受到了许多深度学习研究者的喜爱。

尽管*路线图清单*包含了大量重要的深度学习论文，但对我来说，一口气阅读完这些论文有些让人不堪重负。正如我在介绍中提到的，我相信一些具有里程碑意义的论文，无论它们的应用领域是什么，都能给我们带来宝贵的启示。因此，我希望在这里介绍**100篇顶级深度学习论文**，作为了解深度学习研究的一个良好起点。

## 极好的文献推荐标准

1. 我们建议列出**2012年至2016年间发布的100篇深度学习顶级论文**。
2. 如果某篇论文被加入到列表中，通常需要从“2016年其他论文”部分中移除一篇论文，以保持列表中的100篇论文（因此，移除论文也是一种重要的贡献）。
3. 一些重要的论文，虽然未被纳入前100篇，但仍然值得一提，它们将列在“超越前100篇”部分。
4. 请参考“新论文”和“旧论文”部分，了解过去6个月内发布的论文或2012年之前的论文。

*(引用标准)*
- **< 6个月**：新论文（根据讨论决定）
- **2016年**：+60次引用或“2016年其他论文”
- **2015年**：+200次引用
- **2014年**：+400次引用
- **2013年**：+600次引用
- **2012年**：+800次引用
- **~2012年**：旧论文（根据讨论决定）


如果您有任何建议（如遗漏的论文、新论文、关键研究人员或拼写错误），请随时编辑并提交Pull Request。（请阅读[贡献指南](https://github.com/terryum/awesome-deep-learning-papers/blob/master/Contributing.md)，尽管只告知我们论文标题也能为我们做出巨大贡献。）

（更新）您可以通过[这个链接](https://github.com/terryum/awesome-deep-learning-papers/blob/master/fetch_papers.py)下载所有前100篇论文，并通过[这个链接](https://github.com/terryum/awesome-deep-learning-papers/blob/master/get_authors.py)收集所有作者的姓名。此外，[bib文件](https://github.com/terryum/awesome-deep-learning-papers/blob/master/top100papers.bib)也可供下载。感谢doodhwala、[Sven](https://github.com/sunshinemyson) 和 [grepinsight](https://github.com/grepinsight)！

+ 有没有人能贡献代码，统计前100篇论文作者的相关数据？

## 目录

* [理解 / 泛化 / 迁移](#understanding--generalization--transfer)
* [优化 / 训练技术](#optimization--training-techniques)
* [无监督 / 生成模型](#unsupervised--generative-models)
* [卷积神经网络模型](#convolutional-neural-network-models)
* [图像分割 / 物体检测](#image-segmentation--object-detection)
* [图像 / 视频 / 其他](#image--video--etc)
* [自然语言处理 / RNN](#natural-language-processing--rnns)
* [语音 / 其他领域](#speech--other-domain)
* [强化学习 / 机器人学](#reinforcement-learning--robotics)
* [2016年其他论文](#more-papers-from-2016)

*(超越前100篇)*

* [新论文](#new-papers)：少于6个月
* [旧论文](#old-papers)：2012年之前
* [硬件 / 软件 / 数据集](#hw--sw--dataset)：技术报告
* [书籍 / 调查 / 综述](#book--survey--review)
* [视频讲座 / 教程 / 博客](#video-lectures--tutorials--blogs)
* [附录：超越前100篇](#appendix-more-than-top-100)：未列入的更多论文

---

### 理解 / 泛化 / 迁移
- **神经网络中的知识蒸馏**（2015），G. Hinton等人 [[pdf]](http://arxiv.org/pdf/1503.02531)
- **深度神经网络容易被欺骗：对不可识别图像的高置信度预测**（2015），A. Nguyen等人 [[pdf]](http://arxiv.org/pdf/1412.1897)
- **深度神经网络中的特征迁移能力有多强？**（2014），J. Yosinski等人 [[pdf]](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)
- **CNN特征现成使用：一个惊人的基础线用于识别**（2014），A. Razavian等人 [[pdf]](http://www.cv-foundation.org//openaccess/content_cvpr_workshops_2014/W15/papers/Razavian_CNN_Features_Off-the-Shelf_2014_CVPR_paper.pdf)

### 优化 / 训练技术
- **训练非常深的网络**（2015），R. Srivastava等人 [[pdf]](http://papers.nips.cc/paper/5850-training-very-deep-networks.pdf)
- **批归一化：通过减少内部协方差偏移来加速深度网络训练**（2015），S. Loffe和C. Szegedy [[pdf]](http://arxiv.org/pdf/1502.03167)
- **深入探讨线性整流函数：在ImageNet分类上超越人类水平**（2015），K. He等人 [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)

### 无监督 / 生成模型
- **像素递归神经网络**（2016），A. Oord等人 [[pdf]](http://arxiv.org/pdf/1601.06759v2.pdf)
- **改进的生成对抗网络训练技术**（2016），T. Salimans等人 [[pdf]](http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf)

### 卷积神经网络模型
- **重新思考计算机视觉的Inception架构**（2016），C. Szegedy等人 [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
- **Inception-v4、Inception-ResNet及残差连接对学习的影响**（2016），C. Szegedy等人 [[pdf]](http://arxiv.org/pdf/1602.07261)

### 图像分割 / 物体检测
- **你只看一次：统一的实时物体检测**（2016），J. Redmon等人 [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)
- **全卷积网络用于语义分割**（2015），J. Long等人 [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)

### 图像 / 视频 / 其他
- **基于深度卷积网络的图像超分辨率**（2016），C. Dong等人 [[pdf]](https://arxiv.org/pdf/1501.00092v3.pdf)
- **艺术风格的神经算法**（2015），L. Gatys等人 [[pdf]](https://arxiv.org/pdf/1508.06576)

### 自然语言处理 / RNN
- **命名实体识别的神经架构**（2016），G. Lample等人 [[pdf]](http://aclweb.org/anthology/N/N16/N16-1030.pdf)
- **探索语言模型的极限**（2016），R. Jozefowicz等人 [[pdf]](http://arxiv.org/pdf/1602.02410)

### 语音 / 其他领域
- **端到端基于注意力的大词汇量语音识别**（2016），D. Bahdanau 等人 [[pdf]](https://arxiv.org/pdf/1508.04395)
- **深度语音 2：英语和普通话的端到端语音识别**（2015），D. Amodei 等人 [[pdf]](https://arxiv.org/pdf/1512.02595)
- **使用深度递归神经网络的语音识别**（2013），A. Graves [[pdf]](http://arxiv.org/pdf/1303.5778.pdf)
- **用于语音识别的深度神经网络的声学建模：四个研究小组的共同观点**（2012），G. Hinton 等人 [[pdf]](http://www.cs.toronto.edu/~asamir/papers/SPM_DNN_12.pdf)
- **用于大词汇量语音识别的上下文依赖预训练深度神经网络**（2012），G. Dahl 等人 [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.337.7548&rep=rep1&type=pdf)
- **使用深度置信网络的声学建模**（2012），A. Mohamed 等人 [[pdf]](http://www.cs.toronto.edu/~asamir/papers/speechDBN_jrnl.pdf)

### 强化学习 / 机器人技术
- **端到端深度视觉-运动策略训练**（2016），S. Levine 等人 [[pdf]](http://www.jmlr.org/papers/volume17/15-522/source/15-522.pdf)
- **使用深度学习和大规模数据收集学习机器人抓取的手眼协调**（2016），S. Levine 等人 [[pdf]](https://arxiv.org/pdf/1603.02199)
- **深度强化学习的异步方法**（2016），V. Mnih 等人 [[pdf]](http://www.jmlr.org/proceedings/papers/v48/mniha16.pdf)
- **使用双重Q学习的深度强化学习**（2016），H. Hasselt 等人 [[pdf]](https://arxiv.org/pdf/1509.06461.pdf)
- **使用深度神经网络和树搜索掌握围棋**（2016），D. Silver 等人 [[pdf]](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html)
- **使用深度强化学习的连续控制**（2015），T. Lillicrap 等人 [[pdf]](https://arxiv.org/pdf/1509.02971)
- **通过深度强化学习实现人类水平的控制**（2015），V. Mnih 等人 [[pdf]](http://www.davidqiu.com:8888/research/nature14236.pdf)
- **深度学习用于检测机器人抓取**（2015），I. Lenz 等人 [[pdf]](http://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf)
- **通过深度强化学习玩Atari游戏**（2013），V. Mnih 等人 [[pdf]](http://arxiv.org/pdf/1312.5602.pdf)

### 2016年更多论文
- **层归一化**（2016），J. Ba 等人 [[pdf]](https://arxiv.org/pdf/1607.06450v1.pdf)
- **通过梯度下降学习学习（Learning to learn）**（2016），M. Andrychowicz 等人 [[pdf]](http://arxiv.org/pdf/1606.04474v1)
- **对抗性训练的神经网络训练方法**（2016），Y. Ganin 等人 [[pdf]](http://www.jmlr.org/papers/volume17/15-239/source/15-239.pdf)
- **WaveNet：一种生成模型，用于原始音频**（2016），A. Oord 等人 [[pdf]](https://arxiv.org/pdf/1609.03499v2) [[web]](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
- **彩色图像上色**（2016），R. Zhang 等人 [[pdf]](https://arxiv.org/pdf/1603.08511)
- **自然图像流形上的生成式视觉操作**（2016），J. Zhu 等人 [[pdf]](https://arxiv.org/pdf/1609.03552)
- **纹理网络：前馈合成纹理和风格化图像**（2016），D. Ulyanov 等人 [[pdf]](http://www.jmlr.org/proceedings/papers/v48/ulyanov16.pdf)
- **SSD：单次多框检测器**（2016），W. Liu 等人 [[pdf]](https://arxiv.org/pdf/1512.02325)
- **SqueezeNet：用50倍更少的参数和小于1MB的模型大小达到AlexNet水平的准确率**（2016），F. Iandola 等人 [[pdf]](http://arxiv.org/pdf/1602.07360)
- **Eie：压缩深度神经网络的高效推理引擎**（2016），S. Han 等人 [[pdf]](http://arxiv.org/pdf/1602.01528)
- **二值神经网络：训练时权重和激活被限制为+1或-1的深度神经网络**（2016），M. Courbariaux 等人 [[pdf]](https://arxiv.org/pdf/1602.02830)
- **用于视觉和文本问答的动态记忆网络**（2016），C. Xiong 等人 [[pdf]](http://www.jmlr.org/proceedings/papers/v48/xiong16.pdf)
- **用于图像问答的堆叠注意力网络**（2016），Z. Yang 等人 [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_Stacked_Attention_Networks_CVPR_2016_paper.pdf)
- **使用动态外部记忆的神经网络混合计算**（2016），A. Graves 等人 [[pdf]](https://www.gwern.net/docs/2016-graves.pdf)
- **谷歌的神经机器翻译系统：弥合人类与机器翻译之间的差距**（2016），Y. Wu 等人 [[pdf]](https://arxiv.org/pdf/1609.08144)

以下是重新整理后的内容，更易于中国读者理解：

---

### 新论文
*近期（6个月内）值得阅读的论文*
- 《MobileNets：适用于移动视觉应用的高效卷积神经网络》 (2017), Andrew G. Howard 等人 [[pdf]](https://arxiv.org/pdf/1704.04861.pdf)
- 《卷积序列到序列学习》 (2017), Jonas Gehring 等人 [[pdf]](https://arxiv.org/pdf/1705.03122)
- 《基于知识的神经对话模型》 (2017), Marjan Ghazvininejad 等人 [[pdf]](https://arxiv.org/pdf/1702.01932)
- 《精确的大批量SGD：1小时训练ImageNet》 (2017), Priya Goyal 等人 [[pdf]](https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h3.pdf)
- 《TACOTRON：走向端到端语音合成》 (2017), Y. Wang 等人 [[pdf]](https://arxiv.org/pdf/1703.10135.pdf)
- 《深度照片风格迁移》 (2017), F. Luan 等人 [[pdf]](http://arxiv.org/pdf/1703.07511v1.pdf)
- 《进化策略：一种可扩展的强化学习替代方法》 (2017), T. Salimans 等人 [[pdf]](http://arxiv.org/pdf/1703.03864v1.pdf)
- 《可变形卷积网络》 (2017), J. Dai 等人 [[pdf]](http://arxiv.org/pdf/1703.06211v2.pdf)
- 《Mask R-CNN》 (2017), K. He 等人 [[pdf]](https://128.84.21.199/pdf/1703.06870)
- 《利用生成对抗网络发现跨领域关系》 (2017), T. Kim 等人 [[pdf]](http://arxiv.org/pdf/1703.05192v1.pdf)
- 《Deep Voice：实时神经文本到语音》 (2017), S. Arik 等人 [[pdf]](http://arxiv.org/pdf/1702.07825v2.pdf)
- 《PixelNet：像素的表示，按像素表示，为像素表示》 (2017), A. Bansal 等人 [[pdf]](http://arxiv.org/pdf/1702.06506v1.pdf)
- 《批归一化：减少批量归一化模型中的批量依赖》 (2017), S. Ioffe [[pdf]](https://arxiv.org/abs/1702.03275)
- 《Wasserstein GAN》 (2017), M. Arjovsky 等人 [[pdf]](https://arxiv.org/pdf/1701.07875v1)
- 《理解深度学习需要重新思考泛化问题》 (2017), C. Zhang 等人 [[pdf]](https://arxiv.org/pdf/1611.03530)
- 《最小二乘生成对抗网络》 (2016), X. Mao 等人 [[pdf]](https://arxiv.org/abs/1611.04076v2)

### 经典论文
*2012年前发布的经典论文*
- 《无监督特征学习中的单层网络分析》 (2011), A. Coates 等人 [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_CoatesNL11.pdf)
- 《深度稀疏修正神经网络》 (2011), X. Glorot 等人 [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_GlorotBB11.pdf)
- 《几乎从零开始的自然语言处理》 (2011), R. Collobert 等人 [[pdf]](http://arxiv.org/pdf/1103.0398)
- 《基于循环神经网络的语言模型》 (2010), T. Mikolov 等人 [[pdf]](http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf)
- 《堆叠去噪自编码器：在深度网络中使用局部去噪准则学习有用的表示》 (2010), P. Vincent 等人 [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.3484&rep=rep1&type=pdf)
- 《识别的中级特征学习》 (2010), Y. Boureau [[pdf]](http://ece.duke.edu/~lcarin/boureau-cvpr-10.pdf)
- 《训练限制玻尔兹曼机的实用指南》 (2010), G. Hinton [[pdf]](http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf)
- 《理解深度前馈神经网络训练难度》 (2010), X. Glorot 和 Y. Bengio [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf)
- 《为什么无监督预训练有助于深度学习》 (2010), D. Erhan 等人 [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_ErhanCBV10.pdf)
- 《为AI学习深度架构》 (2009), Y. Bengio [[pdf]](http://sanghv.com/download/soft/machine%20learning,%20artificial%20intelligence,%20mathematics%20ebooks/ML/learning%20deep%20architectures%20for%20AI%20(2009).pdf)
- 《卷积深度置信网络：可扩展的无监督学习层级表示》 (2009), H. Lee 等人 [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.802&rep=rep1&type=pdf)
- 《深度网络的贪婪分层训练》 (2007), Y. Bengio 等人 [[pdf]](http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2006_739.pdf)
- 《使用神经网络减少数据的维度》 (2006), G. Hinton 和 R. Salakhutdinov [[pdf]](http://homes.mpimf-heidelberg.mpg.de/~mhelmsta/pdf/2006%20Hinton%20Salakhudtkinov%20Science.pdf)
- 《深度置信网络的快速学习算法》 (2006), G. Hinton 等人 [[pdf]](http://nuyoo.utm.mx/~jjf/rna/A8%20A%20fast%20learning%20algorithm%20for%20deep%20belief%20nets.pdf)
- 《基于梯度的学习应用于文档识别》 (1998), Y. LeCun 等人 [[pdf]](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- 《长短期记忆》 (1997), S. Hochreiter 和 J. Schmidhuber [[pdf]](http://www.mitpressjournals.org/doi/pdfplus/10.1162/neco.1997.9.8.1735)

### 计算机硬件/软件/数据集
- 《SQuAD：100,000+个机器理解文本的问题集》 (2016), Rajpurkar 等人 [[pdf]](https://arxiv.org/pdf/1606.05250.pdf)
- 《OpenAI Gym》 (2016), G. Brockman 等人 [[pdf]](https://arxiv.org/pdf/1606.01540)
- 《TensorFlow：异构分布式系统上的大规模机器学习》 (2016), M. Abadi 等人 [[pdf]](http://arxiv.org/pdf/1603.04467)
- 《Theano：用于快速计算数学表达式的Python框架》 R. Al-Rfou 等人
- 《Torch7：一个类似Matlab的机器学习环境》 (2011), R. Collobert 等人 [[pdf]](https://ronan.collobert.com/pub/matos/2011_torch7_nipsw.pdf)
- 《MatConvNet：Matlab中的卷积神经网络》 (2015), A. Vedaldi 和 K. Lenc [[pdf]](http://arxiv.org/pdf/1412.4564)
- 《ImageNet大规模视觉识别挑战赛》 (2015), O. Russakovsky 等人 [[pdf]](http://arxiv.org/pdf/1409.0575)
- 《Caffe：用于快速特征嵌入的卷积架构》 (2014), Y. Jia 等人 [[pdf]](http://arxiv.org/pdf/1408.5093)

### 书籍/综述/评论
- 《深度学习的起源》 (2017), H. Wang 和 Bhiksha Raj [[pdf]](https://arxiv.org/pdf/1702.07800)

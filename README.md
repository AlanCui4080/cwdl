# CWDL

用于识别CW Morse通讯的深度学习系统

## 数据集和预处理

oxford5000：牛津常见5000词，剔除了长度小于3个字母的词汇
radioabbr：常见CW通讯缩写
random6：10000个3-6位随机长度的随机字母数字

随后这些数据被合成为30WPM到60WPM的CW音频，其中：
- 加入随机最大值在10Hz-100Hz的频率偏倚，FM调制函数是在0.2-1Hz的三角波
- 加入WPM偏倚，范围为正负20%
- 叠加指定的附加噪声功率后归一化
- 经过200Hz带宽60dB/dec的成型滤波器
- 采样率统一为48000Hz，前20ms和后20ms是保护间隔。

随后进行STFT,抽取为375Hz高度的灰度频谱图，其中每23.4375Hz定义为一个像素高（这是为了fft长度为2048），即高度为16px，每10ms定义为一个像素长。这些带保护边缘的wav会生成以128px为单位的，带保护间隔的音频和频谱图，随后按50%重叠率分块，其中被切断的电码会被替换为-以便于CTC训练。

训练集是以附加噪声功率和WPM分类的，附加噪声功率的范围是+18dB到-9dB，3dB每步。WPM有30,40,50,60四种，以此组合，每组合随机抽取3000个序列，共计120000条。每3000条中500条是在oxford5000和radioabbr中随机抽取的，剩下2500条是在random6中抽取的（这是为了避免在学习过程中学习到自然语言结构），SWM*WPM组合内抽取结果不重复。

验证集和测试集也是按上方标准抽取的，附加噪声功率的规律不变，WPM为25 30 35 40 45 50 55 60 65，以此组合，每组合随机抽取300个序列，三个集合词汇间互斥，词汇池比例为6/2/2。

每个词组具有不定长度的后黑，这是刻意保留而未填充为白噪声的，意图使得模型学会处理实时输入和可能的后黑。

## 架构和训练
分为两个神经网络，第一个CNN通过提取音频的频谱图得到特征向量，第二个BiGRU通过提取前后特征最终给出解码结论，两个网络是协同训练的，在训练中，CNN接受预切割好的窗口，每个上下文训练滑动整个词汇的一个区块，整个序列不去重的送入BiGRU，去重由CTC实现。

CNN网络首先通过卷积层快速将高度压缩到1px，意图使网络快速学会忽略和压缩频偏，然后做1D CNN提取点划空特征，随后提取到的这些特征送入BiGRU，接入CTC后进行学习。

v1 结构 (0.17M)：
- 3x3 conv2d stride4x1 padding0x1 Norm ReLU 16ch
- 3x3 conv2d stride4x1 padding0x1 Norm ReLU 32ch
至此，模型已经完全和压缩频偏（高度为1），随后：
- 3x1 conv1d dilation1 padding1 Norm ReLU 32->64ch (感受野3步)
- 3x1 conv1d dilation2 padding2 Norm ReLU 64ch (总感受野7步)
- 3x1 conv1d dilation4 padding4 Norm ReLU 64ch (总感受野15步)
- BiGRU layer2 input64 hidden64 drop0.3
最终贪心以后得到结果，损失直接使用标准CTCLoss。
第一次最佳 epoch=27, cer=0.1137 手动降低学习率，随后
第二次最佳 epoch=30, cer=0.1110 改进学习率下降函数 降低学习率
第二次最佳 epoch=43, cer=0.1097 
![](cer_heatmap.png)

v1.1(参数加大加宽版 0.7M) 结构：
- 3x3 conv2d stride4x1 padding0x1 Norm ReLU 32ch
- 3x3 conv2d stride4x1 padding0x1 Norm ReLU 64ch
至此，模型已经完全和压缩频偏（高度为1），随后：
- 3x1 conv1d dilation1 padding1 Norm ReLU 64->128ch (感受野3步)
- 3x1 conv1d dilation2 padding2 Norm ReLU 128ch (总感受野7步)
- 3x1 conv1d dilation4 padding4 Norm ReLU 128ch (总感受野15步)
- 3x1 conv1d dilation8 padding8 Norm ReLU 128ch (总感受野30步)
- BiGRU layer2 input128 hidden128 drop0.3
最终贪心以后得到结果，损失直接使用标准CTCLoss。
第一次最佳 epoch=7, cer=0.1172 手动降低学习率，随后
第二次最佳 epoch=33, cer=0.1092
![](cer_heatmapv11.png)

纵然CER仍相对较高，但是我认为这是测试集中存在较低SNR和较极端WPM的样本，
尤其的，有较低WPM的样本，模型感受野可能不足以识别整个点划。

v2 (~~变形金刚~~版 1.92M) 结构：
- 3x3 conv2d stride4x1 padding0x1 Norm ReLU 64ch
- 3x3 conv2d stride4x1 padding0x1 Norm ReLU 128ch
此处完全移除了1D卷积层，意图使Transfomer模型直接理解1D莫尔斯序列
- EncoderOnlyTransfomer dmodel128 dffn512 nhead4 layer4 drop0.3
最终贪心以后得到结果，损失直接使用标准CTCLoss。效果十分不好（）

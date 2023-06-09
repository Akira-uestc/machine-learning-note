K-近邻算法（KNN）全称为K Nearest Neighbor，是指如果⼀个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的⼤多数属于某⼀个类别，则该样本也属于这个类别。

简单来说，就是看你的邻居是哪个类别，你就是哪个类别

由于KNN最邻近分类算法在分类决策时只依据最邻近的一个或者几个样本的类别来决定待分类样本所属的类别，而不是靠判别类域的方法来确定所属类别的，因此对于类域的交叉或重叠较多的待分样本集来说，KNN方法较其他方法更为适合。

我们使用欧氏距离（两个点之间的距离）来进行判断两个样本之间的距离，进而判断最相似。

训练样本是多维特征空间向量，其中每个训练样本带有一个类别标签。算法的训练阶段只包含存储的特征向量和训练样本的标签。

在分类阶段，k是一个用户定义的常数。一个没有类别标签的向量（查询或测试点）将被归类为最接近该点的k个样本点中最频繁使用的一类。

一般情况下，将欧氏距离作为距离度量，但是这是只适用于连续变量。

在文本分类这种离散变量情况下，另一个度量——重叠度量（或海明距离）可以用来作为度量。

通常情况下，如果运用一些特殊的算法来计算度量的话，k近邻分类精度可显著提高，如运用大间隔最近邻居或者邻里成分分析法。

## KNN算法的步骤
(1) 样本的所有特征都要做可比较的量化

    若是样本特征中存在非数值的类型，必须采取手段将其量化为数值。例如样本特征中包含颜色，可通过将颜色转换为灰度值来实现距离计算。

(2) 样本特征要做归一化处理

    样本有多个参数，每一个参数都有自己的定义域和取值范围，他们对距离计算的影响不一样，如取值较大的影响力会盖过取值较小的参数。
    
    所以样本参数必须做一些 scale 处理，最简单的方式就是所有特征的数值都采取归一化处置。
    
(3) 需要一个距离函数以计算两个样本之间的距离

    通常使用的距离函数有：欧氏距离、余弦距离、汉明距离、曼哈顿距离等，一般选欧氏距离作为距离度量，但是这是只适用于连续变量。
    
    在文本分类这种非连续变量情况下，汉明距离可以用来作为度量。
    
    通常情况下，如果运用一些特殊的算法来计算度量的话，K近邻分类精度可显著提高，如运用大边缘最近邻法或者近邻成分分析法。

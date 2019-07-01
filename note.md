# CTC 源码分析
## 源码位置
git clone --recursive https://github.com/parlance/ctcdecode.git
## 安装
1. 新建虚拟环境 python=2.7
2. 安装pytorch1.0 和wget
3. 提前下载好openfst-1.6.7.tar.gz 和boost_1_67_0.tar.gz 放在third_party 目录下。可以提前解压
3. pip install .
4. 在虚拟环境的site-packdge目录下的ctcdecode/_ext/ 目录下添加空白的__init__.py 文件。并且不要在ctcdecode 的源代码目录下运行。
5. 需要先import torch ，再import ctcdecode, 否则会找不到符号
## ctc_beam_search_decoder函数（ctc_beam_search_decoder.cpp）
### 参数
* probs_seq: softmax, 二维list，（time_step * num_class + 1）
* vocabulary： 所有的字符列表, size：num_class
* beam_size: 指定的最大beam数目
* cutoff_prob：每个step时，选取概率的前百分比对应的class
* cutoff_top_n: 每个step时，选取概率的top n的class
* blank_id: 分割blank的索引值
* ext_scorer：使用LM时计算scorer的工具类

### 主要步骤
1. 只有当使用LM且不是基于char的时候使用字典
2. 按time_step循环
    1. 取第N步的概率 prob
    2. 如果指定了ext_scorer，则计算min_cutoff 和判断是否分值数目已经达到beam size。*不懂为什么要减去beta*
    3. 对所有可能剪枝，只取出当前step的前cutoff_top_n或者指定比例cutoff_prob的可能字符及其对应的概率。*源代码中这部分有错误*
    4. 按照上一步中的每一个可能的字符c循环
        1. 取字符c及其对应的概率log_prob_c
        2. 按照prefixes 循环
            1. 判断当前beam 是否已经满，同时叠加当前字符的概率 小于 叠加 blank的概率， 则中断当前step后续的字符可能
            2. 如果当前字符c是blank，则只更新$prob_blank_cur$. 因为 blank + blank = blank， no blank + blank = blank 。所以p_b = p_c * (p_b + p_nb). 当前分支不需要后续操作
            3. 如果当前字符c等于prefix的最后一个字符时，只更新no blank的概率。 p_nb = p_c * p_nb
            4. **获取新的分支get_path_trie**。新的分支是当前分支的孩子节点
            5. 计算新增节点的p_nb_cur
                + 如果c等于prefix的最后一个字符， 且  p_nb_pre 不等于0，则新的p_c = p_c * p_nb_pre
                + 如果c 不等于上一个字符，则新的p_c = p_c * (p_nb_pre + p_b_pre)
                + 当c是空格(基于word LM)或者是基于char的LM时，**叠加LM的影响**。 现根据当前节点，往父节点找到ngram对应的单词或者字母，在查LM得到概率。更新后的概率p_nb = p_c * p_lm^ alpha * beta
    5. 更新prefixes，且只保留前beam_size 个数的分支
3. 对于使用word lM的情况，处理不以space结尾的prefix
4. 计算大概的ctc分数，移除了LM的影响之后的分数

### 字符串组合情况分析
假设任意字符串 *（最后一个字符为A）与一个新字符的组合有以下几种情况：
- *blank + blank = *blank
- *blank + char = （*char）no_blank
- *no_blank + blank = *blank
- *no_blank + char_(no_A) = (* + char_(no_A)) no_blank
- *no_blank + A = *no_blank

### 几个实验
1. trie树转为prefix list时，如果只考虑叶子节点，效果不好，噪声很大。但是按照目前算法，有可能会丢掉若干字符，alpha越大，丢的概率越大。

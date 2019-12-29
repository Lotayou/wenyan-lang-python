# wenyan-lang-python
A python compiler and parser for converting wenyan into python scripts. The basic syntax will be compatible with the original [wenyan-lang](https://github.com/LingDong-/wenyan-lang) repo, but some new pythonic features will be explored and supported if possible. In essence, the purpose of this project is to create a "dialect" of wenyan-lang that is more flexible and effective.

### Relationships with [wenyan-lang](https://github.com/LingDong-/wenyan-lang)
The purpose of wenyan-lang-python is to create an effecient and concise wrapper for python language along with an intepreter to translate existing python codes back to wenyan. Unlike wenyan-lang that focus on designing a context-free syntax, this project is exclusively targeted for python for better cross translation and mix language development. Although the basic syntax would be made compatible to the original wenyan-lang, some new features will be introduced, and the syntax and keywords will be slightly different than LingDong's version. 

For the record, I will use simplified Chinese characters 简体字 instead of traditional Chinese 繁體字。

### TODO LIST
- [ ] Basic syntax support
- [ ] Flexible Function parameters: *args, \**kwargs
- [ ] list, tuple, dict, set, and list comprehension
- [ ] [Nested functions](https://github.com/LingDong-/wenyan-lang/issues/322), partial functions, lambdas
- [ ] class and methods
- [ ] File I/O
- [ ] Advanced Import Mechanism
- [ ] pytorch wrapper for basic deep learning

### Building Deep Learning Networks in Wenyan
To demonstrate the capability of the perceived wenyan-lang-python, here is an example of a simple convolutional neural network and the corresponding pytorch version. The original post can be found [here](https://github.com/LingDong-/wenyan-lang/issues/281)

```
吾观一书，名曰「火炬心法」  # torch
自「火炬心法」之书 引「炼法」之篇，「备料」之篇，「丹器」之篇
自「火炬心法」之书 引「檀瑟」之器  # tensor
自「火炬心法」之书「备料」之篇 引「料堆」，「料铲」
自「火炬心法」之书「丹器」之篇引「丹炉」之器，「高炉」之器

吾观一书，名曰「火眼金睛」  # torchvision
自「火眼金睛」之书「备料」之篇引「缩放」之术，「中和」之术，「翻转」之术

吾有一术。名曰「川流」。欲行是术。
    必先得一列。曰「诸炉」。列中诸元。皆为「丹炉」。
    
    吾有一术。名曰「高炉」。欲行是术。
        必先得一「檀瑟」之器。名曰「料」。
            凡「诸炉」中之各「层」。
                施「层」之术于「料」。赋还其身  # x = t(x)
            乃得「料」也。
        
    乃得「高炉」之术。
是谓「川流」之术也。

批曰。吾人欲炼金丹，需先造丹炉
吾有一丹炉。名曰「八卦炉」。欲造此炉。
    必先得四数。
        曰「入」。其值原应为三。
        曰「类」。其值原应为十。
        曰「料尺」。其值原应为廿八。
        曰「通数」。其值原应为六十有四。
        
    必先得两爻。
        曰「弃乎」。其值原应为阳。
        曰「归一乎」。其值原应为阳。
       
    乃造此炉如下。
        造「八卦炉」之「基座」  #super(...,self).__init__()

        吾有两数。曰「前通」。曰「后通」。
        昔之「前通」者。今「通数」是矣。
        昔之「后通」者。今「通数」是矣。
        吾有一列。曰「方炉」。
            充「方炉」以「卷积」之层。其形制如下。
                进口「入」个，出口「后通」个。「核」长宽各七。入料时「镶边」各三。每隔一「步」炼之
            充「方炉」以「池化」之层。其形制如下。
                凡每一进口。取邻域长宽各「二」。采其「均值」。
            充「方炉」以「激活」之层。其形制如下。
                凡入之诸元，取其值与零之大者赋之
            
            昔之「前通」者，今「后通」是矣。
            乘「后通」以二。
            除「料尺」以二。
            
            充「方炉」以「卷积」之层。其形制如下。
                进口「前通」个，出口「后通」个。「核」长宽各三。入料时「镶边」各一。每隔一「步」炼之
            充「方炉」以「池化」之层。其形制如下。
                凡每一进口。取邻域长宽各「二」。采其「均值」。
            充「方炉」以「激活」之层。其形制如下。
                凡入之诸元，取其值与零之大者赋之
            
            除「料尺」以二。
                
        施「川流」之术于「方炉」。得一「高炉」。名之曰「特征」
        
        乘「后通」以「料尺」以「料尺」。记之曰「入维」
        吾有一列。曰「线炉」。
            充「线炉」以「线性」之层。其形制如下。
                进口长曰「入维」，出口长曰「类」。 批曰。如何添加bias
            若「弃乎」为阳。
                充「线炉」以「阻滞」之层。其功用如下。
                    随缘关闭炉内通道。只留其「半数」。
            若「归一乎」为阳。
                充「线炉」以「归一」之层。其实现如下。
                    凡「入料」中之「物」。皆取幂。得一列。记之曰「概率」
                    施「列和」之数于「概率」之列。得一数。记之曰「幂和」
                    凡「概率」中之「数」。除「数」以「幂和」。  批曰。易证「概率」之「列和」为一也
        
        施「川流」之术于「线炉」。得一「高炉」。名之曰「预测」
    至此。炉乃成。
        
    此炉有「炼丹术」。欲行是术。必先得一「檀瑟」之器。名曰「入料」。
        乃行「炼丹术」如下。
        
        观「入料」之形，得一列。名之曰「尺寸」
        若夫「尺寸」之长 不为「四」或 「尺寸」之三 其值不为 廿八：
            警云「「入料与丹炉方圆不合，慎之慎之！」」
            
        「入料」进「特征」之炉炼之。产物记之曰「中料」
        施「整形」之术于「中料」。
        「中料」进「预测」之炉炼之。产物记之曰「出品」
        乃得「出品」。
    是谓「炼丹术」也。
        
如此「八卦炉」乃成。
```

```
import torch
from torch import nn, optim, data
from torch.data.utils import Dataset, DataLoader
from torch.nn import Module, Sequential

def sequential(*layers):
    def _chain_process(x -> torch.Tensor):
        for l in layers:
            x = l(x)
        return x
    
    return _chain_process

# We're gonna build a large furnace for alchemic experiments
class BaGuaFurnace(nn.Module):
    def __init__(self, 
        dim=3, class_num=10, im_size=28, nf=64,
        use_dropout=True, use_sigmoid=False):
        super(BaGuaFurnace, self).__init__()
        
        indim, outdim = dim
        conv = [
            nn.Conv2d(dim, outdim, kernel_size=7, padding=3, stride=1),
            nn.AvgPool2d(stride=2),
            nn.ReLU(),
        ]
        indim, outdim = outdim, outdim * 2
        im_size = im_size // 2
        conv += [
            nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=1),
            nn.AvgPool2d(stride=2),
            nn.ReLU(),
        ]
        im_size = im_size // 2
        self.feature = sequential(*conv)
        
        fc_indim = im_size * im_size * outdim
        fc = [nn.Linear(fc_indim, class_num, use_bias=True)]
        if use_dropout:
            fc += [nn.Dropout(0.5)]
        if use_sigmoid:
            fc += [nn.Sigmoid()]
        self.predict = sequential(*fc)
        
    def forward(self, in):
        shape = mid.size()
        if len(shape) != 4 or shape[3] != 28:
            raise(Warning('Oi, wrong size!'))
        mid = self.feature(in)
        mid = mid.view(shape[0], -1)
        out = self.predict(mid)
        return out
```

### Contributing
Feel free to open an issue or submit a PR if you have new ideas.



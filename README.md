# ViT net for OCTA saturation  
The net is based on project []()  
## Environment 
```

```
## Usage of mixloss
mixloss() take inputs of class logits +  regression value as input, and implement $BCEwithlogits() + MSELoss()$ as criterion for backpropagation 
$$mixloss = \alpha * bce + (1-\alpha) * mse $$
Tips: it supports bal-bce
usage: 
```
import oct_vit/util/utilize as util

criterion = mixloss(a, pos_weight)     # a is the weight of bce
# target = [0.9, 0.2, 0.1, 0.2, 98]   4class logits + regression value
# label = [1, 0, 0, 0, 97]
loss = mixloss(target, loss)
```
## Usage of 

## Usage of implementing train 

## Usage of 2dtrain-3dvote-inference


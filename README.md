# ViT net for OCTA saturation  
The net is based on project []()  
## Environment 
```

```
## Usage of mixloss
mixloss() take inputs of class logits +  regression value as input, and implement $BCEwithlogits() + MSELoss()$ as criterion for backpropagation 
$$mix = \alpha * bce + (1-\alpha) * mse $$
usage: 
```
import oct_vit/util/utilize as util
criterion = mixloss(a) # a is the weight of bce
# target = [0.9, 0.2, 0.1, 0.2, 98]
# label = [1, 0, 0, 0, 97]
loss = mixloss(target, loss)


```
## Usage of 

## Usage of implementing train 

## Usage of 2dtrain-3dvote-inference


### how to use [BaseLoss](loss.py)
```
BaseLoss(torch.nn.NLLLoss(), device=True/False, )
```
includes:
* [BaseLoss](loss.py)



### how to use [general loss](base)
```
# succeed from 1) BaseLoss or 2) Module and then pack it with BaseLoss
TripletLoss(device=True) # eg. TripletLoss 
```
includes:
* [TripleLoss](base/triple_loss.py)


### how to use [package loss](base)
```
# you CANNOT build a loss directly, but have to assing an exiting loss class
OHEMLoss(nn.NLLLoss, device=True, ) 
```
includes:
* [OHEMLoss](base/ohem_loss.py)

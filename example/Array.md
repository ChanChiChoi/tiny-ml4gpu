you can use ipython to write this code
```
from Array import Array  
import numpy as np  
npdata = np.random.randn(100,60).astype(np.float32)  
data = Array(npdata)  

# transfer numpy data from host to gpu device  
data.cuda()  

# display current data info  
data.display()  

# transfer data on gpu deivce to cpu host, the result is numpy data  
res = data.cpu()  
```

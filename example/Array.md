you can use ipython to write this code
```
from Array import Array  
import numpy as np  
npdata = np.random.randn(100,60).astype(np.float32)  
data = Array(npdata)  

# transfer numpy data from host to gpu device  
data.cuda()  

# display current meta info of data  
data.display_meta()  

# display current data, current only support ndim == 1or2
data.display_data()

# transfer data on gpu deivce to cpu host, the result is numpy data  
res = data.cpu()  
```

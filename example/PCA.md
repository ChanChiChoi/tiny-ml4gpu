
you can use ipython to input this code

```
from Array import Array
from PCA import PCA
import numpy as np
npdata = np.random.randn(100,6).astype(np.float32)

data = Array(npdata)

data.cuda()

# init PCA, you need send 
pca = PCA(3)
pca.fit(data)
res_data = pca.transform(data)

# there is a bug on cpu() function, the right values can be displayed by display_cpu() or display_cuda()
npres = res_data.cpu()

# display value on host side, the value then transfer onto numpy side, but there is a problem that after transfering,
# the value will be changed.

res_data.display_cpu()
```

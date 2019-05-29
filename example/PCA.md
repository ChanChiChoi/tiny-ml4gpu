
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

# the right values can be displayed by display_cpu() or display_cuda()
npres = res_data.cpu()

# display value on host side, the value then transfer onto numpy side,
# the value of res_data.display_cpu() should be equal to res_data.dpsplay_cuda()
# and res_data.cpu()

res_data.display_cpu()
```

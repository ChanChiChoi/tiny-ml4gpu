
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

npres = res.data.cpu()
```

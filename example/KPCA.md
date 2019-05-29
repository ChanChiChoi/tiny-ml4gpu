
you can use ipython to input this code

```
from KPCA import KPCA
from Array import Array
import numpy as np
nptraindata = np.random.randn(100,10).astype(np.float32)

traindata = Array(nptraindata)

traindata.cuda()

# init KPCA, you need send 
kpca = KPCA(3)
kpca.fit(traindata)

nptestdata = np.random.randn(80,10).astype(np.float32)
testdata = Array(nptestdata)
testdata.cuda()
res_data = kpca.transform(traindata, testdata)

npres = res_data.cpu()
```

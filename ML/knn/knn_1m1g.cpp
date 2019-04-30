#include<assert.h>
#include "common/buffer_info_ex.h"

class KNN{


    Buf buf;

public:

    KNN() {}

    KNN(Buf &buf){

    }

    // need overload 2 case: i)numpy -> fit; ii)numpy -> pre -> fit
    KNN & fit(py::array_t<float> &array);

    KNN & fit(Buf &buf);

    predict(){


    }
}


KNN & KNN::fit(py::array_t<float> &mat){

    auto mat_info = mat.requests()

    if (mat_info.format != py::format_descriptor<float>::format())
        throw std::runtime_error("Incompatible format: excepted a float32 array!");
    if (mat_info.ndim != 2)
        throw std::runtime_error("Incompatible buffer dimension!");

    //copy to gpu


    return *this;
};



/*
 * from ml4gpu import KNN
 * knn = KNN()
 * train = buf(np.train)
 * train.cuda()
 * knn.fit(train)
 * test = buf(np.test)
 * test.cuda()
 * ans = knn.pred(test)
 * train.cpu() // release the memory
 * test.cpu()
 * -------------------------------
 * from ml4gpu import KNN,dataPre
 * norData = dataPre()
 * gpu.train = norData.normal(np.train)
 *
 * knn = KNN()
 * knn.fit(gpu.train)
 * gpu.test = norData.transform(np.test)
 * ans = knn.pred(gpu.test)
 *
 *
 *
 * */

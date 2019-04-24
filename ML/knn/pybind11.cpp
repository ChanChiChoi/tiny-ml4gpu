#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include "knn.h"

namespace py = pybind11;


//float *knn_one(float *x, float *dataset, float *ans,
//             const unsigned int col, const unsigned int num_samples)
void 
knn_vec_mat(py::array_t<float> &x, py::array_t<float> &dataset, 
           py::array_t<float> &ans, unsigned int col, unsigned int num_samples){
    py::buffer_info buf_x = x.request();
    py::buffer_info buf_dataset = dataset.request();
    py::buffer_info buf_ans = ans.request();
    
    float *p_x = (float *)buf_x.ptr;
    float *p_dataset = (float *)buf_dataset.ptr;
    float *p_ans = (float *)buf_ans.ptr;
    knn_one(p_x, p_dataset, p_ans, col, num_samples);

}


PYBIND11_MODULE(example, m){
    m.doc() = "just for knn";
    m.def("knn_vec", &knn_vec_mat, "a function for knn classify"
          );

}

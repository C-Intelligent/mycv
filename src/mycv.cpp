#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <queue>
#include <vector>
#include <algorithm>

namespace py = pybind11;

//交互 numpy
template<typename T>
py::array_t<T> add_arrays(py::array_t<T> input1, py::array_t<T> input2) {
    py::buffer_info buf1 = input1.request(), buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shapes must match");

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<T>(buf1.size);

    py::buffer_info buf3 = result.request();

    T *ptr1 = (T *) buf1.ptr,
           *ptr2 = (T *) buf2.ptr,
           *ptr3 = (T *) buf3.ptr;

    for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        ptr3[idx] = ptr1[idx] + ptr2[idx];

    return result;
}

py::array_t<double> add_arrays_2d(py::array_t<double>& input1, py::array_t<double>& input2) {

    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();

    if (buf1.ndim != 2 || buf2.ndim != 2)
    {
        throw std::runtime_error("numpy.ndarray dims must be 2!");
    }
    if ((buf1.shape[0] != buf2.shape[0])|| (buf1.shape[1] != buf2.shape[1]))
    {
        throw std::runtime_error("two array shape must be match!");
    }

    //申请内存
    auto result = py::array_t<double>(buf1.size);
    //转换为2d矩阵
    result.resize({buf1.shape[0],buf1.shape[1]});


    py::buffer_info buf_result = result.request();

    //指针访问读写 numpy.ndarray
    double* ptr1 = (double*)buf1.ptr;
    double* ptr2 = (double*)buf2.ptr;
    double* ptr_result = (double*)buf_result.ptr;

    for (int i = 0; i < buf1.shape[0]; i++)
    {
        for (int j = 0; j < buf1.shape[1]; j++)
        {
            auto value1 = ptr1[i*buf1.shape[1] + j];
            auto value2 = ptr2[i*buf2.shape[1] + j];

            ptr_result[i*buf_result.shape[1] + j] = value1 + value2;
        }
    }

    return result;

}

py::array_t<double> add_arrays_3d(py::array_t<double>& input1, py::array_t<double>& input2) {

    //unchecked<N> --------------can be non-writeable
    //mutable_unchecked<N>-------can be writeable
    auto r1 = input1.unchecked<3>();
    auto r2 = input2.unchecked<3>();

    py::array_t<double> out = py::array_t<double>(input1.size());
    out.resize({ input1.shape()[0], input1.shape()[1], input1.shape()[2] });
    auto r3 = out.mutable_unchecked<3>();

    for (int i = 0; i < input1.shape()[0]; i++)
    {
        for (int j = 0; j < input1.shape()[1]; j++)
        {
            for (int k = 0; k < input1.shape()[2]; k++)
            {
                double value1 = r1(i, j, k);
                double value2 = r2(i, j, k);

                //下标索引访问 numpy.ndarray
                r3(i, j, k) = value1 + value2;
            
            }
        }
    }

    return out;

}

template<typename T>
py::array_t<T> mediablur(py::array_t<T>& img_rgb, int height, int width) {
    py::buffer_info img_info = img_rgb.request();
    if (img_rgb.ndim()!=3) throw std::runtime_error("Image  must be 3 dim!!");
    int chanels = img_info.shape[2];
    
    int filter_c_h = height >> 1, filter_c_w = width >> 1;
    int border_t = filter_c_h, border_b = height - filter_c_h, 
        border_l = filter_c_w, border_r = width - filter_c_w;
    
    auto res = py::array_t<T>(img_info.size);
    //transfer
    res.resize({img_info.shape[0], img_info.shape[1], img_info.shape[2]});
    auto res_info = res.request();
    auto ptr = (T*)res_info.ptr, img = (T*)img_info.ptr;
    memcpy(ptr, img, img_info.size * sizeof(T));
    
    //media
    auto med = [&](int _i, int _j, int off) -> T {
        std::vector<T> arr;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = ((_i + i) * img_info.shape[1] + j + _j) * chanels;
                arr.push_back(img[idx + off]);
            }
        }
        std::sort(arr.begin(), arr.end());
        return arr[arr.size() / 2];
    };
    //main part
    for (int i = border_t; i < img_info.shape[0] - border_b; i++) {
        for (int j = border_l; j < img_info.shape[1] - border_r; j++) {
            for (int k = 0; k < chanels; k++) {
                int idx = (i * img_info.shape[1] + j) * chanels;
                ptr[idx + k] = med(i - border_t, j - border_l, k);
            }
        }
    }

    return res;
}

template<typename T, typename N>
py::array_t<T> rgb_filter(py::array_t<T>& img_rgb, py::array_t<N>& filter) {
    py::buffer_info img_info = img_rgb.request(), fil_info = filter.request();
    if (img_rgb.ndim()!=3 && img_info.shape[2] != 3) throw std::runtime_error("RGB image must has 3 channels!");
    if (filter.ndim()!=2) throw std::runtime_error("filter must be 2d!");
    
    int filter_c_h = fil_info.shape[0] >> 1, filter_c_w = fil_info.shape[1] >> 1;
    int border_t = filter_c_h, border_b = fil_info.shape[0] - filter_c_h, 
        border_l = filter_c_w, border_r = fil_info.shape[1] - filter_c_w;
    
    auto res = py::array_t<T>(img_info.size);
    //transfer
    res.resize({img_info.shape[0], img_info.shape[1], img_info.shape[2]});
    auto res_info = res.request();
    auto ptr = (T*)res_info.ptr, img = (T*)img_info.ptr;
    auto fil = (N*)fil_info.ptr;
    memcpy(ptr, img, img_info.size * sizeof(T));
    
    //normal
    N sum = 0;
    for (int i = 0; i < fil_info.shape[0] * fil_info.shape[1]; i++) sum += fil[i];
    if (sum > 0.001) for (int i = 0; i < fil_info.shape[0] * fil_info.shape[1]; i++) fil[i] = fil[i]/sum;
    auto mul = [&](int _i, int _j, int off) -> N {
        N ret = 0;
        for (int i = 0; i < fil_info.shape[0]; i++) {
            for (int j = 0; j < fil_info.shape[1]; j++) {
                int idx = ((_i + i) * img_info.shape[1] + j + _j) * 3;
                ret += fil[i * fil_info.shape[1] + j] * (int32_t)img[idx + off];
            }
        }
        return ret;
    };
    //main part
    for (int i = border_t; i < img_info.shape[0] - border_b; i++) {
        for (int j = border_l; j < img_info.shape[1] - border_r; j++) {
            int idx = (i * img_info.shape[1] + j) * 3;
            ptr[idx + 0] = mul(i - border_t, j - border_l, 0);
            ptr[idx + 1] = mul(i - border_t, j - border_l, 1);
            ptr[idx + 2] = mul(i - border_t, j - border_l, 2);
        }
    }

    return res;
}

#define _NOR_LINEAR      0
#define _NOR_TRUMC       1
#define _NOR_FACTOR      2
template<typename T, typename N>
py::array_t<T> rgb_nor_filter(py::array_t<T>& img_rgb, py::array_t<N>& filter, uint8_t nor_method) {
    py::buffer_info img_info = img_rgb.request(), fil_info = filter.request();
    if (img_rgb.ndim()!=3 && img_info.shape[2] != 3) throw std::runtime_error("RGB image must has 3 channels!");
    if (filter.ndim()!=2) throw std::runtime_error("filter must be 2d!");
    
    int filter_c_h = fil_info.shape[0] >> 1, filter_c_w = fil_info.shape[1] >> 1;
    int border_t = filter_c_h, border_b = fil_info.shape[0] - filter_c_h, 
        border_l = filter_c_w, border_r = fil_info.shape[1] - filter_c_w;
    
    auto res = py::array_t<T>(img_info.size);
    //transfer
    res.resize({img_info.shape[0], img_info.shape[1], img_info.shape[2]});
    auto res_info = res.request();
    auto ptr = (T*)res_info.ptr, img = (T*)img_info.ptr;
    auto fil = (N*)fil_info.ptr;
    memcpy(ptr, img, img_info.size * sizeof(T));
    
    //normal
    double max = 0, min = 0;
    double *buf = new double[img_info.size];
    
    auto mul = [&](int _i, int _j, int off) -> double {
        double ret = 0;
        for (int i = 0; i < fil_info.shape[0]; i++) {
            for (int j = 0; j < fil_info.shape[1]; j++) {
                int idx = ((_i + i) * img_info.shape[1] + j + _j) * 3;
                ret += fil[i * fil_info.shape[1] + j] * (int32_t)img[idx + off];
            }
        }
        max = max > ret ? max : ret;
        min = min < ret ? min : ret;
        return ret;
    };
    //main part
    for (int i = border_t; i < img_info.shape[0] - border_b; i++) {
        for (int j = border_l; j < img_info.shape[1] - border_r; j++) {
            int idx = (i * img_info.shape[1] + j) * 3;
            buf[idx + 0] = mul(i - border_t, j - border_l, 0);
            buf[idx + 1] = mul(i - border_t, j - border_l, 1);
            buf[idx + 2] = mul(i - border_t, j - border_l, 2);
        }
    }
    //
    T _max = ~0;
    float _factor = (_max + 0.0) / (max - min);
    if (_NOR_FACTOR == nor_method) {
        float sum = 0;
        for (int i = 0; i < fil_info.shape[0] * fil_info.shape[1]; i++) sum += fil[i];
        _factor = 1 / sum;
    }
    
    for (int i = border_t; i < img_info.shape[0] - border_b; i++) {
        for (int j = border_l; j < img_info.shape[1] - border_r; j++) {
            int idx = (i * img_info.shape[1] + j) * 3;

            if (_NOR_TRUMC == nor_method) {
                ptr[idx + 0] = buf[idx + 0] > _max ? _max : (buf[idx + 0] < 0 ? 0 : buf[idx + 0]);
                ptr[idx + 1] = buf[idx + 0] > _max ? _max : (buf[idx + 1] < 0 ? 0 : buf[idx + 1]);
                ptr[idx + 2] = buf[idx + 0] > _max ? _max : (buf[idx + 2] < 0 ? 0 : buf[idx + 2]);
            }
            else if (_NOR_FACTOR == nor_method) {
                ptr[idx + 0] = _factor * buf[idx + 0];
                ptr[idx + 1] = _factor * buf[idx + 1];
                ptr[idx + 2] = _factor * buf[idx + 2];
            }
            else if (_NOR_LINEAR == nor_method) {
                ptr[idx + 0] = _factor * (buf[idx + 0] - min);
                ptr[idx + 1] = _factor * (buf[idx + 1] - min);
                ptr[idx + 2] = _factor * (buf[idx + 2] - min);
            }
            else {
                ptr[idx + 0] = buf[idx + 0];
                ptr[idx + 1] = buf[idx + 1];
                ptr[idx + 2] = buf[idx + 2];
            }
            
        }
    }
    delete buf;
    return res;
}

template<typename T>
py::array_t<T> rgb_2_gray(py::array_t<T>& img_rgb) {
    py::buffer_info img_info = img_rgb.request();
    if (img_rgb.ndim()!=3 && img_info.shape[2] != 3) throw std::runtime_error("RGB image must has 3 channels!");

    
    auto res = py::array_t<T>(img_info.size);
    //transfer
    res.resize({img_info.shape[0], img_info.shape[1], img_info.shape[2]});
    auto res_info = res.request();
    auto ptr = (T*)res_info.ptr, img = (T*)img_info.ptr;
    
    //main part
    for (int i = 0; i < img_info.shape[0]; i++) {
        for (int j = 0; j < img_info.shape[1]; j++) {
            int idx = (i * img_info.shape[1] + j) * 3;
            uint32_t sum = 0;
            sum += img[idx];
            sum += img[idx + 1];
            sum += img[idx + 2];
            ptr[idx + 0] = sum / 3;
            ptr[idx + 1] = sum / 3;
            ptr[idx + 2] = sum / 3;
        }
    }

    return res;
}

//m :  pybind11::module_
PYBIND11_MODULE(mycv, m) {

    m.doc() = "pybind11 my computer version module";

    // Add bindings here
    m.def("version", []() {
        return "version 1.0";
    });

    //绑定函数
    m.def("add_arrays", &add_arrays<float>, "Add two NumPy arrays");

    m.def("mediablur", &mediablur<uint8_t>, "mediablur");
    m.def("rgb_filter", &rgb_filter<uint8_t, float>, "filter rgb image, type: uint8/uint16");
    m.def("rgb_nor_filter", &rgb_nor_filter<uint8_t, float>, "filter rgb image, type: uint8/uint16");

    m.def("rgb_2_gray", &rgb_2_gray<uint8_t>, "filter rgb image, type: uint8/uint16");

    //关键字参数
    m.def("sub", [](int i, int j) {
        return i - j;
    }, "A function which sub two num", py::arg("i") = 1, py::arg("j") = 2);

    //导出变量
    py::object what = py::cast("version 1.0");
    m.attr("version") = what;
    m.attr("num") = 123;

    m.attr("nor_method_trunc") = _NOR_TRUMC; 
    m.attr("nor_method_linear") = _NOR_LINEAR;
    m.attr("nor_method_factor") = _NOR_FACTOR;
}
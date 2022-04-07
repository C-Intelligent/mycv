#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

//m :  pybind11::module_
PYBIND11_MODULE(example, m) {

    m.doc() = "pybind11 example module";

    // Add bindings here
    m.def("foo", []() {
        return "Hello, World!";
    });

    //绑定函数
    m.def("add", &add, "A function which adds two numbers");

    //关键字参数
    m.def("sub", [](int i, int j) {
        return i - j;
    }, "A function which sub two num", py::arg("i") = 1, py::arg("j") = 2);

    //导出变量
    py::object what = py::cast("Hello pybind!");
    m.attr("what") = what;
    m.attr("num") = 123;
}
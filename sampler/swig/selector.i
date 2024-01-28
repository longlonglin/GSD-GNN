%module selector
%{
    #include "selector.hpp"
%}

%include "selector.hpp"
%include <std_vector.i>
%template(IntVector) std::vector<int>;
%template(DoubleVector) std::vector<double>;
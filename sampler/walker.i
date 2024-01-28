%module walker
%{
    #define SWIG_FILE_WITH_INIT
    #include "walker.hpp"
%}

%include "walker.hpp"
%include <std_vector.i>

%template(UIntVector) std::vector<unsigned int>;
%template(FloatVector) std::vector<float>;
%template(IntVector) std::vector<int>;
%template(SIntVector) std::vector<size_t>;
%begin
%{
#define SWIG_PYTHON_CAST_MODE
%}

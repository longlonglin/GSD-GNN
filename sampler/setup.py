from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super(BuildExt, self).build_extensions()

walker_module = Extension('_walker',
                           sources=['walker_wrap.cxx', 'walker.cpp'],
                           extra_compile_args=['-O3', '-fopenmp', '-g'],
                           extra_link_args=['-O3', '-fopenmp', '-g']
                           )


setup (name = 'walker',
       version = '0.1',
       cmdclass={'build_ext': BuildExt},
       ext_modules = [walker_module],
       py_modules = ["walker"],
       )

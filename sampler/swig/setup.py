from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super(BuildExt, self).build_extensions()

selector_module = Extension('_selector',
                           sources=['selector_wrap.cxx'],
                           extra_compile_args=['-O3', '-fopenmp'],
                           extra_link_args=['-O3', '-fopenmp']
                           )


setup (name = 'selector',
       version = '0.1',
    #    cmdclass={'build_ext': BuildExt},
       ext_modules = [selector_module],
       py_modules = ["selector"],
       )

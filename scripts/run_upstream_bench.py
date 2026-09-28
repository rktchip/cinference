import torch  # first: let torch load its own DLLs its way
import ctypes
import runpy
import sys

_TORCH_LIB = "C:/src/tabby-exl3/.venv/Lib/site-packages/torch/lib"
for _dll in ("cublas64_13.dll",):
    try:
        ctypes.WinDLL("%s/%s" % (_TORCH_LIB, _dll))
    except Exception as e:
        print("preload %s: %s" % (_dll, e))
sys.path.insert(0, "C:/src/exl151-stock/.venv/Lib/site-packages")
sys.argv = ["upstream_exl3_gemv_bench.py"] + sys.argv[1:]
runpy.run_path("C:/Users/chipw/upstream_gemv_harness/upstream_exl3_gemv_bench.py",
               run_name="__main__")

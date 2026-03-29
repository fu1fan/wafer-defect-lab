import importlib
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def patch_pandas_for_legacy_pickle():
    """
    给新版本 pandas 打补丁，让它能加载旧 pickle 中引用的老模块路径。
    只在读取 WM-811K 等旧数据前调用即可。
    """
    # 旧代码里常见的一些路径：pandas.indexes, pandas.core.indexes.numeric 等
    # 这里用现有的 Index / indexes 子模块来“顶上”
    try:
        import pandas.core.indexes as core_indexes  # type: ignore[attr-defined]
    except Exception:
        core_indexes = None

    # 映射表：老路径 -> 现有模块
    alias_map = {}

    # pandas.indexes.* 早期是一个子包，现在大多被并入 pandas.core.indexes
    if core_indexes is not None:
        alias_map["pandas.indexes"] = core_indexes
        alias_map["pandas.core.indexes"] = core_indexes

        # 某些旧 pickle 会引用这个模块路径；当前 pandas 未必仍暴露它。
        try:
            numeric_indexes = importlib.import_module("pandas.core.indexes.numeric")
        except ImportError:
            numeric_indexes = None

        if numeric_indexes is not None:
            alias_map["pandas.core.indexes.numeric"] = numeric_indexes

    # 把 alias_map 写入 sys.modules
    for old_name, module in alias_map.items():
        if old_name not in sys.modules:
            sys.modules[old_name] = module


def read_legacy_pickle(path: str | Path) -> pd.DataFrame | pd.Series:
    """
    读取 Python 2 / 旧版 pandas 产出的 pickle 文件。

    WM-811K 的 `LSWMD.pkl` 既会引用旧 pandas 模块路径，也可能包含
    需要 `latin1` 解码的 Python 2 字符串，因此这里统一封装兼容逻辑。
    """
    patch_pandas_for_legacy_pickle()

    with Path(path).open("rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean",
                category=np.exceptions.VisibleDeprecationWarning,
            )
            return pickle.load(handle, encoding="latin1")

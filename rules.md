## import rules

1) 推荐使用```from X import (XX, YY)```
2) ```__init__.py```建议不要保护任何信息。或者，把同一个package中的所有module导入，便于其他库使用
3) 推荐使用```from .X import XX```的形式，尤其是同一个package中的东西
4) 对于运行某个的脚本（例如package/X.py的），由于可能存在相对引用（```from .X import XX```），因此采用```python -m pakcage.X```的方式运行
5) 禁止使用```sys.path.append```
6) 推荐使用```python -m dataset.codesearchnet.codesearchnet```。这种方式可以把所有的文件路径都考虑进去
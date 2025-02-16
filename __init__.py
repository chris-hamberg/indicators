import sys, os


fpath       = os.path.abspath(__file__)
package_dir = os.sep.join(fpath.split(os.sep)[:-1])
sys.path.append(package_dir)


sys.modules["indicators"] = sys.modules[__name__]


from indicators.interface import Indicators

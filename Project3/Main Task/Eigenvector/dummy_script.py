import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
from functions import save_fig

plt.close("all")
plt.figure()


# NHL = 1
NHN = [25, 50, 100, 200]
v_min = [0.19533614198003013, 0.192312202785383, 0.053849878758502646, 0.14406367514631976]
v_max = [0.09277630175196094, 0.08431447822305149, 0.09498940279788033, 0.1324277936197065]
plt.title("MAE for NHL= 1")
plt.xlabel("Number of Hidden Nodes"); plt.ylabel("Mean Absolute Error")

# NHL = 2 NHN = [100]
NHN = [10, 25, 50, 100]
v_min = [0.012077997619313718, 0.023961336681811368, 0.057535352419362334, 0.15890943548648986]
v_max = [0.010722559242178095, 0.054327705928848856, 0.08339735193946111, 0.12779114634006553]
plt.title("MAE for NHL= 2")
plt.xlabel("Number of Hidden Nodes in the Second Layer"); plt.ylabel("Mean Absolute Error")

NHN = [3, 6, 10, 20]
v_min = [0.007556156125907654, 0.012945225535853583, 0.020930489488280135, 0.019101733201842202]
v_max = [0.006283296539778366, 0.01790058258830299, 0.05036835859282631, 0.01859256535518411]
plt.title("MAE for NHL= 3")
plt.xlabel("Number of Hidden Nodes in the Third Layer"); plt.ylabel("Mean Absolute Error")
plt.xticks(np.arange(3, 23, 2))

plt.plot(NHN, v_min, "o--", label= r"MAE $v_{\min}$")
plt.plot(NHN, v_max, "o--", label= r"MAE $v_{\max}$")
plt.legend()
save_fig("../Figures/eigvec_error_3_101.png")
plt.show()
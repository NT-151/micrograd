import numpy as np


s = [np.random.uniform(-1, 1, 10) for _ in range(3)]

old = s
print(old)

# def __call__(self, x):
#     print(f"x {x},                      weights{self.w}")
#     act = np.sum(x * self.w, initial=self.b)
#     # print(f"{x}                         is acting up {x[0]}")
#     if isinstance(x[0], Value) or isinstance(act, Value):
#         # print("its happening")
#         act = act
#     else:
#         # print("its not happening")
#         act = Value(act)
#     # print(f"{type(x[0])} of x[0]                        act {act}                    value in input  {x}")
#     return act.relu() if self.nonlin else act

for i in s:
    i -= 0.01 * 3.4


print(s)

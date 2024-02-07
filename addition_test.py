import os
from discrete_rv import DiscreteRV


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

DIGITS = 7

number1 = DiscreteRV([0.1 for _ in range(10)], 0, 9)
number2 = DiscreteRV([0.1 for _ in range(10)], 0, 9)
for _ in range(DIGITS - 1):
    number1 = number1 * 10
    number2 = number2 * 10
    number1 = number1 + DiscreteRV([0.1 for _ in range(10)], 0, 9)
    number2 = number2 + DiscreteRV([0.1 for _ in range(10)], 0, 9)

sum = number1 + number2

print(sum)
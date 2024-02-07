from discrete_rv import DiscreteRV

bitwidth = 15

number1 = DiscreteRV([0.1 for _ in range(2 ** bitwidth)], 0, 2 ** bitwidth - 1)
number2 = DiscreteRV([0.1 for _ in range(2 ** bitwidth)], 0, 2 ** bitwidth - 1)

print(number1 <= number2)
print(number1 == number2)
print(number1 != number2)
print((number1 + number2).E())

from discrete_rv import DiscreteRV

number1 = DiscreteRV([0.1 for _ in range(10)], 0, 9)
number2 = DiscreteRV([0.1 for _ in range(10)], 5, 15)

print(number1 <= number2)

from discrete_rv import DiscreteRV

number1 = DiscreteRV([0.1 for _ in range(10)], 0, 9)

print(number1 <= 5)

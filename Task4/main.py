from __future__ import annotations
import math
import random
from typing import *
import matplotlib.pyplot as plt

NGEN = 50
POP_SIZE = 100
MUT_P = 0.2
CROSS_P = 0.2
FROM = 0.5
TO = 2.5
PRECISION_BITS = 4
HOW_MANY_BITS = math.ceil(math.log(1 + (TO - FROM) * 10 ** PRECISION_BITS, 2))

def decode(x):
    return FROM + x * (TO - FROM) / (2 ** HOW_MANY_BITS - 1)

def fun_to_optimise(x):
    return ((math.e**x)*math.sin(10*math.pi*x) + 1) / x + 5

class individual:
    def __init__(self, decimal: int, eval=fun_to_optimise):
        self.eval = eval
        self.decimal = decimal

    def __str__(self):
        return f'ind({decode(self.decimal)})'

    def evaluate(self) -> float:
        return self.eval(decode(self.decimal))

    def mutate(self) -> individual:
        new_decimal = self.decimal ^ (1 << random.randint(0, HOW_MANY_BITS - 1))
        return individual(new_decimal)

    def cross(self, other: individual) -> List[individual]:
        crosspoint = random.randint(0, HOW_MANY_BITS - 1)
        mask_LSB = 2**crosspoint-1
        mask_MSB = (2 ** HOW_MANY_BITS - 1) & ~mask_LSB
        child1_decimal = (self.decimal & mask_MSB) | (other.decimal & mask_LSB)
        child2_decimal = (self.decimal & mask_LSB) | (other.decimal & mask_MSB)
        return [individual(child1_decimal), individual(child2_decimal)]

def roulette_selection(pop: List[individual]):
    evaluated = []
    for i,indiv in enumerate(pop):
        if i == 0:
            evaluated.append(indiv.evaluate())
        else:
            evaluated.append(evaluated[i-1]+indiv.evaluate())
    selected_to_return = []
    while len(selected_to_return) < POP_SIZE:
        drawn = random.uniform(0,1)*evaluated[-1]
        for i, val in enumerate(evaluated):
            if drawn <= val:
                selected_to_return.append(pop[i])
                break
    best_fit = pop[evaluated.index(max(evaluated))]
    return selected_to_return, best_fit


#initial population preparation
pop = [individual(random.randint(0, 2 ** HOW_MANY_BITS - 1)) for _ in range(POP_SIZE)]
best_indiv_ever = []

#SGA
for gen in range(NGEN):
    print(f"generation{gen+1}")
    next_pop = []
    selected, best_fit = roulette_selection(pop)
    best_indiv_ever.append(best_fit)
    while len(selected)>0:
        random_unif = random.uniform(0,1)
        if random_unif <= CROSS_P:
            first = selected.pop()
            try:
                next_pop += first.cross(selected.pop())
            except IndexError:
                next_pop.append(first)
        elif random_unif <= MUT_P + CROSS_P:
            next_pop.append(selected.pop().mutate())
        else:
            next_pop.append(selected.pop())
    pop = next_pop
    s =''
    for ind in pop:
        s+= str(ind)
    print(s)

last_evals = [ind.evaluate() for ind in pop]
best_indiv_ever.append(pop[last_evals.index(max(last_evals))])
best_indiv_ever_evals = [x.evaluate() for x in best_indiv_ever]
maximum_found = max(best_indiv_ever_evals)
x_of_maximum = best_indiv_ever[best_indiv_ever_evals.index(maximum_found)]
print(f"maximum equal to {maximum_found} found for x={x_of_maximum}")
plt.plot(best_indiv_ever_evals)
plt.show()

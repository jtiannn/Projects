from sys import argv, exit
from numpy.random import randint
from random import sample, shuffle
from simanneal import Annealer

CONSTRAINTS = None
NUM_CONSTRAINTS = 0
NUM_WIZARDS = 0

class NotBetweenness(Annealer):
    def move(self):
        # Change this!
        if randint(2) == 0:
            self.swap_random()
        else:
            self.shift_random_element()

    def swap_random(self):
        """
        Randomly swap two wizards
        """
        a, b = randint(NUM_WIZARDS), randint(NUM_WIZARDS)
        while(a == b):
            b = randint(NUM_WIZARDS)

        self.state[a], self.state[b] = self.state[b], self.state[a]

    def cut_and_swap(self):
        """
        Cuts the route into two and swaps the subroutes
        Absolutely useless do not use
        """
        a = randint(NUM_WIZARDS)
        self.state = self.state[a:] + self.state[:a]

    def swap_two_adjacent(self):
        """
        Like swap_random but for two adjacent things
        Note: this will probably not work too well
        Note Update: well its better than cut_and_swap, but not amazing...
        """
        a = randint(NUM_WIZARDS - 1)
        self.state[a], self.state[a + 1] = self.state[a + 1], self.state[a]

    def swap_random_subroutes(self):
        """
        Take two subroutes and then swap them
        Not amazing
        """
        x = sorted(sample(range(NUM_WIZARDS), 4))
        self.state = self.state[:x[0]] + self.state[x[2]:x[3]] + \
                     self.state[x[1]:x[2]] + self.state[x[0]:x[1]] + \
                     self.state[x[3]:]

    def reverse_random_subroute(self):
        """
        Take a subroute and reverse it
        It's ok, much higher overhead
        """
        a, b = randint(NUM_WIZARDS), randint(NUM_WIZARDS)
        while(a == b):
            b = randint(NUM_WIZARDS)
        if b < a:
            a, b, = b, a

        self.state[a:b] = self.state[a:b][::-1]

    def shift_random_subroute(self):
        """
        A new mutation created by Jaymo
        Takes a sub route and shifts it to another location
        It doesn't work right now (something gets hung...)
        """
        a, b = randint(NUM_WIZARDS), randint(NUM_WIZARDS)
        while(a == b):
            b = randint(NUM_WIZARDS)
        if b < a:
            a, b, = b, a

        c = randint(NUM_WIZARDS - (b - a))
        sub = self.state[a:b]
        rest = self.state[:a] + self.state[b:]
        self.state = rest[:c] + sub + rest[c:]

    def shift_random_element(self):
        """
        A new mutation created by Jaymo
        Take one element and move it to somewhere else, shifting everything
        else.
        Like shift_random_subroute but with one thing.
        Works pretty damn well
        """
        a, b = randint(NUM_WIZARDS), randint(NUM_WIZARDS)
        while(a == b):
            b = randint(NUM_WIZARDS)

        if a < b:
            self.state = self.state[:a] + self.state[a + 1 : b] + \
                        [self.state[a]] + self.state[b:]
        else:
            self.state = self.state[:b] + [self.state[a]] + \
                        self.state[b:a] + self.state[a + 1:]

    def energy(self):
        """
        Number of constraints not satisfied, energy of function.
        Taken from output_validator
        """
        m = {k: v for v, k in enumerate(self.state)}

        constraints_satisfied = NUM_CONSTRAINTS
        for c in CONSTRAINTS:
            wiz_a = m[c[0]]
            wiz_b = m[c[1]]
            wiz_mid = m[c[2]]

            if (wiz_a < wiz_mid < wiz_b) or (wiz_b < wiz_mid < wiz_a):
                pass
            else:
                constraints_satisfied -= 1
        return constraints_satisfied


def write_output(filename, solution):
    """
    Properly formats solution into file. From given Solver.py
    """
    with open(filename, "w") as f:
        for wizard in solution:
            f.write("{0} ".format(wizard))

# maps wizards to numbers and back. int-to-int comparisons take
# much less time than string-to-string
wizard_map = {}
inv_map = {}
current_index = 0

# parse input
input_file = argv[1]
with open(input_file) as f:
    NUM_WIZARDS = int(f.readline())
    NUM_CONSTRAINTS = int(f.readline())
    CONSTRAINTS = []
    for _ in range(NUM_CONSTRAINTS):
        c = f.readline().split()
        for i in [0, 1, 2]:
            if c[i] in wizard_map:
                c[i] = wizard_map[c[i]]
            else:
                wizard_map[c[i]] = current_index
                inv_map[current_index] = c[i]
                c[i] = current_index
                current_index += 1
        CONSTRAINTS.append(c)
wizards = inv_map.keys()
shuffle(wizards)

# Actually perform stuff annealing
problem = NotBetweenness(wizards)
auto_schedule = problem.auto(minutes=20)
problem.set_schedule(auto_schedule)

solution, energy = problem.anneal()

# Convert back from numbers to wizards
to_output = []
for thing in solution:
    to_output.append(inv_map[thing])

named_constraints = []
for thing in CONSTRAINTS:
    named_constraints.append([inv_map[thing[0]], inv_map[thing[1]], \
                              inv_map[thing[2]]])
CONSTRAINTS = named_constraints

# save file to output
output_file = argv[2]
try:
    # Write only if ordering is better than what was there before
    with open(output_file) as f:
        old_ordering = f.readline().split()
        m = {k: v for v, k in enumerate(old_ordering)}

        old_energy = NUM_CONSTRAINTS
        for c in CONSTRAINTS:
            wiz_a = m[c[0]]
            wiz_b = m[c[1]]
            wiz_mid = m[c[2]]

            if (wiz_a < wiz_mid < wiz_b) or (wiz_b < wiz_mid < wiz_a):
                pass
            else:
                old_energy -= 1
        print ""
        print energy, old_energy
        if energy > old_energy:
            print ""
            print "not replaced"
            exit()

except IOError:
    # File not found
    pass

write_output(output_file, to_output)

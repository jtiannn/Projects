from subprocess import call
command = ["python2", "simulatedAnnealing.py", None, None]
inputs = ["submission_4715128_input20.in", "submission_4715128_input35.in"]

for com in inputs:
	command[2] = "all_submissions/" + com
	command[3] = "outputs/" + com[:-3] + ".out"
	print command[2]
	call(command)


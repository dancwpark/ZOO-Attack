python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u #done -- 100% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -u #done -- 20% success rate
# If there is successes, see if averaging works
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -k 100 -u #done -- 20% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -k 150 -u #done -- 20% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -k 200 -u #done -- 30% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -k 500 -u #done -- 20% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -k 600 -u #done -- 20% success rate

# Change noise_act in setup_mnist_noise.py -- set noise to 0.00001
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u #done -- 100%
# Try adding both with averaging to see results
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -u #done -- 30% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -u -k 100 #done -- 30% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -u -k 200 #done -- 20% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -u -k 300 #done -- 20% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -u -k 400 #done -- 20% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -u -k 500 #done -- 30% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -u -k 600 #done -- 20% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -D -N 0.00001 -u -k 1000 #done -- 30% success rate

# Changed noise_act in setup_mnist_noise.py -- set noise to 0.01
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u #done -- 30% success rate
# Changed noise_act in setup_mnist_noise.py -- set noise to 1. --> stddev=1 does not alter the cls acc. of true data
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u #done -- 0% # Changed noise_act in setup_mnist_noise.py -- set noise to 0.1 
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u #done -- 40% success rate

# Can the adversary average away noise? noise_act is set to 0.1
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -k 200 #done -- 40% success rate
# Trying again
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -k 200 #done -- 40% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -k 100 #done -- 40% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -k 250 #done -- 40% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -k 300 #done -- 40% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -k 400 #done -- 40% success rate

# Try noise_act against white-box --- this causes the gradients to be non-differentiable.
python3 test_all.py -d mnist -a white -n 20 --solver adam -b 9 -u #done -- 25% success rate
python3 test_all.py -d mnist -a white -n 20 --solver adam -b 9 -u -k 10 #done -- 20% success rate


# How much before we can average?
# JUST POST NOISE
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -D -N 0.001 #done -- 0% success rate
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -D -N 0.001 -k 1000 #done -- 02.2% success rate
    # query count = 152,415
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -D -N 0.001 -k 3000 -m 50000 #20% success
    # query count was crazy
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -d -N 0.001 -m 50000 #%0 success
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -D -N 0.001 -k 3000 -m 50000 #20%
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -D -N 0.001 -k 3000 -m 6000 #20%
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -D -N 0.001 -k 1000 -m 4000 #20%

# ADD PRE NOISE
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u # 0.01, 60%
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u # 0.1, 40%
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u # 1, 0%
python3 test_all.py -a black -d mnist -n 10 --solver adam -b 9 -u -k 1000 -m 4000 # 1, 0%
python3 test_all.py -a nlack -d mnist -n 10 --solver adam -b 9 -u -k 1000 -m 50000 # 1, 0%
python3 test_all.py -a nlack -d mnist -n 10 --solver adam -b 9 -u -k 5000 -m 50000 # 1, 0%

# Black Box Substitution Model Attack

## Prereqs
* tensorflow
* numpy
* keras
* pillow

## Assumptions
* in progress of changing

## Files
### `l2_attack.py`
Carlini's code for CW l2 version.

### `l2_attack_black.py`
Carlini's code for CW l2 version for black box model.

### `setup_mnist.py`
Contains MNIST data. Also contains main MNIST model and 
Simplified MNIST models.

### `setup_mnist_sub.py`
Contains MNIST data. Also contains various subsitute models.

### `substitution_attack.py`
Basic subsitution attack.
TODO: Check difference with `run.py`.

### `substitution_attack_transfer_distill.py`
Test substitution attack's ability to transfer examples to a simpler
model.

### `substitution_attack_change_sub.py`
Test substitution attack with different substitution models.

### `substitution_attack_attack_simple.py`
Test substitution attack against 'complex' and 'simple' target models.

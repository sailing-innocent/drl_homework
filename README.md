# DRL_Homework

The Deep Reinforcment Learning Course Homework and Final Report

## Build & Run

This is a python project mainly depends on 

- pytorch
- gymnasium

### Prerequest

- anaconda/miniconda python > v3.10
- (recommend)install `pytest` using `pip install pytest`, otherwise you may not able to run test scripts in `/test` 
- (recommend)using CUDA to accelerate training 
- require: `pytorch`,`numpy`,`matplotlib`,`tensorboard`

### PYTEST

we use `pytest` to organize our target. You can just choose the test case that you want to run, modify its mark from `@pytest.mark.app` to `@pytest.mark.current`

And then run `pytest app -m current -s` to run the choosen testcase.


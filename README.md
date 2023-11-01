# Self_Adaptive_PID

This is the implementation of the work published in the following paper "Optimized self-adaptive PID speed control for autonomous vehicles".
The paper is freely accessible at this link: https://hal.science/hal-03442081/document 

# Steps to run the code:

This implementation requires MATLAB 2018b or a more recent version.

# a) NN-PID:

          1) Load the neural network initial weights found in (NN_weights.mat).
          2) Open the simulink file (Vehicle_model.slx) and run the simulation.
          3) Try and experiment with different learning rates, speed reference, slop, and wind disturbance.
          The file MLP_PID.py is a Python code to build an MLP network and train it to learn the data of PID optimal gains based on linear regression (This can be used for the case of NN-PID as well). Note that to run the code you have to install all the required libraries : TensorFlow, Keras...etc

# b) GA-PID:

          1) To launch the GA optimization too, just run the script in the file (GA_PID_APP).
          2) The GA tool will optimize the PID gains for different working conditions and save the data in an Excel file.
          3) Try optimizing for different working conditions, you can also try using different cost functions and changing the hyperparameters of the GA tool.

# c) GA_NN_PID_COMP:

          1) Load the neural network weights.
          2) Run the script in the file (Launch_Comparison.m) and the simulation will start.
          3) The model compares the performance of NN-PID against GA-PID

# If you find this work useful or use it in your work please cite the main paper:

Kebbati, Y., Ait-Oufroukh, N., Vigneron, V., Ichalal, D., & Gruyer, D. (2021, September). Optimized self-adaptive PID speed control for autonomous vehicles. In 2021 26th International Conference on Automation and Computing (ICAC) (pp. 1-6). IEEE.

@inproceedings{kebbati2021optimized,
  title={Optimized self-adaptive PID speed control for autonomous vehicles},
  author={Kebbati, Yassine and Ait-Oufroukh, Naima and Vigneron, Vincent and Ichalal, Dalil and Gruyer, Dominique},
  booktitle={2021 26th International Conference on Automation and Computing (ICAC)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}

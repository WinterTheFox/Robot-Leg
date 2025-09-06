# Robot-Leg
Material for a Robot Leg parameter identification, and Kinematic/Dinamic simulation in MATLAB.

The simulation model is based on the following figure:
<p align="center">
  <img width="300" alt="SingleDOF" src="https://github.com/user-attachments/assets/190bdd07-a66b-485c-ad4b-94295412fe52" />
</p>

The center of mass position of the first and second links are defined as follows:
$$
    x_1 &= x_0 + l_1 \cos(\theta_1) \\
    z_1 &= z_0 - l_1 \sin(\theta_1)
$$

$$
    x_2 &= x_0 + l_1 \cos(\theta_1) + l_2 \cos(\theta_1 + \theta_2) \\
    z_2 &= z_0 - l_1 \sin(\theta_1) - l_2 \sin(\theta_1 + \theta_2)
$$


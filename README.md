# Robot_Leg
Material for a Robot Leg parameter identification, and Kinematic/Dinamic simulation in MATLAB.

The simulation model is based on the following figure:
<p align="center">
  <img width="300" alt="SingleDOF" src="https://github.com/user-attachments/assets/190bdd07-a66b-485c-ad4b-94295412fe52" />
</p>

The modeled reaction forces are shown by the orange lines. The dotted lines are physical rails that restrict the side movement.
The Dinamics simulation can be done using a PID or an included RL model, the RL model uses a DDPG agent and a supervisor, a trained agent is included.

# ising_model


  Neils Bohr showed in 1911 that paramagnetismm, diamagnetism, and ferromanetism could not arrise purley from classical phenomenon, and thus must be a quantum mechanical effect. To solve, analytically, all of the features of this quantum mechanical systems would be a daughnting task, so Wilhelm Lenz invented the Ising model in the 1920. This assumes that the only interactions between molecules in the material are the closest neighbour spin interactions. 

Thus the hamiltonian of the material is given by :

![Hamiltonian](https://github.com/Zach-Robertson19/ising_model/blob/master/images/Hamiltonian.jpg)
  
  Our goal now, given this hamiltonian, is to solve for quantities of interest as a function of temperature. The 1-D and 2-D case have been solve analytically in the absence of a magnetic field. In the 1-D case we see no phase transiton of the magnetization, and for the 2-D case we do see a phase transtion of the magnetization. A phase transition being a change in state relative to temperatrue, meaning that below a certain temperature, T<sub>c</sub>, all the spins will align to proudce a non-zero magnetization. The 2-D case was solved analytically using Mean Field Theory, a graph of this MFT solution of the Magnetization is show below.

![Mean Field Theory Magnetization for the 2-D Ising Model](https://github.com/Zach-Robertson19/ising_model/blob/master/images/meanfield.jpg)

  The two programs provided in this repository solve, numerically, the 2-D and 3-D Ising model. They do this by utilizing randomly sampling, specifially using the Monte-Carlo method to find the stable spin-states of a spin-field at a certain temperature. This is done by finding the lowest energy/entropy configuration. 
  
  These programs have methods that will produce .csv files containing a variety of data for different values of interest. Namely the magnetization per site, specific heat per site, total enegy, Magnetic susceptability, and natural log of magnetization per site as a function of temperature. As well as other classes to plot these properties from the csv files as well as saving a plotting the actual spin fields after equilibration.

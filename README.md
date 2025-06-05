# agri-jetson-nano-guide
A guide on how to use NVIDIA Jetson Nano for precision agriculture.

Jetson Nano is a system-on-chip with high capacities of processing AI algorithms and useful for Edge Computing

![Jetson Nano](https://github.com/user-attachments/assets/708ded48-cfaa-4506-9a99-5beb8fd0f2b9)

Two documents have been provided:<br>

**· Nvidia_Jetson_Nano.pdf** : Explain how to inicialize the Jetson Nano, the computer components it has, the packages provides, the sotware system and diferent frameworks used. This guide is also explained using visual images to grasp better the information. <br>

**· Carrot_Detection_JetsonNano.pdf** : Explain the different approaches and method in similar precision agricultural projects. Explain how the AI system works in order to detect brush, form the input, the training guidelines and the results obtained. Notice this algorithm runs at the Jetson Nano, so adaption of framework versions are sensitive.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

In the repository there are also two folder which are:

**· system** -> collects all the scripts necessary to run out the model. Consists of a main.py which is donde by running the .sh file. 

**· train** -> explains how the system was trained, how many data and tranfer learning method was used. Also contains the Trt_model.py which optimizes the model to make it less heavy and faster to inference. The model version is located in the release section of the repository.

This represents the Carrot Detection System designed to be executed in the Jetson Nano. Regarding the Jetson will be equipped on a UGV, which will take photos from the carrot crops all over the cultivation area.
In this case the photos were similutad using a dataset. The main components of the system are:

🐍**UNet_TR.py** : extract the image, stores it in the Jetson Nano, makes the AI prediction and calculates the percentage of healthy and brush plant. Consists of the optimize model by using TensorRT.

🐍**Load_DB.py** : loads in MySQL database the stats and paths of the original and predicted image once the AI finishes. 

🐍**Main.py** : governor of the entire system responsible for executing each part systematically.

💻**Run_unetRT.sh**: executes the main.py using CLI arguments to execute it automatically. There's a load of a framework necessary for correct operation of the AI model.

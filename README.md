# Repository for the Nilgai binary and multigroup classification. 

This is a tutorial that implements a trained CNN to classify camera trap images.

The reposiotry includes:
* Instructions to install the appropriate dependencies
* Trained weights for a binary and multigroup classifier
* Jupyter notebooks to run inference on a sample of images
* sample of test set images

The notebooks are designed to be run locally by cloning or downloading the project. It is highly recommended to create a virtual environment. This project requries Python 3+. If you don't have python installed, go to the official [python page](https://www.python.org/downloads/release/python-380/) and follow the instructions. Be sure to "add python to PATH" when you're asked and choose python 3.8 . Once you have python, 
move on to install and create a virtual env. See below instruction on installing a virtual environment on window10 or Mac OS X. 

# Getting Started
To help with creating a virtual environment, we've listed a few steps that might help new users. Feel free to skip this section if you are already familiar with clone repos and creating virtual environments. 

1. Download the repo using the green "Code" button above and place the Nilgai-master folder in your Desktop folder.

2. Next, open your command line and `cd` into the Nilgai-master folder. 

For Windows10 and MAC OS X, type:
	
		cd Desktop\Nilgai-master


3. Install the python virtualenv module by typing (for Windows10):

		py -m pip install --user virtualenv

For MacOSX:

		python3 -m pip install --user virtualenv

4. Now create your virtual environment and give it a name. Here I'm using "my_venv_name" without the quotes, but this can be changed
to whatever you want:

Windows10:

		py -m venv my_venv_name

MacOSX:

		python3 -m venv my_venv_name

5. Activate your virtual environment

Windows10:

		my_venv_name\Scripts\activate

MacOSX:

		source my_venv_name/bin/activate

You should now have my_venv_name prepended to your command line prompt like (my_venv_name).


6. After creating your virtual environment, install the necessary dependencies using the requirements.txt file:


Windows10*:

		pip3 install -r requirements.txt


MacOSX:

		pip3 install -r requirements.txt


7. Open Jupyter Lab:
In the command line type (for window10 and MacOSX):

		jupyter lab 
         
        
A browser window should pop-up and you should see the Nilgai folders, README.txt, and requirements.txt files. 

8. Go to the /notebooks folder and run one of the ipynb files. Double click and follow the instructions. 


Troubleshooting*:

Sometimes, especially in Windows, modules won't get installed. If you have prblems with a certain package, manually install it to your virtual environement by typing:

		pip install package_name

Alternatively, copy and paste this line of code to install all the necessary libraries. Be sure you have your virtual environment activated:
		
		pip install jupyterlab tensorflow tensorflow-hub pandas numpy seaborn scikit-learn shutil jupyterlab matplotlib opencv-python






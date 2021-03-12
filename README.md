# 552_Lab_3_Code
A script to help students perform image analysis for the images collected with the ICCD camera

Steps to installing anaconda and the apprepriate packages on a Windows computer
- Install anaconda: https://docs.anaconda.com/anaconda/install/windows
  - Don't worry about the data integrity step
- Open an anaconda prompt and install the opencv package for python by entering the following command
  - `conda install opencv`
- You can now run the code from the anaconda prompt by navigating to the python file location inside the terminal. 
  Alternatively, you can install an IDE like PyCharm to run the code. Fir help setting the PyCharm interpreter see:
  https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#existing-environment
- See the video posted by Dr. Blunck and the notes inside the code for the steps to generate the data you need

Helpful tips:
- For Windows computers the path defined in the code must have `//` separating the folders. The default folder is:
  `'/home/derek/Documents/552_Lab_Data/Example_Lab_3/'` but for windows the folder path will look something like:
  `'C:\\derek\\Documents\\552_Lab_Data\\Example_Lab_3\\'`
 - You must close the pop-up plot windows before you can enter data into the terminal for the pixel positions. Make sure to save the plots you need before closing them
 - Pick and enter the points in the following order to determine the flame shape (see the video for more detail):
     1. Point at the bottom of the **left** hotspot along the centerline of the hotspot
     2. Point at the top of the **left** hotspot along the centerline of the hotspot
     3. Point at the bottom of the **right** hotspot along the centerline of the hotspot
     4. Point at the top of the **right** hotspot along the centerline of the hotspot
 - You will need the last figure that pops up and some of the data that prints in the terminal window were the numbers were input

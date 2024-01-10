# Data-Analytics-Functions
Some functions that I did for a Data Analytics Project (Regression). 


#  Install dependencies.

Most of the dependecies installation are very strighforward.

The only I would like to share my experience is with: Light Gradient-Boosting.

I used this step-by-step to install it. (https://www.geeksforgeeks.org/how-to-install-xgboost-and-lightgbm-on-macos/)

This project was done in a MAC with chip : Apple M3 Max

Operating system: MacOS Sonoma 14.1

- In an already created virtual env. Run this command "curl -O https://files.pythonhosted.org/packages/7a/6d/db0f5effd3f7982632111f37fcd2fa386b8407f1ff58ef30b71d65e1a444/lightgbm-4.2.0.tar.gz" 
or just download the file that is here (https://pypi.org/project/lightgbm/#files)

- Then run the command "tar -xzvf lightgbm-4.2.0.tar"

- After that I had to download "homebrew" because I did not have it. In an terminal instance of your mac (not in the virtual env) run this commands :
      1. (echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> /Users/reynellvillabona/.zprofile
      2. "$(/opt/homebrew/bin/brew shellenv)"
      3. This is for verification: echo $PATH | grep -q /opt/homebrew/bin && echo "Homebrew está en tu PATH" || echo "Homebrew NO está en tu PATH"
      4.Reboot your computer.

- Then go again to your virtual env terminal and run this command "brew doctor" just to make sure brew it is recognized as well in your virtual env.

- Then go inside the folder which for me was "lightgbm-4.2.0" and then run this commands:

    1.brew install cmake
    2.brew install gcc
    3.brew install libomp

- Finally you can try run this command in your jupyter notebook "import lightgbm".


#  Data Cleaning .py file.

Those are formulas for doing some data cleaning.

- Filling Nan values with close points data either using the median, the closest value or the meand and STD.
- Also a function just to check nan values in columns and how many are.

#  Data Analytics .py file (Regression problem).

After having done data cleaning and data encodification (I just used get_dummies function from pandas library). You will have three different datasets or at least in my case :). One numerical dataset, one categorical and one dataset with both.

The idea of these functions is to make in a quick way the comparison of all three datasets using 4 different models (Random Forest Regression, XGBoost, LGBM y SVM).

Hope this can help.




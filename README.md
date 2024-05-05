---ABOUT---
This repo contains the files to a financial expiriment where the author tried to predict crypto price trends by training a RNN off of price history.  The author tried many different input parameters,
 and he found a few different models that were profitable, but ultimtaely none were significantly profitable after applying brokerage trading fees (bid/ask spread, commission, etc.).

The author believes he could improve the predictive power of his models if he were to add twitter sentiment into the training data, but he never got around to this.


---GETTING STARTED---
There are two main files in this repo.  

(1) "BuildLiveDataPT/runBuildTestSequence.py"
This file creates RNN LSTM models by randomly selecting input parameters, then it tests the model and measures its profitability, and finally logs the model and profitability
 within "BuildLiveDataPT/featureCombinationResults.txt"

(2) "BuildLiveDataPT/main.py"
The author's vision for this project was to find a profitable model, then display the model against live data in a TKinter GUI.  When the model flagged a buy/sell opporunity, it would send an API trigger
 to TradingView, a website that allows users to set up automatic trading based off custom alerts. 

---SETUP---
The only tricky library to install is "TA-LIB."  If you are on MAC, running [pip install -r "requirements.txt"] should work fine.  If you are on windows x64, you will have to follow these steps 
(https://gist.github.com/mdalvi/e08115381992e42b43cad861dfe417d2):

Download and Unzip ta-lib-0.4.0-msvc.zip
Move the Unzipped Folder ta-lib to C:\
Download and Install Visual Studio Community 2015
Remember to Select [Visual C++] Feature
Build TA-Lib Library
From Windows Start Menu, Start [VS2015 x64 Native Tools Command Prompt]
Move to C:\ta-lib\c\make\cdr\win32\msvc
Build the Library nmake
Then pip3 install ta-lib

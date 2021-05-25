![BFH Banner](https://trello-attachments.s3.amazonaws.com/542e9c6316504d5797afbfb9/542e9c6316504d5797afbfc1/39dee8d993841943b5723510ce663233/Frame_19.png)
# Music-Generator
Wanna get new songs to jam to ? Here's your virtual lyricist to help you. Just type in a few words and get lyrics of Manglish songs similar to those of singer Sithara. He's still learning so keep in mind that the words you enter are in the **Wordlist.txt** file. Go crazy.

## Team members
1. Sambhu Nampoothiri G [https://github.com/sambhugit]
2. Ramgopal J [https://github.com/ramgj28]
3. Sualih Siyad [https://github.com/Sualih786]
## Team Id
BFH/recdxdGbDrmVeqTUH/2021
## Link to product walkthrough
[link to video]
## How it Works ?
The dataset contain the lyrics (malayalam lyrics written in english) of various Sithara songs. After preprocessing the dataset, it is used to train a lstm generator model and weight file is saved in 'Model' directory. The model can be inferred either by running inference.py in a python interface or by running flask API ( app.py ) which creates a web interface
## Libraries used
flask == 2.0.1  
numpy == 1.19.5  
sklearn    
tensorflow == 2.2.0    
keras == 2.5.0

**Requires python 3.6 or higher**

## How to configure
1.Clone this repository by running $ git clone https://github.com/sambhugit/Music-Generator or download the files.
2.Open terminal and then Run command "pip install -r requirements.txt" for installing the required packages.
## How to Run
1.Open terminal and run "python app.py" or "inference.py"
2.A webpage pops up. Enter few words as seed and click submitto get your lyrics.



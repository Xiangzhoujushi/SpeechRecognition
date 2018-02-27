--------------
Group Members:
--------------
Patrick Green, Trevor Rambacher, Peiyuan Tang

-----------
How to run:
-----------
From within the SpeechRecognition folder, any of the python scripts can be run with "python <script name>" 
The names are relatively self explanatory but are also listed below with the extension they are associated with. 
The scripts do assume that there is a directory "pos" that contains the text files within it in the same directory as where the script is being run. 

------------
Extension 0:
------------
(viterbi_POS_Tagger.py)
Normal viterbi results in 94.9% accuracy

------------
Extension 3:
------------
(viterbi_Case_Insensitive.py)
Whenever a word is recorded or used it is changed to lowercase only, allowing for case insensitivity. The Case Insensitive viterbi results in a 93.9% accuracy.
This loss in accuracy is most likely due to the loss of the feature/knowledge of if the word is capital or not. A capitalized word would indicate that it is 
starting a sentence or a proper noun. 

------------
Extension 4:
------------
(Vertebri_trigram.py)
It works well, but it somehow takes very long time to run. Because the nodes having 0 probability are taken into accounts. Please wait carefully. It has 96% of accuracy when I run

------------
Extension 5:
------------
(viterbi_Backoff_Tagger.py)
For this extension we used the Katz back-off method. If the count is equal to 5 or less then we replace the count with the Good-Turing count instead.
This resulted in a very slight increase to 95% accuracy. There is not a huge improvement since only small counts have been modified, but this does leave room
in the probability for unknown/unseen words. 

Running the file dose not need any arguments.
We assume the position of the testing file.


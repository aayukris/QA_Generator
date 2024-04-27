# QA_Generator - Final year Project (capstone)
As part of my final year project of my Btech Engineering in CSE with specialzation in CPS, I had developed a web site which is capable of generating question, answer and options for the given URL input by the user.

## Azure Custom Question Answering tool
This tool has been used for webscarping purpose. A very effective azure tool which has been specifically used for extracting the content. The API call is mentioned in app.py file and check the attached link for how to do API call.
https://learn.microsoft.com/en-us/rest/api/language/question-answering-projects?view=rest-language-2023-04-01
and u can access the resource through this service
https://learn.microsoft.com/en-us/rest/api/language/question-answering-projects?view=rest-language-2023-04-01

## T5 Model
The T5 base model has been used to generate question, answer and options from the generated cotent. The model cannot be uploaded here due to size restrictions. T5 model has been trained on race and squad dataset.

## Screenshots
Main web interface
![image](https://github.com/aayukris/QA_Generator/assets/72030892/00bdc5eb-37a0-4be9-b85a-70b41c0fa61d)

user input URL
![image](https://github.com/aayukris/QA_Generator/assets/72030892/ad48a901-14c9-4212-a9ec-27de6707e967)

Output
![image](https://github.com/aayukris/QA_Generator/assets/72030892/d14a0f8e-7f59-457e-a527-9097e7c0dfa7)
![image](https://github.com/aayukris/QA_Generator/assets/72030892/2537f63a-fb40-4f08-9029-5874ab05ed19)

## Scope of Improvement
Time taken to generate question answer and options are lil high. This can be optimized and also the difficulty of question of tough hard and difficult.

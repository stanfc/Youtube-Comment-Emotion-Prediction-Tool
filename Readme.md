# Youtube Comment Emotion Prediction Tool

## Introduction
Sometime it is hard to decide how crowds think about a youtube video. 
This is a tool that helps you summarize positive/negative youtube comments in to two separate word cloud. 
This tool is mainly design for Mandarin video. However, it may be easy to modify this project into your language.

## How to run
Use
```
python3 extract.py
```
to extract your train data (typically not needed if you are using my model). 
And then you can run by
```
python3 hiproject.py
```
It will automatically check if you have existed model. If there is, it will use the model.
It will output two png file: negative_wordcloud.png, positive_wordcloud.png.
If you want to change the appearance of the word cloud, you can change taiwan.jpg into the shape you want.
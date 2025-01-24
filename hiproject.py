import os
import json
import googleapiclient.discovery
import jieba
import jieba.analyse
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import random
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

def get_youtube_client(api_key):
    return googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

# get replies from comment
def get_comment_replies(parent_id, youtube):
    replies = []
    next_page_token = None

    while True:
        request = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            pageToken=next_page_token,
            maxResults=100,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response['items']:
            reply = item['snippet']['textDisplay']
            replies.append(reply)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return replies

# get all comment from video
def get_video_comments(video_id, api_key):
    youtube = get_youtube_client(api_key)
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=100,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response['items']:
            top_comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(top_comment)

            # 抓取這個頂層留言的所有回覆
            total_reply_count = item['snippet']['totalReplyCount']
            if total_reply_count > 0:
                parent_id = item['id']
                replies = get_comment_replies(parent_id, youtube)
                comments.extend(replies)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments

def segment_comments(comments):
    segmented_comments = [' '.join(jieba.lcut(comment)) for comment in comments]
    return segmented_comments

def keep_chinese_chars(text):
    pattern = re.compile(r'[^\u4e00-\u9fff]')
    chinese_text = re.sub(pattern, '', text)
    return chinese_text

def extract_keywords(text):
    text = keep_chinese_chars(text)
    return jieba.analyse.extract_tags(text, topK=5)

def train_model(train_texts, train_labels, model_save_path):
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)

    # tokenize
    print("Tokenizing the data")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })

    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10, 
    )


    print("Starting training")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    
    # save model
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    return mode
def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

# emotion prediction
def predict_emotions(model, tokenizer, comments):
    inputs = tokenizer(comments, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).tolist()

    return predictions

if __name__ == "__main__":
    print("Starting the script")
    tStart = time.time()

    api_key = "AIzaSyAYu6KqHx8E96iIM96WD5saOdF2RbfoQmE" # 替換為你的 YouTube Data API v3 密鑰
    video_url = input("請輸入影片 URL: ")
    video_id = video_url.split("v=")[1].split("&")[0]  # 確保只取到 video_id

    # fetch comment
    print("Fetching comments")
    comments = get_video_comments(video_id, api_key)

    # segmenting comment
    print("Segmenting comments")
    old_segmented_comments = segment_comments(comments)
    segmented_comments = []
    for c in tqdm(old_segmented_comments, desc="Extracting keywords"):
        strs = extract_keywords(c)
        str = ','.join(strs)
        segmented_comments.append(str)
    print(len(segmented_comments))

    
    print("Checking for existing model")
    model_save_path = './trained_model'

    if os.path.exists(model_save_path):
        # train model if no model exists
        print("Loading existing model")
        model, tokenizer = load_model(model_save_path)
    else:
        # use existed model
        train_texts = []
        train_labels = []
        with open("trainword.json", encoding='utf-8') as f:  # 指定编码为 utf-8
            data = json.load(f)
        for i in tqdm(data, desc="Processing training data"):
            for j in i:
                x = random.random()
                if x <= 1:
                    strs = extract_keywords(j[0])
                    str = ','.join(strs)
                    train_texts.append(str)
                    train_labels.append(int(j[1]))
        print("Training model")
        print("train size:", len(train_texts))
        model = train_model(train_texts, train_labels, model_save_path)
        tokenizer = BertTokenizer.from_pretrained(model_save_path)

    # emotion prediction
    print("Predicting emotions")
    predictions = predict_emotions(model, tokenizer, segmented_comments)

    # split comments into positive and negative 
    positive_comments = []
    negative_comments = []

    for idx, (comment, prediction) in enumerate(zip(segmented_comments, predictions)):
        if prediction in [1, 5]:
            positive_comments.append(comment)
        elif prediction in [2, 3, 4]:
            negative_comments.append(comment)

    # draw wordcloud
    def create_wordcloud(text, mask_image_path, output_image_path, font_path):
        mask = np.array(Image.open(mask_image_path))
        wordcloud = WordCloud(font_path=font_path, width=800, height=500, background_color='white', mask=mask).generate(text)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(output_image_path)
        plt.show()

    # generate positive and negative wordcloud
    font_path = "AdobeFanHeitiStd-Bold.otf"  # chinese wordcloud
    positive_text = ' '.join(positive_comments)
    negative_text = ' '.join(negative_comments)

    create_wordcloud(positive_text, 'taiwan.jpg', 'positive_wordcloud.png', font_path)
    create_wordcloud(negative_text, 'taiwan.jpg', 'negative_wordcloud.png', font_path)

    tEnd = time.time()

   

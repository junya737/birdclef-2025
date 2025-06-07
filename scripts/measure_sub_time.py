#!/usr/bin/env python
# coding: utf-8

# In[1]:


from kaggle.api.kaggle_api_extended import KaggleApi
import datetime
from datetime import timezone
import time

api = KaggleApi()
api.authenticate()

COMPETITION = "birdclef-2025"
result_ = api.competition_submissions(COMPETITION)[1]
latest_ref = str(result_)  # 最新のサブミット番号
submit_time = result_.date

status = ''

while status != 'complete':
    
    list_of_submission = api.competition_submissions(COMPETITION)
    for result in list_of_submission:
        if str(result.ref) == latest_ref:
            break
    status = result.status


    now = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
    elapsed_seconds = int((now - submit_time).total_seconds())
    minutes, seconds = divmod(elapsed_seconds, 60)

    if status == 'complete':
        print('\r', f'run-time: {minutes} min {seconds} sec, LB: {result.publicScore}')
    else:
        print('\r', f'elapsed time: {minutes} min {seconds} sec', end='')
        
        time.sleep(10)


# 


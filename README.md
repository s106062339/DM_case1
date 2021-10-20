Overview
---
The Task is to analyze the Electronic medical records(EMR), and to do the binary text classification task to tell whether the patient is obese or not.

We have prepared the data in ./Case Presentation 1 Data.

Under the folder, there are 3 sub-folder (Train_Textual, Test_Intuitive, Validation) which contain 400, 400, 50 Electronic medical records.

We've mentioned that the label of train data is Textual-based. That is, you can find some keyword(obese, obesity...) in the EMR.

The test data is Intuitive-based. Although there is not obvious keyword in the EMR, there's still a label given by medical expert.

Our pipeline
---
    
We use both "keyword" if-else algorithm and Bert model to solve this problem. Below is out flowchart and how we final produce our submission.csv

![](https://i.imgur.com/L3mLeYo.png)
![](https://i.imgur.com/Qt0a6gD.png)

For the keyword, we use Bag-of-words algorithm to analyze the training data, and choose [obese, obesity, sleep apnea, nasal, hypoxia, obstructive, OSA, morbid obese] as our final keyword.

For the Bert model, we finally use only 2 bert-model(Train on "physical exam" data , and train on "hospital course" data) to produce the result. More detail about model: we use 'bert-base-uncased’ pre-trained version which contained 12-layer, 110M parameter. And finetune in our data for 15 epoch.We use the Adam as the optimizer and the lr = 0.00001.

![](https://i.imgur.com/6m5MKrd.png)

Prerequisite
---
    OS:Linux
     environment: 
        
        Python 3.6.9
        
        torch == 1.9.1 
        
        CUDA version == 10.0
        
        we also prepared the requirements for quick environment setup.
    
    It is recommend to have at least 1 GPU to run the experiment.
In "requirement.txt", we have:

    torch==1.9.1
    folium==0.2.1
    transformers==4.11.3
    tqdm==4.62.3
    boto3==1.18.64
    requests==2.26.0
    regex==2021.10.8
    sklearn
    csv

Usage
---
git clone this page to the local
```git=
git clone https://github.com/s106062339/DM_case1.git
```

use the virtualenv to create the virtual environment:
```bash=
virtualenv -p /usr/bin/python3 myenv
source myenv/bin/activate
pip install -r requirements.txt
```

Then we can start to prepare the data by running "prepare_data.py" , you need to modify the variable 'Data_dir' to meet your Case Presentation 1 Data file.

In this file, we also need to set the variable 'Search_Len' as any value of 100, 300, 500, 1000, for example, to decide the length of text to be chosen from the original case file.

The new created data will be saved in ./Case Presentation 1 Data/{target}_{p,h}_{Search_Len}

So, if we want to run the experiment in Search_Len = 500, we will create 6 sub-folder under ./Case Presentation 1 Data:

:::info
./Train_p_500
        ./Train_h_500
        ./Test_p_500
        ./Test_h_500
        ./Validation_p_500
        ./Validation_h_500
:::

:bulb:Run the command below to execute     
```bash=
python prepare_data.py
```

For the file "Bert_obesity_classifier.py": 

You need to set the Data_dir to meet yor Case Presentation 1 Data file.
Please name the variable 'csv_name', which is the output csv file name stored in ./Submission folder, and remember to set the variable 'Search_Len' to run the experiment.

If you encounter the CUDA Out of Memory problem, please try to half the batch size, and we can run "Bert_obesity_classifier.py" to train the model.
We will save the P and H model in ./model which can easy reproduce our result.

:bulb:Run the command below to execute     
```bash=
CUDA_VISIBLE_DEVICES=0 python Bert_obesity_classifier.py
```

Hyperparameters
---
In our model, we use the pre-train model, so there are not too much hyperparameters to set, we basicly follow the architecture and the parameter value. The only one we can adjust is learning rate, which we give the initial value 0.00001

Result:
---
We have no validation label, so we show our kaggle leaderboard to present our performance. We have achieved 0.65714 f1_score on validation set, which tied for second place.

![](https://i.imgur.com/Jq33lhv.png)

Also, we give the training and testing Acc, Loss, and f1_score to present our performance, we can achieve higher 0.85 f1_score on testing set. We find that when Search_Len is 500 or 1000, the model can have better performance. 

![](https://i.imgur.com/bJjeJld.png)
![](https://i.imgur.com/tasCwrA.png)

However, we can see that there is still an overfitting problem. So,maybe you can try more effort on data pre-precessing to make more clean data or try "Bert-large_uncased" model to train.
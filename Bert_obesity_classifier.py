import os
import csv
import random
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

###### setting ######
Data_dir = "/home/cjho/DM_case1/Case Presentation 1 Data"
csv_name = "1019_04.csv"
Search_Len = 300
EPOCHS = 15
BATCH_SIZE = 16
###### setting ######

NUM_LABELS = 2
model_version = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_version)

model_path = "/home/cjho/DM_case1/Model"
submission_folder =  "/home/cjho/DM_case1/Submission"

Validation_folder = os.path.join(Data_dir, "Validation")

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(submission_folder):
    os.makedirs(submission_folder)

def get_learnable_params(module):
    return [p for p in module.parameters() if p.requires_grad]
     
def show_hight_level_module(model):

    # high-level 顯示此模型裡的 modules
    print("""name            module----------------------""")
    for name, module in model.named_children():
        if name == "bert":
            for n, _ in module.named_children():
                print(f"{name}:{n}")
        else:
            print("{:15} {}".format(name, module))

def show_model_params_num(model):

    model_params = get_learnable_params(model)
    clf_params = get_learnable_params(model.classifier)

    print(f"""
    整個分類模型的參數量：{sum(p.numel() for p in model_params)}
    線性分類器的參數量：{sum(p.numel() for p in clf_params)}
    """)

    # model.config

def get_predictions(model, dataloader, compute_acc=False):
    
    predictions = None
    gt_labels = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors, label_ids = data[:4]
            
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
            
            if gt_labels is None:
                gt_labels = label_ids
            else:
                gt_labels = torch.cat((gt_labels, label_ids))

    if compute_acc:
        acc = correct / total
        
        predictions_list = predictions.tolist()
        
        gt_labels_list = gt_labels.tolist()
        
        f1score = f1_score(gt_labels_list, predictions_list)

        return predictions, acc, f1score
    
    return predictions

class CustomDataset(Dataset):

    def __init__(self, tokenizer, path, mode):

        self.path = path
        self.mode = mode
        self.tokenizer = tokenizer
 
        self.datapath = os.path.join(Data_dir, self.path)

        self.df = os.listdir(self.datapath)
        random.shuffle(self.df)

        self.len = len(self.df)

    # @pysnooper.snoop()
    def __getitem__(self, idx):
        
        if self.mode != "Val":

            data_name = os.path.join(self.datapath, self.df[idx])
            
            id = int(self.df[idx][:-4].split("_")[2])
            
            id = torch.tensor(id)
            
            if self.df[idx].split("_")[0]== "Y":
                label = 1
            else:
                label = 0

        else:
            data_name = os.path.join(self.datapath, self.df[idx])
        
            id = int(self.df[idx][:-4].split("_")[1])
            id = torch.tensor(id)
            
            label = -1

        with open(data_name, "r") as f: content = f.read()
        
        label_tensor = torch.tensor(label)
  
        word_pieces = ["[CLS]"]
        tokens_content = self.tokenizer.tokenize(content)
        word_pieces += tokens_content + ["[SEP]"]
        len_a = len(word_pieces)

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)

        tokens_tensor = torch.tensor(ids)
        
        segments_tensor = torch.tensor( [0] + [1] * (len_a-2) + [0], dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor, id)
    
    def __len__(self):
        return self.len

def create_mini_batch(samples):
   
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    label_ids = torch.stack([s[2] for s in samples])
    ids = torch.stack([s[3] for s in samples])

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids, ids

p_trainset = CustomDataset(tokenizer=tokenizer, path = "Train_p_" + str(Search_Len), mode = "Train")
h_trainset = CustomDataset(tokenizer=tokenizer, path = "Train_h_" + str(Search_Len), mode = "Train")
p_trainloader = DataLoader(p_trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
h_trainloader = DataLoader(h_trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

p_testset  = CustomDataset(tokenizer=tokenizer, path = "Test_p_" + str(Search_Len), mode = "Test")
h_testset  = CustomDataset(tokenizer=tokenizer, path = "Test_h_" + str(Search_Len), mode = "Test")
p_testloader  = DataLoader(p_testset,  batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
h_testloader  = DataLoader(h_testset,  batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

p_valset   = CustomDataset(tokenizer, path = "Validation_p_" + str(Search_Len), mode = "Val")
h_valset   = CustomDataset(tokenizer, path = "Validation_h_" + str(Search_Len), mode = "Val")
p_valloader   = DataLoader(p_valset,   batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
h_valloader   = DataLoader(h_valset,   batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

print("p_trainset : " + str(len(p_trainset)))
print("p_testset : "  + str(len(p_testset)))
print("p_valset : "   + str(len(p_valset)))

print("h_trainset : " + str(len(h_trainset)))
print("h_testset : "  + str(len(h_testset)))
print("h_valset : "   + str(len(h_valset)) + "\n")

h_model = BertForSequenceClassification.from_pretrained(model_version, num_labels=NUM_LABELS, output_attentions=True)
p_model = BertForSequenceClassification.from_pretrained(model_version, num_labels=NUM_LABELS, output_attentions=True)

h_optimizer = torch.optim.Adam(h_model.parameters(), lr=1e-5)
p_optimizer = torch.optim.Adam(p_model.parameters(), lr=1e-5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

h_model = h_model.to(device)
p_model = p_model.to(device)

def p_train():

    max_f1score = 0
    for epoch in range(EPOCHS):
    
        running_loss = 0.0

        val_running_loss = 0.0

        p_model.train()

        for data in p_trainloader:

            tokens_tensors, segments_tensors, masks_tensors, labels, ids = [t.to(device) for t in data]

            p_optimizer.zero_grad()
            
            outputs = p_model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors, labels=labels)

            loss = outputs[0]

            loss.backward()
            p_optimizer.step()

            running_loss += loss.item()
            
        _, acc, f1score = get_predictions(p_model, p_trainloader, compute_acc=True)

        print('Train [epoch %d] loss: %.3f, acc: %.3f, f1_score: %.3f' % (epoch + 1, running_loss, acc, f1score))
        
        p_model.eval()

        for data in p_testloader:
            
            tokens_tensors, segments_tensors, masks_tensors, labels, ids = [t.to(device) for t in data]

            p_optimizer.zero_grad()
            
            outputs = p_model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors, labels=labels)

            loss = outputs[0]

            val_running_loss += loss.item()
            
        _, acc, f1score = get_predictions(p_model, p_testloader, compute_acc=True)

        if f1score > max_f1score:
            max_f1score = f1score
            torch.save(p_model, os.path.join(model_path, "P_model_" + str(Search_Len) + ".pt"))
            
        print('Test [epoch %d] loss: %.3f, acc: %.3f, f1_score: %.3f' % (epoch + 1, val_running_loss, acc, f1score))

        p_val_map = {}
        
        for data in p_valloader:
        
            data = [t.to("cuda:0") for t in data if t is not None]

            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            
            outputs = p_model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)
                    
            logits = outputs[0]
            
            _, pred = torch.max(logits.data, 1)

            for id, pred_label in zip(data[4].tolist(), pred.tolist()):

                if id not in p_val_map: 
                    p_val_map[id] = pred_label

                else:
                    if pred_label == 1:
                        p_val_map[id] = 1
        
        print(p_val_map)
        print("label 0 :", sum(value == 0 for value in p_val_map.values()), " label 1 :", sum(value == 1 for value in p_val_map.values()))
        print("\n")

def h_train():

    max_f1score = 0
    for epoch in range(EPOCHS):
        
        running_loss = 0.0

        val_running_loss = 0.0

        h_model.train()

        for data in h_trainloader:

            tokens_tensors, segments_tensors, masks_tensors, labels, ids = [t.to(device) for t in data]

            h_optimizer.zero_grad()
            
            outputs = h_model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors, labels=labels)

            loss = outputs[0]

            loss.backward()
            h_optimizer.step()

            running_loss += loss.item()
            
        _, acc, f1score = get_predictions(h_model, h_trainloader, compute_acc=True)

        print('Train [epoch %d] loss: %.3f, acc: %.3f, f1_score: %.3f' % (epoch + 1, running_loss, acc, f1score))
        
        h_model.eval()

        for data in h_testloader:
            
            tokens_tensors, segments_tensors, masks_tensors, labels, ids = [t.to(device) for t in data]

            h_optimizer.zero_grad()
            
            outputs = h_model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors, labels=labels)

            loss = outputs[0]

            val_running_loss += loss.item()
            
        _, acc, f1score = get_predictions(h_model, h_testloader, compute_acc=True)

        if f1score > max_f1score:
            max_f1score = f1score
            torch.save(h_model, os.path.join(model_path, "H_model_" + str(Search_Len) + ".pt"))

        print('Test [epoch %d] loss: %.3f, acc: %.3f, f1_score: %.3f' % (epoch + 1, val_running_loss, acc, f1score))

        h_val_map = {}
        
        for data in h_valloader:
        
            data = [t.to("cuda:0") for t in data if t is not None]

            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            
            outputs = h_model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)
                    
            logits = outputs[0]
            
            _, pred = torch.max(logits.data, 1)
            
            for id, pred_label in zip(data[4].tolist(), pred.tolist()):
                
                if id not in h_val_map: 
                    h_val_map[id] = pred_label
                else:
                    if pred_label == 1:
                        h_val_map[id] = 1

        print(h_val_map)
        print("label 0 :", sum(value == 0 for value in h_val_map.values()), " label 1 :", sum(value == 1 for value in h_val_map.values()))
        print("\n")

def make_submission():
    
    final_map = {}
    p_val_map = {}
    h_val_map = {}

    for data in p_valloader:
        
        data = [t.to("cuda:0") for t in data if t is not None]

        tokens_tensors, segments_tensors, masks_tensors = data[:3]
        
        outputs = p_model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)
                
        logits = outputs[0]
        
        _, pred = torch.max(logits.data, 1)

        for id, pred_label in zip(data[4].tolist(), pred.tolist()):

            if id not in p_val_map: 
                p_val_map[id] = pred_label

            else:
                if pred_label == 1:
                    p_val_map[id] = 1
        
    for data in h_valloader:
        
        data = [t.to("cuda:0") for t in data if t is not None]

        tokens_tensors, segments_tensors, masks_tensors = data[:3]
        
        outputs = h_model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)
                
        logits = outputs[0]
        
        _, pred = torch.max(logits.data, 1)
        
        for id, pred_label in zip(data[4].tolist(), pred.tolist()):
            
            if id not in h_val_map: 
                h_val_map[id] = pred_label
            else:
                if pred_label == 1:
                    h_val_map[id] = 1
    
    
    print(p_val_map)
    print("label 0 :", sum(value == 0 for value in p_val_map.values()), " label 1 :", sum(value == 1 for value in p_val_map.values()))
    print("\n")
    
    print(h_val_map)
    print("label 0 :", sum(value == 0 for value in h_val_map.values()), " label 1 :", sum(value == 1 for value in h_val_map.values()))
    print("\n")

    for id, label in p_val_map.items():
        final_map[id] = label
    
    for id, label in h_val_map.items():
        if id not in final_map:
            final_map[id] = label
        else:
            if label == 1:
                final_map[id] = 1

    print(final_map)
    print(len(final_map))

    final_map = keyword_classifier(final_map)

    make_csv(final_map)

def CleanData(content):

    article = content.replace("\n", " ")

    article = list(filter(bool, article.splitlines()))
    
    article = "".join(article)
    
    article = article.lower()

    return article

def keyword_classifier(final_map):

    ansDict = {}
    keyWords = ['obesity', 'morbid', 'apnea', 'nasel', 'hypoxia', 'obstructive', ' osa ']

    for file in os.listdir(Validation_folder): 
        id = int(file[3:-4])
        f = open(os.path.join(Validation_folder, file), 'r')        
        article = CleanData(f.read())
        decided = 0

        for word in keyWords:
            if word in article:
                ansDict[id] = 1
                decided = 1
                break
        
        if not decided and 'obese' in article:
            searchIndex = 0
            for numOfObese in range(article.count('obese')) :
                obeseIndex = article[searchIndex:].find('obese')
                searchIndex = obeseIndex + 5

                if ' no' in article[obeseIndex - 20 : obeseIndex] :
                    decided = 1
                    ansDict[id] = 0
                    break
                if 'male' in article[obeseIndex + 5 : obeseIndex + 25] :
                    decided = 1
                    ansDict[id] = 1
                    break

            if not decided :
                a = random.random()
                if (a > 0.8):
                    ansDict[id] = 1
                else: 
                    ansDict[id] = 0

        f.close()
        
    for id, label in ansDict.items():
        final_map[id] = label
    
    return final_map

def make_csv(final_map):
    print("=========")
    print(final_map)
    csv_path = os.path.join(submission_folder, csv_name)

    with open(csv_path,'w') as csvfile:
    
        writer = csv.writer(csvfile)

        writer.writerow(['Filename', 'Obesity'])

        for item in sorted(final_map):
            name = "ID_" + str(item) + ".txt"
            label = final_map[item]
            writer.writerow([name, str(label)])
            
p_train()
h_train()
print("=================================== Make Submission ========================================")
make_submission()

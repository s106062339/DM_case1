import os

###### setting ######
Data_dir = "/home/cjho/DM_case1/Case Presentation 1 Data"

# Test_Intuitive, Train_Textual, Validation
target = "Test_Intuitive"
half_len = 300
###### setting ######


foldername_target_map = {"Train_Textual": "Train", "Test_Intuitive": "Test", "Validation": "Validation"}

New_Data_dir_p = os.path.join(Data_dir, foldername_target_map[target] +"_p_" + str(half_len))
New_Data_dir_h = os.path.join(Data_dir, foldername_target_map[target] +"_h_" + str(half_len))

if not os.path.exists(New_Data_dir_p):
    os.makedirs(New_Data_dir_p)

if not os.path.exists(New_Data_dir_h):
    os.makedirs(New_Data_dir_h)

current_folder = os.path.join(Data_dir, target)

keyword_list = ['physical exam', 'hospital course']

def CleanData(content):

    article = content.replace("\n", " ")

    article = list(filter(bool, article.splitlines()))
    
    article = "".join(article)
    
    article = article.lower()

    return article

def GetLabel(item):
    
    gt_label = item.split("_")[0]

    if gt_label == "Y" : gt_label = 1
    else: gt_label = 0

    return gt_label


for item in sorted(os.listdir(current_folder)):
    
    gt_label = GetLabel(item)

    current_file_name = os.path.join(current_folder, item)
    
    with open(current_file_name, "r") as f:

        content = CleanData(f.read())

        have_keyword = False

        new_file_index = 1

        if 'physical exam' in content:
            
            keyword = 'physical exam'
            
            have_keyword = True

            ending_index = 0
            
            for i in range(content.count(keyword)):

                index = content[ending_index:].find(keyword)

                starting_index = index + ending_index

                ending_index = starting_index + len(keyword)
                
                new_file_name = os.path.join(New_Data_dir_p, item[:-4] + "_p_" + str(new_file_index) + ".txt")

                new_file_content = content[starting_index:ending_index+half_len]

                with open(new_file_name, "w") as nf:
                    nf.write(new_file_content)

                new_file_index += 1
        
        
        new_file_index = 1

        if 'hospital course' in content:
            
            keyword = 'hospital course'
            
            have_keyword = True

            ending_index = 0
            
            for i in range(content.count(keyword)):

                index = content[ending_index:].find(keyword)

                starting_index = index + ending_index

                ending_index = starting_index + len(keyword)
                
                new_file_name = os.path.join(New_Data_dir_h, item[:-4] + "_h_" + str(new_file_index) + ".txt")

                new_file_content = content[starting_index:ending_index+half_len]

                with open(new_file_name, "w") as nf:
                    nf.write(new_file_content)

                new_file_index += 1


        if have_keyword == False:
            print(item)
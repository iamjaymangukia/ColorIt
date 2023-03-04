import os

path = os.chdir("D:\\sdp_6th\\color")

i = 0
for file in os.listdir(path):
        new_file_name = "img{}.jpg".format(i)
        os.rename(file, new_file_name)
        
        i = i + 1

#   Mass Encoder- repeatedly calls the encoding script for each file with tweets in our data stores.
#   Dominic Burkart / dominicburkart.com / dominicburkart@nyu.edu
#   For the Moral and Emotional Contagion Projects in Van Bavel's lab at NYU.

#   To run: edit the parent and encoder datafields to the appropriate values. Parent is the folder
#   the program looks for files in and encoder is the script used to codify the files. The encoder script
#   this is designed to run with appends the scores to the end of each line of the TSV.

import os
from subprocess import call

parent = "/Volumes/Burkart/files/Research_Data_Backup/twitter_collections/prechecked_samples"
encoder = "/Volumes/Burkart/files/SEPL/MEC2_scratch/election_dicts/mass_encoder/Spectral_encoding_allDicts.py"

children = []
encoded = 0

def it():
    global parent, encoder, children, encoded
    for name in current:
        fullname = os.path.join(os.getcwd(), name)
        if os.path.isdir(fullname):
            children.append(fullname)
        elif name.endswith("_posts.tsv"):
            if name.find("_user_") > -1:
                continue
                #call("python3 "+'"'+encoder+'"'+" "+'"'+fullname+'"'+" noheader", shell=True)
            else:
                call("python3 "+'"'+encoder+'"'+" "+'"'+fullname+'"', shell=True)
            encoded += 1

os.chdir(parent)
current = os.listdir(os.getcwd())
print("beginning modeling.")
it()
while(len(children) > 0):
    curname = children.pop()
    os.chdir(curname)
    current = os.listdir(os.getcwd())
    it()
print("modeling complete. Total files modeled: "+str(encoded))

#Dict based spectral encoding of given traits based on dictionary word matching!
#Dominic Burkart

#use: take in all given .txt wordlists in the directory this file is saved infor a given
#   trait (eg emotionality) and encode a given set of tweets with the wordcount from the
#   given dictionary and the ratio of words in the tweet also in the dictionary/total #
#   of words in the tweet.

#adapted from version 6 for batch-encoding of multiple files.
#assumes that words in tweet are separated by spaces/punctuation to allow for tokenization
#no error checking for faulty input.

#using standard modules
import os
import sys
import shutil

#get filepath for data + content index
inputfiledir = sys.argv[1]
tw_content_indx = 3
loud = False
if "loud" in sys.argv:
    loud = True
    
#opens our data and output files (fails quickly if i/o or these files are broken)
indoc = open(inputfiledir, encoding = "utf-8")
outdoc= open("/Users/dominicburkart/desktop/spectral_encoding_temp_file", mode = "w", encoding = "utf-8")

#code for cleaning up strings (in dictionaries and in tweets)
punctuation = '''@!"“”#$%&'()*+,.-/:;<=>?[\]^_`{|}~'''
def clean(instring, spaces = True): #removes punctuation and double spaces, replacing them w/ single spaces
    instring.replace("\n"," ")
    instring.replace("\t"," ")
    for x in punctuation:
            instring = instring.replace(x, " ")
    if spaces:
        while instring.find("  ") > -1:
            instring = instring.replace("  ", " ")
    else:
        while instring.find(" ") > -1:
            instring = instring.replace(" ","")
    instring = instring.lower()
    return instring

#gets dictionaries
os.chdir("/Users/dominicburkart/Documents/MyPsychPrograms/SEPL/MEC2_scratch/election_dicts")
curlist = os.listdir(os.getcwd())
temp = []
wordlists = [] #will hold individual words
stemlists = [] #will hold stems (eg funn*)
listnames = [] #will hold the names of keyword files (to be used as variable names)
i = 0
for fname in curlist:
    if fname.endswith(".txt"): #new list of keywords!
        wordlists.append([])
        stemlists.append([])
        temp.append(open(fname, encoding = "utf-8").read().splitlines())
        i_of_x = 0
        for x in temp[i]:
            if temp[i][i_of_x].find("*") > -1:
                stemlists[i].append(clean(temp[i][i_of_x], spaces = False))
            else:
                wordlists[i].append(clean(temp[i][i_of_x], spaces = False))
            i_of_x += 1
        uncheckedSpace = True
        uncheckedBlank = True
        while uncheckedSpace or uncheckedBlank:
            try:
                wordlists[i].remove(" ")
            except ValueError:
                uncheckedSpace = False
            try:
                wordlists[i].remove("")
            except ValueError:
                uncheckedBlank = False
        if loud:
            print("Imported dictionary: "+fname)
        i += 1
        listnames.append(fname.split(".")[0])
if loud:
    print("\n")

#creates list of output datafield names based on wordlist file names
temp = []
for x in listnames:
    temp.append(x+"Count")
for x in listnames:
    temp.append(x+"Ratio")
listnames = temp

#removes duplicates
for x in range(len(wordlists)):
    wordlists[x] = set(wordlists[x])

first = True
#takes a line from the in data and encodes it
def findInTweet(line, wordlists):
    global first
    if first:
        if (len(line) != 18 and len(line) != 19): #19 includes geodata; 18 doesn't
##            print("incorrect length: "+str(len(line))+"\n skipping file.")
##            print("incorrect length filename: "+indoc.name)
            dot = indoc.name.index(".tsv")
            os.rename(indoc.name, indoc.name[0:dot]+"_encoded_"+indoc.name[dot:])
##            print(indoc.name[0:dot]+"_encoded_"+indoc.name[dot:])
            exit(1)
        first = False
    try:
        content = clean(line[tw_content_indx]).split(" ")
    except IndexError:
        if line == ['']:
            return
    counts = []
    ratios = []
    for x in range(len(wordlists)):
        counts.append(0) #populates number of variables (eg emotionality)
        ratios.append(0)
    for lists in wordlists: #start by grabbing words
        for word in lists:
            counts[wordlists.index(lists)] += content.count(word)
    for lists in stemlists:
        for token in content:
            for stem in lists:
                if token.startswith(stem):
                    counts[stemlists.index(lists)] += 1
                    break
    for x in range(len(counts)): #same as len(wordlists)
        ratios[x] = float(counts[x])/len(content)
    line.extend(counts)
    line.extend(ratios)
    writeout(line)

def writeout(linelist):
    first = True
    for value in linelist:
        if not first:
            outdoc.write("\t"+str(value))
        else:
            outdoc.write(str(value))
            first = False
    outdoc.write("\n")

#iterates through the input file, calling the methods to find and write output.
cnt = 0
inheader = True
if "noheader" in sys.argv:
    inheader = False
for line in indoc:
    line = line.replace("\n", "")
    linelist = line.split("\t")
    cnt += 1
    if inheader: #to copy over header to the new doc + add the new columns :)
        linelist.extend(" ")
        linelist.extend(listnames)
        writeout(linelist)
        if loud:
            print("populating output file, please wait.")
        inheader = False
    else: #to count words + ratios for each tweet and then right those values to out :)
        findInTweet(linelist,wordlists)
if loud:
    print("\nencoding complete.")
indoc.close()
outdoc.close()
shutil.copy(outdoc.name, indoc.name)
os.remove(outdoc.name)
dot = indoc.name.index(".tsv")
os.rename(indoc.name, indoc.name[0:dot]+"_encoded_"+indoc.name[dot:])



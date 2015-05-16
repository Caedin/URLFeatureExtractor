
from __future__ import division
from scipy.stats import ks_2samp
import numpy as np
import math
import re
import time
import os
import random
import ahocorasick
import sys
import urlparse
import pickle
from ctypes import cdll
from ctypes import c_int32
import collections

from multiprocessing import Process, Manager, Queue
import multiprocessing

'''
ed_lib = cdll.LoadLibrary("./edit_distance.pyd")
edit_distance = ed_lib.levenshtein
edit_distance.restype = c_int32
'''


stdeng = {}
stdeng['a'] = 0.08167
stdeng['b'] = 0.01492
stdeng['c'] = 0.02782
stdeng['d'] = 0.04253
stdeng['e'] = 0.12702
stdeng['f'] = 0.02228
stdeng['g'] = 0.02015
stdeng['h'] = 0.06094
stdeng['i'] = 0.06966
stdeng['j'] = 0.00153
stdeng['k'] = 0.00772
stdeng['l'] = 0.04025
stdeng['m'] = 0.02406
stdeng['n'] = 0.06749
stdeng['o'] = 0.07507
stdeng['p'] = 0.01929
stdeng['q'] = 0.00095
stdeng['r'] = 0.05987
stdeng['s'] = 0.06327
stdeng['t'] = 0.09056
stdeng['u'] = 0.02758
stdeng['v'] = 0.00978
stdeng['w'] = 0.02360
stdeng['x'] = 0.00150
stdeng['y'] = 0.01974
stdeng['z'] = 0.00074

domain_edit_targets = []
url_edit_targets = []
fuzzy_targets = []
fuzzy_dictionary = {}
n_grams = set()

target_lists_loaded = False


def load_targets():
	global target_lists_loaded
	global edit_distance_upper_limit
	global target_list
	
	target_list = ahocorasick.KeywordTree()
	
	with open('TargetList.txt', 'rb') as target_file:
		for line in target_file:
			line = line.strip()
			line = line.split('.')
			if len(line[0])>3:
				target_list.add(line[0])
			
	target_list.make()
	
	
	with open('domains.csv', 'rb') as domain_targets:
		for line in domain_targets:
			line = line.replace(',', '')
			line = line.strip()
			domain_edit_targets.append(line)
			if len(domain_edit_targets)>100: break
			
	with open('fullURL.csv', 'rb') as url_targets:
		for line in url_targets:
			line = line.replace(',', '')
			line = line.strip()
			url_edit_targets.append(line)
			if len(url_edit_targets)>100: break
			
	with open('whitetargets.txt', 'rb') as fuzzy_file:
		temp = []
		for line in fuzzy_file:
			x1 = re.split('[\r\n]',line)
			x2 = filter(None, x1)
			for string in x2:
				fuzzy_targets.append(string)
				temp.append(len(string))
	target_lists_loaded = True

def extract_n_gram(k, word):
	n_gram = {}
	list = []
	for i in xrange(1,k+1):
		for x in xrange(0, len(word), 1):
			gram = word[x:x+i:1]
			if len(gram) >= i:
				list.append(gram)
	
	for x in list:
		if x not in n_gram:
			n_gram[x] = 1
			if x not in n_grams:
				n_grams.add(x)
		else:
			n_gram[x]+=1

	return n_gram
	

def length(url):
	temp = url.split('/')
	domain = temp[0]
	if len(domain)==0:
		return 0.0
	return round(len(url)/len(domain),4)

def symbols(url):
	count = 0
	if '@' in url:
		count+=1
	if '-' in url:
		count+=1
	return count

def euclideanDistance(url):
	#Generate a normalized frequency of the characters in the URL.
	chars = {}
	for k in url:
		if ord(k)>=97 and ord(k)<=122:
			if k in chars:
				chars[k]+=1
			else:
				chars[k] = 1
	
	for k in chars:
		chars[k] = chars[k]/len(url)
		chars[k] = (chars[k]-stdeng[k])**2
	
	chars = chars.values()
	return round(math.sqrt(sum(chars)),4)
			
def ipFilter(url):
	temp = url.split('/')
	url = temp[0]
	
	ip = url.split('.')
	try:
		for val in ip:
			temp = int(val)
	except ValueError:
		return 0
	return 1

def numTargetWords(url):
	match = target_list.findall(url)
	count = 0
	for k in match:
		count+=1
	return count

def numTld(url):
	pos = url.find('/')
	url = url[pos:]
	
	tlds = ['.com','.net','.org','.edu','.mil','.gov','.biz','.info','.me','.cn','.co']
	count = 0
	for x in tlds:
		while x in url:
			count+=1
			url = url.replace(x, '',1)
	
	return count
	
def numPunctuation(url):
	count = 0
	punc = ['@','.','!','#','$','%','^','&','*',',',';',':',"'"]
	for char in url:
		if char in punc:
			count+=1
	return count
def numSuspicious(url):
	sus_words = ["confirm","account","secure","ebayisapi","webscr","login","signin","submit","update"]
	count = 0
	for k in sus_words:
		if k in url:
			count+=1
	return count

def ksTestValue(url):
	#Standard English
	std_eng = stdeng.values()
	
	#Normalized URL distribution
	chars = {}
	for k in url:
		if ord(k)>=97 and ord(k)<=122:
			if k in chars:
				chars[k]+=1
			else:
				chars[k] = 1
	
	for k in chars:
		chars[k] = chars[k]/len(url)
	chars = chars.values()
	
	x = np.asarray(std_eng)
	y = np.asarray(chars)
	ks_val = ks_2samp(x,y)
	
	if ks_val[0]=='nan':
		return 0
	
	return round(ks_val[0],4)

def klDivergenceValue(url):
	log2 = math.log(2)
	klDiv = 0.0;
	p1 = stdeng.values()
	
	chars = {}
	for x in range(97,123):
		chars[chr(x)] = 0.0
	
	for k in url:
		if ord(k)>=97 and ord(k)<=122:
			if k in chars:
				chars[k]+=1
			else:
				chars[k] = 1
	
	for k in chars:
		chars[k] = chars[k]/len(url)
	p2 = chars.values()
	
	for i in range(0, len(p1)):
		if (p1[i] == 0): continue
		if (p2[i] == 0.0): continue
		
		klDiv += p1[i] * math.log( p1[i] / p2[i] )
	
	
	return round(klDiv / log2,4)
	
def domainEditDistance(url):
	domain = url.split('/')
	domain = domain[0]
	min_distance = 999
	for k in domain_edit_targets:
		temp = __edit_distance(url, k, min_distance)
		if temp == -1:
			continue
		if temp<min_distance:
			min_distance = temp
			
	return min_distance

def urlEditDistance(url):
	min_distance = 999
	for k in url_edit_targets:
		temp = __edit_distance(url, k, min_distance)
		if temp == -1:
			continue
		if temp<min_distance:
			min_distance = temp
			
	return min_distance

def fuzzyTargetMatching(url, fuzzy_dictionary=None):
	INIT_URL = url
	
	if fuzzy_dictionary==None:
		fuzzy_dictionary = {}
	
	url_split = urlparse.urlparse(url)
	
	url_target = ''
	for k in range(1, len(url_split)):
		line = url_split[k]
		if k == 1:
			line = re.split('[.|/|:|\r|\n]',url_split[k])
			del line[-1]
			url_target=url_target+'.'.join(line)
			continue
		url_target=url_target+line

	url_strings = set(re.split('[.|/|:|\r|\n]',url_target)) - set(['www'])
	
	
	#Changes by Keith -- We needed to change the order of the loops to allow adding the minimum distance
	#to the dictionary by word. In addition, we now pass the local min distance to the calculation so that
	#it can return False if the distance is longer that the current minimum.
	url_min_distance = 999
	for y in url_strings:
		if y not in fuzzy_dictionary:
			local_min = 999
			for z in fuzzy_targets:
				if math.fabs(len(z)-len(y))>url_min_distance:
					continue
				distance = __edit_distance(z,y, local_min)
				if distance==-1: 
					continue
				if distance<local_min:
					local_min = distance
				if local_min == 0:
					break
			fuzzy_dictionary[y] = local_min
			if url_min_distance>local_min:
				url_min_distance = local_min
		else:
			local_min = fuzzy_dictionary[y]
			if url_min_distance>local_min:
				url_min_distance = local_min

	# End Changes

	if url_min_distance ==  999:
		with open('error_urls.csv', 'wb') as error:
			error.write(INIT_URL + '\n')
		return -1
	
	return url_min_distance


def __edit_distance(s1,s2, min_distance):
	#return edit_distance(s1,s2,min_distance)
	
	if math.fabs(len(s1)-len(s2))>min_distance:
		return -1
	if min_distance == 0:
		return 0
		
	if len(s1) > len(s2):
		s1,s2 = s2,s1
	distances = range(len(s1) + 1)
	for index2,char2 in enumerate(s2):
		newDistances = [index2+1]
		for index1,char1 in enumerate(s1):
			if char1 == char2:
				newDistances.append(distances[index1])
			else:
				newDistances.append(1 + min((distances[index1],
											 distances[index1+1],
											 newDistances[-1])))
		distances = newDistances
		
		# Changes by Keith
		current_distance = distances[-1]
		if index2<len(s1):
			current_distance = distances[index2+1]
		if current_distance>min_distance:
			return -1
		# End Changes
	
	return distances[-1]
	


	
def convert_string_to_ints(word):
	k = ''
	for x in word:
		k+=str(ord(x))
		k+='-'
	k=k[:-1]
	return k






if __name__ == "__main__":
	in_file = sys.argv[1]
	out_file = sys.argv[2]
	n_gram_max = int(sys.argv[3])
	
	start_time = time.time()
	print 'loading...'
	load_targets()
	urls = []
	with open(in_file, 'rb') as input:
		for line in input:
			line = line.replace("'", '')
			urls.append(line.split(','))
	print "--- "+str(time.time() - start_time)+ " seconds ---"
	
	print 'processing...'
	with open("tmp_file.pickle", 'wb') as tmp_file:
		c = 0
		for k in urls:
			c+=1
			status = str(k[0].strip())
			line = extract_n_gram(n_gram_max, k[1].strip())
			url_original = k[1].strip()
			url = url_original.lower()
			url = url.replace("https://", "");
			url = url.replace("http://", "");
			url = url.replace("ftp://", "");
			url = url.replace("HTTPS://", "");
			url = url.replace("HTTP://", "");
			url = url.replace("FTP://", "");
			
			tmp_val = [status]

			tmp_val.append(length(url))
			tmp_val.append(symbols(url))
			tmp_val.append(euclideanDistance(url))
			tmp_val.append(ipFilter(url))
			tmp_val.append(numTargetWords(url))
			tmp_val.append(numTld(url))
			tmp_val.append(numPunctuation(url))
			tmp_val.append(numSuspicious(url))
			tmp_val.append(ksTestValue(url))
			tmp_val.append(klDivergenceValue(url))
			tmp_val.append(domainEditDistance(url))
			tmp_val.append(urlEditDistance(url))
			tmp_val.append(fuzzyTargetMatching(url_original, fuzzy_dictionary))
			
			tmp_val.append(line)
			
			pickle.dump(tmp_val, tmp_file)
	
	print 'sorting...'
	n_grams = list(n_grams)
	n_grams.sort()
	print 'sort complete'
	reverse_search_table = {}
	for k in xrange(len(n_grams)):
		reverse_search_table[n_grams[k]] = k
	
	print 'writing to output file...'
	with open(out_file, 'wb') as output_file:
		output_file.write('@relation sparse_relation\n\n')
		class_attr = '@attribute status {legit, phishing}\n'
		for k in n_grams:
			line = '@attribute "'+convert_string_to_ints(k)+'" numeric\n'
			line = line.replace('\\"', '\\\"')
			output_file.write(line)
		output_file.write(class_attr)
		
		output_file.write('\n')
		output_file.write('@data\n')
		with open('tmp_file.pickle', 'rb') as tmp_file:
			c = 0
			for x in xrange(len(urls)):
				c+=1
				stored_data = pickle.load(tmp_file)
				status = stored_data[0]
				stored_dict = stored_data[-1]
				stored_list = []
				for x in xrange(1,14):
					tuple = (x-1, stored_data[x])
					stored_list.append(tuple)
				
				for key in stored_dict:
					tuple = (reverse_search_table[key]+14, stored_dict[key])
					stored_list.append(tuple)
				stored_list.sort()
				output = str(stored_list)
				output = output.replace('[', '{')
				output = output.replace(']', '')
				output = output.replace('(', '')
				output = output.replace(',', '')
				output = output.replace(')', ',')
				output = output + ' ' + str(len(reverse_search_table)+14) + ' '+ str(status)
				output+='}'
				
				output_file.write(output+'\n')
	os.remove('tmp_file.pickle')
	print "--- "+str(time.time() - start_time)+ " seconds ---"
	
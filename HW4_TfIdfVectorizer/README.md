# HW4- TfIdfVectorizer


```python
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

# Read Emails

Reading Order 
1. train spam (part1-part9)
2. train ham (kitchen-l)
3. test spam (part10-part12)
4. test ham (lokay-m)


```python
train_spam_folders = ["part1", "part2", "part3", "part4", "part5", "part6", "part7", "part8", "part9"]
test_spam_folders = ["part10", "part11", "part12"]

spam_dir = ".\GP"
train_spam_emails = []
train_ham_emails = []
test_spam_emails = []
test_ham_emails = []


# train spam 
for folder in train_spam_folders: 
    for root, dirs, files in os.walk(os.path.join(spam_dir, folder)):
        for name in files:
            file_path = os.path.join(root, name) 
            with open(file_path) as f:
                data = f.read()
            train_spam_emails.append(data)

# train ham 
for root, dirs, files in os.walk(os.path.join(".\kitchen-l")):
    for name in files:
        file_path = os.path.join(root, name) 
        with open(file_path, errors='surrogateescape') as f:
            data = f.read()
        train_ham_emails.append(data)

# test spam 
for folder in test_spam_folders: 
    for root, dirs, files in os.walk(os.path.join(spam_dir, folder)):
        for name in files:
            file_path = os.path.join(root, name) 
            with open(file_path) as f:
                data = f.read()
            test_spam_emails.append(data) 

# test ham 
for root, dirs, files in os.walk(os.path.join(".\lokay-m")):
    for name in files:
        file_path = os.path.join(root, name) 
        with open(file_path, errors='surrogateescape') as f:
            data = f.read()
        test_ham_emails.append(data)
```


```python
test_spam_emails
```


```python
spam = train_spam_emails + test_spam_emails
ham = train_ham_emails + test_ham_emails
train = train_spam_emails + train_ham_emails
test = test_spam_emails + test_ham_emails
all_emails = train_spam_emails + train_ham_emails + test_spam_emails + test_ham_emails
```

# TfidfVectorizer

Top 25 words in one spam message


```python
tfIdfVectorizer = TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(spam)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))
```

                     TF-IDF
    hoodia         0.433707
    3d             0.398106
    hoodiashop     0.254452
    td             0.236018
    1039           0.216853
    strong         0.175282
    webstat        0.169635
    tr             0.149445
    font           0.144737
    cactus         0.143405
    serif          0.136825
    map            0.136344
    helvetica      0.133508
    sans           0.123358
    width          0.113401
    tablets        0.107112
    diet           0.105157
    arial          0.104686
    face           0.101845
    20             0.091411
    tbody          0.091212
    pict           0.084817
    swisspresence  0.084817
    tabs           0.084143
    table          0.078602
    

Top 25 words in one ham message


```python
tfIdfVectorizer = TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(ham)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))
```

                   TF-IDF
    db           0.287360
    asset        0.251269
    sales        0.189747
    people       0.189732
    and          0.187754
    team         0.179586
    that         0.159656
    maintained   0.157894
    to           0.151978
    we           0.146984
    development  0.140541
    in           0.137500
    of           0.115195
    donahue      0.107326
    the          0.107209
    addition     0.104427
    ross         0.101706
    mitch        0.100801
    including    0.096133
    true         0.092662
    issues       0.091946
    ways         0.091544
    technical    0.088771
    cn           0.086556
    13752014     0.086127
    

# Random Forest Classifier


```python
# true positive (spam), false positive, true negative (ham) and false negatives.

Xtrain = tfIdfVectorizer.fit_transform(all_emails)[:len(train)]
ytrain = [1] * len(train_spam_emails) + [0] * len(train_ham_emails)

Xtest = tfIdfVectorizer.fit_transform(all_emails)[len(train):]
ytest = [1] * len(test_spam_emails) + [0] * len(test_ham_emails)

print(Xtrain.shape)
print(Xtest.shape)
```

    (14308, 413455)
    (5787, 413455)
    


```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
```

Confusion Matrix Results 

True Positives: 3423

True Negatives: 2359 

False Positives: 5

False Negatives: 0


```python
print("Confusion Matrix\n")
print(confusion_matrix(ytest, ypred)) 
print("\n-----------\n\nClassification Report: \n")
print(classification_report(ytest, ypred))
```

    Confusion Matrix
    
    [[2359    5]
     [   0 3423]]
    
    -----------
    
    Classification Report: 
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      2364
               1       1.00      1.00      1.00      3423
    
        accuracy                           1.00      5787
       macro avg       1.00      1.00      1.00      5787
    weighted avg       1.00      1.00      1.00      5787
    
    

# REDO: Use stop words in TfidfVectorizer


```python
stopwords = []

with open('stopwords.txt', encoding="utf-8") as f:
    data = f.read()
    stopwords = data.split()

with open('html_tags.txt', encoding="utf-8") as f:
    data = f.read()
    stopwords += data.split()

stopwords.append('enron')
```

REDO: Top 25 words in one spam message using stopwords


```python
tfIdfVectorizer = TfidfVectorizer(use_idf=True, stop_words=stopwords, analyzer='word')
tfIdf = tfIdfVectorizer.fit_transform(spam)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))
```


                     TF-IDF
    hoodia         0.467555
    3d             0.429177
    hoodiashop     0.274311
    1039           0.233778
    strong         0.188962
    webstat        0.182874
    font           0.156033
    cactus         0.154597
    serif          0.147504
    map            0.146985
    helvetica      0.143928
    sans           0.132985
    tablets        0.115472
    diet           0.113364
    arial          0.112856
    20             0.098545
    tbody          0.098331
    swisspresence  0.091437
    pict           0.091437
    tabs           0.090710
    table          0.084736
    coords         0.082775
    rect           0.079748
    shape          0.069786
    weight         0.066787
    

REDO: Top 25 words in one ham message using stopwords


```python
tfIdfVectorizer = TfidfVectorizer(use_idf=True, stop_words=stopwords, analyzer='word')
tfIdf = tfIdfVectorizer.fit_transform(ham)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))
```


                     TF-IDF
    db             0.358623
    asset          0.313581
    sales          0.236802
    people         0.236783
    team           0.224122
    maintained     0.197050
    development    0.175394
    donahue        0.133942
    addition       0.130323
    ross           0.126928
    mitch          0.125799
    including      0.119973
    true           0.115641
    issues         0.114748
    technical      0.110785
    13752014       0.107485
    competent      0.107485
    1075840807244  0.107485
    core           0.106028
    jeff           0.103877
    ctg            0.099269
    facilitates    0.099269
    require        0.096899
    incumbent      0.094462
    subset         0.092635
    

# REDO: Random Forest Classifier using stopwords


```python
# true positive (spam), false positive, true negative (ham) and false negatives.

Xtrain = tfIdfVectorizer.fit_transform(all_emails)[:len(train)]
ytrain = [1] * len(train_spam_emails) + [0] * len(train_ham_emails)

Xtest = tfIdfVectorizer.fit_transform(all_emails)[len(train):]
ytest = [1] * len(test_spam_emails) + [0] * len(test_ham_emails)

print(Xtrain.shape)
print(Xtest.shape)
```

    (14308, 412321)
    (5787, 412321)
    


```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
```

Confusion Matrix Results

True Positives: 3423

True Negatives: 2360

False Positives: 4

False Negatives: 0


```python
print("Confusion Matrix\n")
print(confusion_matrix(ytest, ypred)) 
print("\n-----------\n\nClassification Report: \n")
print(classification_report(ytest, ypred))
```

    Confusion Matrix
    
    [[2360    4]
     [   0 3423]]
    
    -----------
    
    Classification Report: 
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      2364
               1       1.00      1.00      1.00      3423
    
        accuracy                           1.00      5787
       macro avg       1.00      1.00      1.00      5787
    weighted avg       1.00      1.00      1.00      5787
    
    


```python
tfIdfVectorizer.get_stop_words()
```




    frozenset({'sec',
               'name',
               'kr',
               'lately',
               'thatve',
               'seen',
               '<area>',
               'little',
               'vc',
               '<li>',
               'important',
               'then',
               'none',
               'open',
               'looking',
               '<fieldset>',
               'look',
               'come',
               'ru',
               'md',
               'self',
               'likely',
               'since',
               'proud',
               'hopefully',
               'whoever',
               '<dd>',
               'million',
               "here's",
               'ge',
               'younger',
               'somebody',
               'her',
               'whats',
               'va',
               'om',
               'test',
               'interested',
               'rather',
               'itll',
               'higher',
               'anyone',
               'interests',
               'sup',
               "she'd",
               'seven',
               'amount',
               'cases',
               "might've",
               'ourselves',
               'opposite',
               '<em>',
               'tc',
               'af',
               'r',
               'ended',
               "haven't",
               'wherein',
               'au',
               'ar',
               'work',
               'side',
               'goes',
               'gets',
               '<figure>',
               'pr',
               "that's",
               'as',
               'net',
               'latterly',
               'bo',
               'during',
               'nor',
               'relatively',
               'downing',
               'ill',
               'gl',
               'myse”',
               'they',
               'didnt',
               'mm',
               'ba',
               "what'd",
               'thanx',
               'that',
               'someday',
               'st',
               'inc',
               'isn',
               'she',
               'uses',
               'whatve',
               'briefly',
               'ke',
               'youngest',
               'mostly',
               'undoing',
               'what',
               'is',
               'tries',
               "how'd",
               'sees',
               'fj',
               'bi',
               'didn',
               'uk',
               "isn't",
               'itse”',
               'necessarily',
               'defines',
               'showing',
               'works',
               'similarly',
               'act',
               'best',
               '<ul>',
               'greetings',
               'sides',
               'yt',
               '<!–>',
               'fifth',
               'between',
               'lt',
               'parting',
               'things',
               'usually',
               'resulted',
               'shell',
               'certain',
               'sent',
               'greater',
               'upwards',
               'through',
               '<acronym>',
               'farther',
               'generally',
               'shouldnt',
               'keep',
               'buy',
               'za',
               'got',
               "when's",
               "what've",
               'thorough',
               'point',
               "when'd",
               'of',
               "mustn't",
               'rooms',
               'evermore',
               'gt',
               '<footer>',
               'everything',
               'doesn',
               'sk',
               'cause',
               'number',
               'aren',
               'six',
               'now',
               '<html>',
               'areas',
               'etc',
               'everybody',
               'th',
               'goods',
               'just',
               'tg',
               'inner',
               '<sub>',
               'vols',
               "they'll",
               'without',
               'another',
               'pa',
               'abst',
               'possibly',
               'thus',
               'ir',
               'needs',
               'whither',
               'to',
               'detail',
               'others',
               'np',
               'do',
               'nf',
               'cd',
               'containing',
               '39',
               "you'd",
               'suggest',
               'viz',
               'tn',
               'backed',
               'hello',
               'youve',
               'mean',
               'substantially',
               'using',
               'click',
               'thru',
               'mh',
               'gw',
               'mx',
               'py',
               'related',
               'tz',
               'both',
               'hm',
               'sg',
               'tr',
               '<strike>',
               'eh',
               "'tis",
               'hereby',
               'hid',
               'meantime',
               'sure',
               'ending',
               'doubtful',
               'mk',
               'http',
               'hed',
               'along',
               'nevertheless',
               'much',
               'minus',
               'lest',
               '<tr>',
               'fill',
               'sl',
               'ki',
               'nowhere',
               'neverless',
               'enron',
               'always',
               'shant',
               'getting',
               'es',
               'therefore',
               '<progress>',
               'beings',
               'known',
               'first',
               'describe',
               "oughtn't",
               'directly',
               'than',
               'small',
               'associated',
               'ae',
               'tip',
               'bm',
               'saying',
               'eight',
               "we'd",
               'theyll',
               'fi',
               'amongst',
               'we',
               'il',
               'theyd',
               "he'd",
               'makes',
               'consequently',
               'those',
               'well',
               'backs',
               'although',
               'w',
               'part',
               'ga',
               'seeming',
               'numbers',
               'inside',
               'being',
               'allow',
               'within',
               'interest',
               'un',
               'gr',
               'mill',
               'u',
               'ml',
               'ahead',
               'back',
               'ours',
               'bn',
               'this',
               'couldn',
               'showns',
               're',
               '<big>',
               'in',
               'available',
               'did',
               'ye',
               '<meter>',
               'cs',
               'appear',
               'almost',
               '<map>',
               '<datalist>',
               'uz',
               'regardless',
               'ch',
               'wonder',
               'gmt',
               'hereupon',
               'gy',
               'dont',
               'nearly',
               'show',
               "that'll",
               'ag',
               'particular',
               'tell',
               'these',
               'ff',
               "one's",
               've',
               'date',
               'clearly',
               'can',
               'fr',
               'unfortunately',
               '<script>',
               'ago',
               'or',
               'sh',
               'ck',
               'wasn',
               'whole',
               'zero',
               'yours',
               '<q>',
               'org',
               'thereto',
               'pk',
               'are',
               'from',
               'mc',
               'towards',
               'abroad',
               'does',
               'but',
               'n',
               'refs',
               'specified',
               'herse”',
               'ought',
               'fo',
               'going',
               'began',
               'today',
               'shouldn',
               '<embed>',
               'shown',
               'bottom',
               '<font>',
               'bt',
               'various',
               'forty',
               'get',
               'sd',
               "what'll",
               'invention',
               'believe',
               '<pre>',
               'beginnings',
               'youd',
               'turns',
               'readily',
               'aside',
               'webpage',
               'miss',
               '<canvas>',
               'accordance',
               'aint',
               'successfully',
               'kh',
               'anyways',
               'turned',
               'pointing',
               'differently',
               'int',
               'ro',
               'wouldnt',
               'thing',
               'neither',
               'merely',
               'downed',
               'ableabout',
               'nos',
               'thereby',
               'except',
               'him',
               "we're",
               'kw',
               'more',
               'had',
               'obtained',
               'gm',
               'b',
               'for',
               'here',
               'inasmuch',
               'anything',
               'tj',
               'mrs',
               'ls',
               'com',
               'copy',
               'asking',
               'novel',
               'often',
               '<!DOCTYPE>',
               'comes',
               'years',
               'fm',
               'useful',
               'mp',
               'by',
               'nd',
               'value',
               'due',
               'thoroughly',
               'yourselves',
               'lower',
               'recently',
               'backward',
               'largely',
               'no',
               "weren't",
               'cf',
               'only',
               '<label>',
               '<caption>',
               'serious',
               'ii',
               '<strong>',
               'together',
               '10',
               'zr',
               'wherever',
               'width',
               'ie',
               'ad',
               'thought',
               'follows',
               'be',
               "shouldn't",
               "couldn't",
               '<time>',
               'affecting',
               'et-al',
               'shed',
               'lr',
               '<td>',
               'm',
               'been',
               'inc.',
               'said',
               '<basefont>',
               'dj',
               'unless',
               'id',
               '<source>',
               'his',
               '<del>',
               'ah',
               '<b>',
               'presents',
               'namely',
               'immediate',
               "don't",
               'weren',
               'co.',
               '<legend>',
               'research',
               'changes',
               'vu',
               'affected',
               'else',
               'bf',
               'reasonably',
               'formerly',
               '<address>',
               'bw',
               'neverf',
               'tw',
               'sc',
               'because',
               '<frame>',
               "why'll",
               'seriously',
               '<ol>',
               'Header',
               'problem',
               'mw',
               'pp',
               'je',
               'ug',
               'their',
               'specify',
               'among',
               'use',
               'whenever',
               'affects',
               'corresponding',
               'ed',
               'cannot',
               'resulting',
               'becoming',
               'adopted',
               'whod',
               'yourself',
               'ca',
               'text',
               "they're",
               '<img>',
               'won',
               'darent',
               'mn',
               'once',
               'made',
               'ups',
               'still',
               'su',
               'usefully',
               '<s>',
               'mu',
               '<applet>',
               'section',
               '<small>',
               'thick',
               'twas',
               '<select>',
               '<cite>',
               'whichever',
               'amid',
               'nl',
               'sixty',
               'clear',
               '<meta>',
               'working',
               'unlikely',
               'ring',
               'thats',
               'definitely',
               '<dl>',
               'bd',
               'means',
               'ways',
               'doing',
               'p',
               'give',
               'twice',
               'cy',
               'am',
               'wanted',
               "i'm",
               'concerning',
               "it's",
               'an',
               'away',
               'soon',
               'j',
               'que',
               "he'll",
               'non',
               'insofar',
               'see',
               'the',
               'with',
               'described',
               'o',
               '<u>',
               'whence',
               'gf',
               'fire',
               'meanwhile',
               '<span>',
               'az',
               'overall',
               'newest',
               'he',
               'five',
               'll',
               'furthermore',
               'mg',
               'third',
               'haven',
               'same',
               'q',
               'till',
               'other',
               'while',
               '<noscript>',
               'sj',
               'me',
               'hasnt',
               'myself',
               'hk',
               'effect',
               'i',
               'entirely',
               '<title>',
               'whatll',
               'smallest',
               'added',
               'saw',
               'two',
               "you're",
               'fifteen',
               'uy',
               'fk',
               'thinks',
               "won't",
               'longer',
               'seems',
               'kind',
               'computer',
               'nonetheless',
               'no-one',
               "'twas",
               'like',
               'eg',
               'whereafter',
               'please',
               'went',
               'ua',
               '<var>',
               'y',
               'jo',
               'whose',
               'pg',
               'appreciate',
               '<blockquote>',
               'ok',
               'taking',
               'also',
               'indicates',
               'wish',
               'think',
               'ninety',
               'you',
               'probably',
               "what's",
               '<ins>',
               'td',
               'seventy',
               '<tfoot>',
               'ts',
               'io',
               'whereby',
               'cx',
               'br',
               'sm',
               'top',
               '<code>',
               'elsewhere',
               'on',
               "you've",
               'g',
               'nz',
               'yet',
               '<mark>',
               'hr',
               'pf',
               '<textarea>',
               'dz',
               'able',
               'quite',
               'sensible',
               "there's",
               "i'll",
               'group',
               'thatll',
               'tk',
               'appropriate',
               'cant',
               'nr',
               'ord',
               'unto',
               'points',
               'pt',
               '<wbr>',
               'indeed',
               'therein',
               'forever',
               'long',
               'pn',
               'cz',
               'presumably',
               'whereupon',
               'l',
               'against',
               'system',
               'worked',
               'kp',
               'parted',
               'mustnt',
               'opens',
               'sometimes',
               'provides',
               'apart',
               "would've",
               'puts',
               'mz',
               'al',
               'itself',
               'heres',
               'making',
               'hadnt',
               '<audio>',
               'trying',
               'longest',
               'so',
               'web',
               "who's",
               'new',
               "wouldn't",
               'jp',
               'came',
               'really',
               'bb',
               'right',
               'herein',
               'general',
               'tv',
               '<link>',
               'unlike',
               'hundred',
               'x',
               'place',
               'early',
               'thin',
               'either',
               'later',
               'further',
               "they've",
               'lu',
               'currently',
               'never',
               'cl',
               'nay',
               'hereafter',
               'microsoft',
               'wells',
               '<tbody>',
               'bs',
               'kn',
               "there've",
               'begin',
               'good',
               'fifty',
               'wouldn',
               'ng',
               'every',
               'co',
               'havent',
               'www',
               'beginning',
               'mv',
               'adj',
               'notwithstanding',
               'ni',
               'netscape',
               'theres',
               'us',
               'find',
               'significant',
               'er',
               'iq',
               'problems',
               'outside',
               'seeing',
               'bv',
               'amidst',
               'put',
               'asked',
               'bh',
               'my',
               'anymore',
               "mayn't",
               'ltd',
               'turn',
               'greatest',
               "must've",
               'backwards',
               '<form>',
               'wed',
               'caption',
               'obviously',
               'gone',
               'interesting',
               'de',
               'arpa',
               'ma',
               'upon',
               'own',
               'im',
               'order',
               'hu',
               'la',
               'newer',
               'seem',
               '<thead>',
               'billion',
               'significantly',
               'tm',
               'hither',
               'necessary',
               'ones',
               '<samp>',
               'about',
               'across',
               'ly',
               'amoungst',
               "you'll",
               'below',
               'vi',
               'cu',
               'young',
               'them',
               'all',
               'became',
               'man',
               'year',
               'e',
               'arise',
               'z',
               'evenly',
               'off',
               'ordering',
               'least',
               'somewhere',
               'present',
               'second',
               'wanting',
               '<body>',
               'okay',
               'zm',
               'better',
               'latter',
               'ao',
               'mainly',
               '<colgroup>',
               'pointed',
               "where'll",
               'herself',
               'besides',
               'noone',
               '<sup>',
               'grouping',
               'thereve',
               'k',
               's',
               'uucp',
               '<kbd>',
               'thanks',
               'thirty',
               'ec',
               'mo',
               "aren't",
               '<nav>',
               'accordingly',
               "c'mon",
               "how'll",
               'rd',
               'each',
               'hell',
               'certainly',
               'shes',
               'respectively',
               'nc',
               'words',
               'cr',
               'anyway',
               'run',
               'sincere',
               'would',
               'exactly',
               'downwards',
               'length',
               'ms',
               'according',
               'ever',
               'above',
               'area',
               'gu',
               'grouped',
               'wasnt',
               'displays',
               'places',
               'immediately',
               'men',
               'seemed',
               '<table>',
               'cc',
               'yu',
               'facts',
               'some',
               "could've",
               'should',
               '<main>',
               'sometime',
               'different',
               'dm',
               'approximately',
               'ran',
               'couldnt',
               'via',
               "c's",
               'eleven',
               'last',
               'despite',
               'where',
               "it'd",
               'gs',
               'hes',
               'usefulness',
               'mr',
               'toward',
               'though',
               'tf',
               'neednt',
               "she'll",
               'apparently',
               'plus',
               'past',
               'oh',
               '<h1>',
               'don',
               'needed',
               'shows',
               '<article>',
               'used',
               'auth',
               'specifying',
               "mightn't",
               'considering',
               'perhaps',
               'tis',
               'everywhere',
               "they'd",
               'aw',
               'less',
               "ain't",
               'there',
               'alone',
               'low',
               'kept',
               '<header>',
               'sy',
               '<center>',
               'who',
               'inward',
               "who'll",
               'twelve',
               'moreover',
               'maybe',
               'thou',
               'begins',
               'knows',
               "how's",
               'too',
               'hi',
               'four',
               'looks',
               't',
               '<div>',
               'youre',
               'anyhow',
               'will',
               'cmon',
               'owing',
               'indicate',
               'gd',
               'say',
               'how',
               'asks',
               'ten',
               'differ',
               'oldest',
               'showed',
               'htm',
               'big',
               'whereas',
               'stop',
               'brief',
               'bg',
               'faces',
               'cn',
               'alongside',
               'afterwards',
               'theirs',
               'et',
               ...})


